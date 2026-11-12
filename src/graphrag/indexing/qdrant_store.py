"""Qdrant hybrid store: named dense vector + BM25 sparse vector per chunk.

* Embedded on-disk mode by default (no server needed); set QDRANT_URL to use
  a server / Qdrant Cloud.
* Payload indexes on doc_id, doc_name, domain and chunk_type enable
  *pre*-filtering (filter inside HNSW search), avoiding the post-filter
  starvation where top-k is filled with chunks from the wrong document.
* Each payload stores entity_ids (foreign keys into the graph); each graph
  node stores chunk_ids. That cross-store linking is what lets the graph leg
  and the vector leg fuse.
"""

from __future__ import annotations

import logging
import uuid
from typing import Iterable

from qdrant_client import QdrantClient, models

from ..config import Settings
from ..models import Chunk
from .embeddings import SparseEncoder

log = logging.getLogger(__name__)
DENSE, SPARSE = "dense", "bm25"
_NS = uuid.UUID("6f1c1d7e-3b9a-4a43-9a3f-1f6f0c2b9a10")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


class QdrantStore:
    def __init__(self, settings: Settings, embedder) -> None:
        self.settings = settings
        self.embedder = embedder
        self.sparse = SparseEncoder()
        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        else:
            self.client = QdrantClient(path=str(settings.qdrant_path))
        self.collection = f"{settings.collection}__{_safe(embedder.name)}"
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        dim = getattr(self.embedder, "dim", 384)
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        if self.settings.qdrant_url:  # payload indexes are a no-op in embedded mode
            for field in ("doc_id", "doc_name", "domain", "chunk_type"):
                self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)

    # ------------------------------------------------------------------
    def upsert(self, chunks: list[Chunk], batch: int = 64) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            dense = self.embedder.embed_documents([c.embed_text for c in part])
            points = []
            for c, dv in zip(part, dense):
                si, sv = self.sparse.encode_doc(c.embed_text)
                points.append(
                    models.PointStruct(
                        id=point_id(c.chunk_id),
                        vector={DENSE: dv, SPARSE: models.SparseVector(indices=si, values=sv)},
                        payload=c.model_dump(),
                    )
                )
            self.client.upsert(self.collection, points=points)

    def delete_doc(self, doc_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=_filter({"doc_id": doc_id})),
        )

    # ------------------------------------------------------------------
    def dense_search(self, query: str, k: int, filters: dict | None = None) -> list[tuple[Chunk, float]]:
        res = self.client.query_points(
            self.collection, query=self.embedder.embed_query(query), using=DENSE,
            limit=k, query_filter=_filter(filters), with_payload=True,
        )
        return [(Chunk(**p.payload), p.score) for p in res.points]

    def sparse_search(self, query: str, k: int, filters: dict | None = None) -> list[tuple[Chunk, float]]:
        si, sv = self.sparse.encode_query(query)
        if not si:
            return []
        res = self.client.query_points(
            self.collection, query=models.SparseVector(indices=si, values=sv), using=SPARSE,
            limit=k, query_filter=_filter(filters), with_payload=True,
        )
        return [(Chunk(**p.payload), p.score) for p in res.points]

    def get_chunks(self, chunk_ids: Iterable[str]) -> list[Chunk]:
        ids = [point_id(c) for c in chunk_ids]
        if not ids:
            return []
        pts = self.client.retrieve(self.collection, ids=ids, with_payload=True)
        by_id = {p.payload["chunk_id"]: Chunk(**p.payload) for p in pts}
        return [by_id[c] for c in chunk_ids if c in by_id]

    def scroll(self, filters: dict | None = None, limit: int = 10_000) -> list[Chunk]:
        out, offset = [], None
        while True:
            pts, offset = self.client.scroll(
                self.collection, scroll_filter=_filter(filters), limit=256, offset=offset, with_payload=True
            )
            out.extend(Chunk(**p.payload) for p in pts)
            if offset is None or len(out) >= limit:
                return out

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)[:40]


def _filter(filters: dict | None) -> models.Filter | None:
    if not filters:
        return None
    must = []
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(value))))
        else:
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    return models.Filter(must=must) if must else None
