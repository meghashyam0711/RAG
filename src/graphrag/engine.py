"""GraphRAGEngine: the facade used by the MCP server, CLI and benchmark."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from rapidfuzz import fuzz, process

from .agent.nodes import Nodes, coverage
from .agent.sandbox import run_calculation
from .agent.workflow import build_workflow
from .config import Settings
from .indexing.embeddings import build_dense_embedder
from .indexing.graph_store import build_graph_store
from .indexing.qdrant_store import QdrantStore
from .ingestion.chunker import chunk_blocks
from .ingestion.entity_resolver import EntityResolver
from .ingestion.graph_extractor import GraphExtractor
from .ingestion.layout_parser import blocks_to_markdown, parse_document
from .llm import build_llm
from .models import ENTITY_TYPES, Entity, Evidence, Relation
from .retrieval.hybrid import HybridRetriever
from .retrieval.reranker import build_reranker, rerank

log = logging.getLogger(__name__)


class GraphRAGEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or Settings()
        self._lock = threading.RLock()
        self.llm = build_llm(self.s)
        self.embedder = build_dense_embedder(self.s.embedder, self.s.dense_model)
        self.vectors = QdrantStore(self.s, self.embedder)
        self.graph = build_graph_store(self.s)
        self.resolver = EntityResolver(self.s.entity_match_threshold)
        for e in self.graph.all_entities():
            self.resolver.register(e.id, e.type, [e.name, *e.aliases])
        self.reranker = build_reranker(self.s.reranker_model)
        self.extractor = GraphExtractor(self.llm)
        self.retriever = HybridRetriever(self.vectors, self.graph, self.resolver, self.reranker, self.s)
        self.nodes = Nodes(self.retriever, self.llm, self.s)
        self.workflow = build_workflow(self.nodes)
        self.registry: dict[str, dict] = (
            json.loads(self.s.registry_path.read_text()) if self.s.registry_path.exists() else {}
        )

    # ------------------------------------------------------------------
    def capabilities(self) -> dict:
        return {
            "embedder": self.embedder.name,
            "reranker": getattr(self.reranker, "name", "?"),
            "graph": self.graph.stats().get("backend"),
            "vector_store": "qdrant-server" if self.s.qdrant_url else "qdrant-embedded",
            "llm": f"{self.llm.provider}:{self.llm.model}" if self.llm else "none (extractive mode)",
            "data_dir": str(self.s.data_dir),
        }

    def _save_registry(self) -> None:
        self.s.registry_path.write_text(json.dumps(self.registry, indent=2))

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest(self, file_path: str, domain_tag: str = "general", doc_name: str | None = None, force: bool = False) -> dict:
        t0 = time.perf_counter()
        path = Path(file_path).expanduser().resolve()
        domain = domain_tag.lower() if domain_tag.lower() in ENTITY_TYPES else "general"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        name = doc_name or path.stem
        doc_id = re.sub(r"[^\w\-]+", "_", name.lower()).strip("_")[:48] + "-" + hashlib.sha1(str(path).encode()).hexdigest()[:6]

        with self._lock:
            prev = self.registry.get(doc_id)
            if prev and prev.get("sha256") == digest and not force:
                return {**prev, "status": "unchanged (already indexed; pass force=true to re-ingest)"}
            if prev:
                self.vectors.delete_doc(doc_id)
                self.graph.delete_doc(doc_id)

            blocks, backend = parse_document(path)
            (self.s.parsed_dir / f"{doc_id}.md").write_text(blocks_to_markdown(blocks))
            chunks = chunk_blocks(blocks, doc_id=doc_id, doc_name=name, domain=domain,
                                  chunk_tokens=self.s.chunk_tokens, overlap=self.s.chunk_overlap)

            entities: dict[str, Entity] = {}
            relations: list[Relation] = []
            for chunk in chunks:
                result = self.extractor.extract(chunk)
                name_to_id: dict[str, str] = {}
                for ex in result.entities:
                    eid = self.resolver.resolve(ex.name, ex.type, doc_id)
                    name_to_id[ex.name.lower()] = eid
                    ent = entities.get(eid)
                    if ent is None:
                        ent = entities[eid] = Entity(id=eid, name=ex.name, type=ex.type, description=ex.description)
                    if ex.name not in ent.aliases and ex.name != ent.name:
                        ent.aliases.append(ex.name)
                    # Section nodes link to chunks of that section; others to chunks that mention them.
                    if chunk.chunk_id not in ent.chunk_ids and (ex.type != "Section" or ex.name == chunk.section):
                        ent.chunk_ids.append(chunk.chunk_id)
                    if doc_id not in ent.doc_ids:
                        ent.doc_ids.append(doc_id)
                    if eid not in chunk.entity_ids:
                        chunk.entity_ids.append(eid)
                for rel in result.relations:
                    s_id, t_id = name_to_id.get(rel.source.lower()), name_to_id.get(rel.target.lower())
                    if s_id and t_id and s_id != t_id:
                        relations.append(Relation(source_id=s_id, target_id=t_id, type=rel.type,
                                                  chunk_id=chunk.chunk_id, evidence=rel.evidence))

            self.vectors.upsert(chunks)
            self.graph.upsert(list(entities.values()), relations)

            rel_counts: dict[str, int] = {}
            for r in relations:
                rel_counts[r.type] = rel_counts.get(r.type, 0) + 1
            entry = {
                "doc_id": doc_id, "doc_name": name, "path": str(path), "domain": domain, "sha256": digest,
                "parser": backend, "chunks": len(chunks), "tables": sum(c.chunk_type == "table" for c in chunks),
                "pages": max((c.page_end or 0 for c in chunks), default=0) or None,
                "entities": len(entities), "relations": rel_counts,
                "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self.registry[doc_id] = entry
            self._save_registry()
        warn = (" WARNING: parsed with pypdf (no layout model) - install docling for tables/multi-column."
                if backend == "pypdf" else "")
        return {**entry, "status": "indexed" + warn, "seconds": round(time.perf_counter() - t0, 2)}

    # ------------------------------------------------------------------
    def resolve_doc_filter(self, doc_filter: str | list[str] | None) -> dict | None:
        if not doc_filter:
            return None
        wanted = [doc_filter] if isinstance(doc_filter, str) else list(doc_filter)
        ids: list[str] = []
        names = {d["doc_name"].lower(): d["doc_id"] for d in self.registry.values()}
        for w in wanted:
            if w in self.registry:
                ids.append(w)
                continue
            wl = w.lower().removesuffix(".pdf").removesuffix(".md")
            if wl in names:
                ids.append(names[wl])
                continue
            match = process.extractOne(wl, list(names.keys()), scorer=fuzz.WRatio) if names else None
            if match and match[1] >= 80:
                ids.append(names[match[0]])
            else:
                raise ValueError(f"No indexed document matches {w!r}. Indexed: {sorted(names) or 'none'}")
        return {"doc_id": ids if len(ids) > 1 else ids[0]}

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def ask(self, query: str, doc_filter: str | list[str] | None = None) -> dict:
        t0 = time.perf_counter()
        if not self.registry:
            return {"status": "abstained", "answer": "No documents are indexed yet. Call ingest_complex_doc first.",
                    "trace": []}
        filters = self.resolve_doc_filter(doc_filter)
        out = self.workflow.invoke({"query": query, "filters": filters, "trace": []}, {"recursion_limit": 40})
        final = dict(out["final"])
        final["trace"] = out.get("trace", [])
        final["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        return final

    def trace_entity(self, entity_name: str, max_hops: int = 2, doc_filter: str | None = None, limit: int = 40) -> dict:
        max_hops = max(1, min(int(max_hops), 4))
        matches = self.resolver.lookup(entity_name, limit=5, score_cutoff=70)
        if not matches:
            return {"entity": entity_name, "found": False, "message": "No matching entity in the knowledge graph."}
        filters = self.resolve_doc_filter(doc_filter)
        allowed_docs = set(filters["doc_id"] if isinstance(filters["doc_id"], list) else [filters["doc_id"]]) if filters else None
        seed = matches[0][0]
        dist, edges = self.graph.neighborhood([seed], hops=max_hops, limit=400)
        if allowed_docs:
            edges = [e for e in edges if str(e.get("chunk_id", "")).split("::")[0] in allowed_docs]
        edges.sort(key=lambda e: (e["hop"], e["type"] == "MENTIONS"))
        chunk_ids = list(dict.fromkeys(e["chunk_id"] for e in edges if e.get("chunk_id")))
        chunks = {c.chunk_id: c for c in self.vectors.get_chunks(chunk_ids[:200])}
        rows = []
        for e in edges[:limit]:
            c = chunks.get(e.get("chunk_id"))
            rows.append({"hop": e["hop"], "source": e["source"], "relation": e["type"], "target": e["target"],
                         "evidence": e.get("evidence") or "", "cited_at": c.citation if c else None})
        ent = self.graph.get(seed)
        community = getattr(self.graph, "community", lambda _x: None)(seed)
        return {
            "entity": entity_name, "found": True, "resolved_id": seed,
            "alternatives": [m[0] for m in matches[1:]],
            "type": ent.type if ent else None, "aliases": ent.aliases if ent else [],
            "documents": sorted({self.registry.get(d, {}).get("doc_name", d) for d in (ent.doc_ids if ent else [])}),
            "community": community,
            "reachable_entities": len(dist) - 1, "relations": rows, "truncated": len(edges) > limit,
        }

    def extract_table(self, topic: str, doc_name: str | None = None, compute: str | None = None, top_n: int = 3) -> dict:
        filters = self.resolve_doc_filter(doc_name) or {}
        filters = {**filters, "chunk_type": "table"}
        cands: dict[str, Evidence] = {}
        for c, _ in self.vectors.dense_search(topic, 15, filters) + self.vectors.sparse_search(topic, 15, filters):
            cands.setdefault(c.chunk_id, Evidence(chunk=c, sources=["vector"]))
        # Graph: tables in sections linked to entities named in the topic.
        for eid in self.retriever.link_query_entities(topic)[:4]:
            ent = self.graph.get(eid)
            for c in self.vectors.get_chunks((ent.chunk_ids if ent else [])[:10]):
                if c.chunk_type == "table" and (not doc_name or c.doc_id in str(filters.get("doc_id"))):
                    cands.setdefault(c.chunk_id, Evidence(chunk=c, sources=["graph"]))
        if not cands:
            return {"found": False, "message": f"No table found for {topic!r}."}
        ranked = rerank(self.reranker, topic, list(cands.values()))[:top_n]
        best = ranked[0]
        cov, missing = coverage(topic, [best])
        result = {
            "found": True,
            "verified": cov >= 0.6,
            "match_coverage": round(cov, 2),
            "unmatched_terms": missing,
            "source": best.chunk.citation,
            "chunk_id": best.chunk.chunk_id,
            "units": best.chunk.units,
            "table_markdown": best.chunk.text,
            "rows": parse_markdown_table(best.chunk.text),
            "other_candidates": [{"source": e.chunk.citation, "chunk_id": e.chunk.chunk_id,
                                  "preview": e.chunk.text.splitlines()[0][:120]} for e in ranked[1:]],
        }
        if compute:
            calc = run_calculation(compute, "user expression")
            if calc["ok"]:
                nums_in_table = {re.sub(r"[,$()%]", "", n) for n in re.findall(r"\(?\$?\d[\d,]*(?:\.\d+)?\)?", best.chunk.text)}
                operands = [re.sub(r"[,$()%]", "", n) for n in re.findall(r"\(?\$?\d[\d,]*(?:\.\d+)?\)?", compute)]
                calc["operands_found_in_table"] = {o: (o in nums_in_table) for o in operands}
                calc["all_operands_grounded"] = all(calc["operands_found_in_table"].values()) if operands else False
            result["calculation"] = calc
        return result

    def list_documents(self) -> dict:
        return {"documents": list(self.registry.values()), "graph": self.graph.stats(), "capabilities": self.capabilities()}

    def close(self) -> None:
        self.vectors.close()
        self.graph.close()


def parse_markdown_table(md: str) -> list[dict]:
    lines = [l for l in md.splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].strip().strip("|").split("|")]
    rows = []
    for l in lines[1:]:
        if re.match(r"^\s*\|?\s*:?-{2,}", l):
            continue
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        rows.append(dict(zip(header, cells)))
    return rows
