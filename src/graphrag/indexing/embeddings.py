"""Dense + sparse (BM25-style) encoders.

Dense:
  * fastembed (``pip install '.[embeddings]'``) -> real ONNX model, default
    BAAI/bge-small-en-v1.5 (set GRAPHRAG_DENSE_MODEL for bge-large etc.)
  * fallback: a deterministic hashed n-gram embedder (no downloads, works
    offline / air-gapped). Lower semantic quality, but keeps the pipeline
    fully functional; lexical recall is carried by the sparse leg.

Sparse:
  Term-frequency vectors over hashed token ids with BM25 TF saturation and
  document-length normalisation. Qdrant applies IDF server-side via
  ``Modifier.IDF``, which together gives BM25 scoring that stays correct as
  documents are added.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:[.\-][A-Za-z0-9_]+)*")
_STOP = set(
    "a an the of to in on for and or is are was were be been by with as at from that this these those it its "
    "into than then there their which what who whom how when where why does do did not no can could should "
    "would may might will shall about over under per via".split()
)


def tokenize(text: str) -> list[str]:
    toks = []
    for t in _WORD_RE.findall(text.lower()):
        if t in _STOP:
            continue
        toks.append(t)
        # also index sub-parts of identifiers: max_requests_per_minute -> max, requests, minute
        if "_" in t or "." in t or "-" in t:
            toks.extend(p for p in re.split(r"[_.\-]", t) if p and p not in _STOP)
    return toks


def _h(token: str, dim: int) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little") % dim


class SparseEncoder:
    """BM25 document/query encoder over a hashed vocabulary (IDF applied by Qdrant)."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_len: float = 256.0, vocab: int = 2**31 - 1) -> None:
        self.k1, self.b, self.avg_len, self.vocab = k1, b, avg_len, vocab

    def encode_doc(self, text: str) -> tuple[list[int], list[float]]:
        toks = tokenize(text)
        tf = Counter(_h(t, self.vocab) for t in toks)
        dl = len(toks) or 1
        norm = self.k1 * (1 - self.b + self.b * dl / self.avg_len)
        idx = list(tf.keys())
        vals = [(c * (self.k1 + 1)) / (c + norm) for c in tf.values()]
        return idx, vals

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        tf = Counter(_h(t, self.vocab) for t in tokenize(text))
        return list(tf.keys()), [1.0] * len(tf)


class HashEmbedder:
    """Offline dense embedder: signed feature hashing of words, bigrams and char trigrams."""

    name = "hash-384"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        toks = tokenize(text)
        feats: list[tuple[str, float]] = [(f"w:{t}", 1.0) for t in toks]
        feats += [(f"b:{a}_{b}", 0.7) for a, b in zip(toks, toks[1:])]
        for t in toks:
            padded = f"#{t}#"
            feats += [(f"c:{padded[i:i+3]}", 0.3) for i in range(len(padded) - 2)]
        for f, w in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += w if (h >> 40) & 1 else -w
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FastEmbedder:
    def __init__(self, model: str) -> None:
        from fastembed import TextEmbedding  # type: ignore

        self._model = TextEmbedding(model_name=model)
        self.name = model
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        q = getattr(self._model, "query_embed", None)
        vec = next(iter(q([text]))) if q else next(iter(self._model.embed([text])))
        return list(map(float, vec))


def build_dense_embedder(kind: str, model: str):
    if kind in ("auto", "fastembed"):
        try:
            emb = FastEmbedder(model)
            log.info("Dense embedder: fastembed %s (dim=%d)", model, emb.dim)
            return emb
        except Exception as exc:
            if kind == "fastembed":
                raise
            log.warning("fastembed unavailable (%s); using offline hashed embedder", exc)
    return HashEmbedder()
