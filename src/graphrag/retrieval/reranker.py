"""Second-stage reranking.

Cross-encoder (``pip install '.[rerank]'``, default BAAI/bge-reranker-v2-m3)
when available; otherwise a lexical precision scorer that rewards exact
identifier / number / phrase matches, which is what matters most for specs
and filings.
"""

from __future__ import annotations

import logging
import math
import re

from ..indexing.embeddings import tokenize
from ..models import Evidence

log = logging.getLogger(__name__)
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


class LexicalReranker:
    name = "lexical"

    def score(self, query: str, texts: list[str]) -> list[float]:
        q_toks = tokenize(query)
        if not q_toks:
            return [0.0] * len(texts)
        q_set = set(q_toks)
        q_bigrams = set(zip(q_toks, q_toks[1:]))
        q_idents = {t for t in q_set if "_" in t or "." in t}
        q_nums = {n.replace(",", "") for n in _NUM_RE.findall(query)}
        scores = []
        for text in texts:
            toks = tokenize(text)
            tset = set(toks)
            if not toks:
                scores.append(0.0)
                continue
            coverage = len(q_set & tset) / len(q_set)
            bigram = len(q_bigrams & set(zip(toks, toks[1:]))) / max(1, len(q_bigrams))
            ident = len(q_idents & tset) / max(1, len(q_idents)) if q_idents else 0.0
            nums = {n.replace(",", "") for n in _NUM_RE.findall(text)}
            num = len(q_nums & nums) / max(1, len(q_nums)) if q_nums else 0.0
            length_pen = 1.0 / (1.0 + math.log1p(max(0, len(toks) - 400) / 400))
            scores.append((0.55 * coverage + 0.2 * bigram + 0.15 * ident + 0.10 * num) * length_pen)
        return scores


class CrossEncoderReranker:
    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder  # type: ignore

        self._m = CrossEncoder(model, max_length=512)
        self.name = model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        raw = self._m.predict([(query, t) for t in texts])
        return [1 / (1 + math.exp(-float(s))) for s in raw]


def build_reranker(model: str):
    try:
        r = CrossEncoderReranker(model)
        log.info("Reranker: cross-encoder %s", model)
        return r
    except Exception as exc:
        log.info("Cross-encoder unavailable (%s); using lexical reranker", type(exc).__name__)
        return LexicalReranker()


def rerank(reranker, query: str, evidence: list[Evidence]) -> list[Evidence]:
    scores = reranker.score(query, [f"{e.chunk.section}\n{e.chunk.text}" for e in evidence])
    for e, s in zip(evidence, scores):
        e.score = float(s)
    return sorted(evidence, key=lambda e: -e.score)
