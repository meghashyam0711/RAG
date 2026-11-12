"""Hybrid graph-vector retrieval with Reciprocal Rank Fusion.

    query ──┬─► dense (pre-filtered)  ─┐
            ├─► sparse BM25           ─┼─► RRF ─► rerank top-N ─► top-k
            └─► entity link ─► k-hop ─┘           + override/cross-ref
                graph traversal                    evidence pinned in

Graph seeds come from (a) entities mentioned in the query, resolved against
the canonical entity index, and (b) the section entities of the top vector
hits, so even a query that names no entity gets "connected-section"
expansion (Section 2 -> modified by -> Appendix B).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..ingestion.entity_resolver import EntityResolver, section_key
from ..models import Chunk, Evidence
from .reranker import rerank

RRF_K = 60
EDGE_WEIGHT = {"MODIFIES": 1.0, "DEFINES": 0.9, "REFERENCES": 0.8, "HAS_VALUE": 0.8, "DEPENDS_ON": 0.7,
               "REPORTED_IN": 0.6, "PART_OF": 0.5, "MENTIONS": 0.35}
_QUERY_MENTION_RE = re.compile(
    r"`([^`]+)`|\"([^\"]+)\"|\b((?:Section|Appendix|Annex|Note|Item|Clause|Article|Exhibit|Schedule|Chapter)\s+[\w.]+)"
    r"|\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+|[a-z]\w*(?:\.\w+){1,4})\b",
    re.I,
)


@dataclass
class RetrievalResult:
    evidence: list[Evidence]
    graph_facts: list[str] = field(default_factory=list)
    seeds: list[str] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def rrf(ranked_lists: dict[str, list[str]], k: int = RRF_K) -> list[tuple[str, float, list[str]]]:
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    for name, ids in ranked_lists.items():
        for rank, cid in enumerate(ids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            sources.setdefault(cid, []).append(name)
    return sorted(((c, s, sources[c]) for c, s in scores.items()), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, vector_store, graph_store, resolver: EntityResolver, reranker, settings) -> None:
        self.vs = vector_store
        self.gs = graph_store
        self.resolver = resolver
        self.reranker = reranker
        self.s = settings

    # ------------------------------------------------------------------
    def link_query_entities(self, query: str) -> list[str]:
        mentions: list[str] = []
        for m in _QUERY_MENTION_RE.finditer(query):
            mentions.append(next(g for g in m.groups() if g))
        # n-grams catch multi-word names ("gross margin", "token bucket")
        words = re.findall(r"[\w']+", query)
        for n in (3, 2, 1):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i : i + n])
                if len(gram) > 3:
                    mentions.append(gram)
        seeds: dict[str, float] = {}
        for mention in mentions:
            is_explicit = bool(_QUERY_MENTION_RE.fullmatch(mention)) or section_key(mention) is not None
            cutoff = 88.0 if is_explicit else 95.0
            for eid, score in self.resolver.lookup(mention, limit=3, score_cutoff=cutoff):
                if eid.startswith("Section:") and section_key(mention) is None and score < 100:
                    continue  # don't seed on fuzzy section titles
                seeds[eid] = max(seeds.get(eid, 0.0), score)
        return [e for e, _ in sorted(seeds.items(), key=lambda x: -x[1])][:8]

    def _section_entity(self, chunk: Chunk) -> str | None:
        if not chunk.section:
            return None
        return self.resolver.canonical_id(chunk.section, "Section", chunk.doc_id)

    # ------------------------------------------------------------------
    def retrieve(self, query: str, filters: dict | None = None, hops: int | None = None,
                 vector_k: int | None = None, final_k: int | None = None, use_sparse: bool = True) -> RetrievalResult:
        """hops=0 disables the graph leg; use_sparse=False disables BM25 (both used for ablations)."""
        hops = self.s.graph_hops if hops is None else hops
        vector_k = vector_k or self.s.vector_k
        final_k = final_k or self.s.final_k

        dense = self.vs.dense_search(query, vector_k, filters)
        sparse = self.vs.sparse_search(query, vector_k, filters) if use_sparse else []
        pool: dict[str, Chunk] = {c.chunk_id: c for c, _ in dense + sparse}

        # --- graph leg ----------------------------------------------------
        query_seeds = self.link_query_entities(query)
        prelim = rrf({"dense": [c.chunk_id for c, _ in dense], "sparse": [c.chunk_id for c, _ in sparse]})
        hit_seeds = []
        for cid, _, _ in prelim[:2]:
            sec = self._section_entity(pool[cid])
            if sec:
                hit_seeds.append(sec)
        seeds = list(dict.fromkeys(query_seeds + hit_seeds))
        dist, edges = self.gs.neighborhood(seeds, hops=hops) if (seeds and hops > 0) else ({}, [])

        graph_scores: dict[str, float] = {}
        facts: list[str] = []
        edge_by_chunk: dict[str, str] = {}
        for e in edges:
            w = EDGE_WEIGHT.get(e["type"], 0.3) / e["hop"]
            # Seeds named in the query get full weight; seeds from vector hits get less.
            if e["source"] in query_seeds or e["target"] in query_seeds:
                w *= 1.5
            cid = e.get("chunk_id")
            if cid:
                graph_scores[cid] = max(graph_scores.get(cid, 0.0), w)
                edge_by_chunk.setdefault(cid, f"{_short(e['source'])} -[{e['type']}]-> {_short(e['target'])}")
            if e["type"] not in ("MENTIONS",):
                facts.append(f"{_short(e['source'])} -[{e['type']}]-> {_short(e['target'])}"
                             + (f'  (evidence: "{e["evidence"][:140]}")' if e.get("evidence") else ""))
        for nid, d in dist.items():
            if d == 0 and nid in query_seeds:
                ent = self.gs.get(nid)
                for cid in (ent.chunk_ids if ent else [])[:6]:
                    graph_scores[cid] = max(graph_scores.get(cid, 0.0), 1.2)
        # Chunks of sections that MODIFY a seed/hit section are always graph candidates.
        modifier_sections = {e["source"] for e in edges if e["type"] == "MODIFIES"} | \
                            {e["target"] for e in edges if e["type"] == "MODIFIES"}

        graph_ranked = [c for c, _ in sorted(graph_scores.items(), key=lambda x: -x[1])]
        missing = [c for c in graph_ranked if c not in pool]
        for c in self.vs.get_chunks(missing[: 3 * vector_k]):
            if _passes(c, filters):
                pool[c.chunk_id] = c
        graph_ranked = [c for c in graph_ranked if c in pool][: vector_k]

        fused = rrf({
            "dense": [c.chunk_id for c, _ in dense],
            "sparse": [c.chunk_id for c, _ in sparse],
            "graph": graph_ranked,
        })[: self.s.fused_k]
        candidates = [Evidence(chunk=pool[cid], score=s, sources=src, graph_path=edge_by_chunk.get(cid))
                      for cid, s, src in fused]
        ranked = rerank(self.reranker, query, candidates)
        final = ranked[:final_k]

        # Pin override evidence: silently dropping "Appendix B supersedes Section 2"
        # is the most damaging omission a spec/filing QA system can make.
        chosen_sections = {self._section_entity(e.chunk) for e in final}
        pinned = 0
        for e in ranked[final_k:] + [Evidence(chunk=pool[c], sources=["graph"], graph_path=edge_by_chunk.get(c))
                                      for c in graph_ranked if c not in {x.chunk.chunk_id for x in ranked}]:
            sec = self._section_entity(e.chunk)
            if pinned < 2 and sec in modifier_sections and sec not in chosen_sections and modifier_sections & chosen_sections:
                e.sources = list(dict.fromkeys(e.sources + ["graph:MODIFIES"]))
                final.append(e)
                chosen_sections.add(sec)
                pinned += 1

        return RetrievalResult(
            evidence=final,
            graph_facts=list(dict.fromkeys(facts))[:15],
            seeds=seeds,
            debug={"dense": len(dense), "sparse": len(sparse), "graph": len(graph_ranked), "hops": hops,
                   "query_entities": query_seeds},
        )


def _short(entity_id: str) -> str:
    t, _, rest = entity_id.partition(":")
    return f"{t}:{rest.split('/')[-1]}"


def _passes(chunk: Chunk, filters: dict | None) -> bool:
    if not filters:
        return True
    for k, v in filters.items():
        val = getattr(chunk, k, None)
        if isinstance(v, (list, tuple, set)):
            if val not in v:
                return False
        elif val != v:
            return False
    return True
