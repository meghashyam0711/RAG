"""LangGraph state for the self-corrective loop."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from ..models import Evidence


class RAGState(TypedDict, total=False):
    query: str
    filters: dict | None
    sub_questions: list[str]
    search_queries: list[str]          # current (possibly rewritten) query per sub-question
    hops: int
    vector_k: int
    round: int                         # retrieval rounds completed
    evidence: list[Evidence]
    graph_facts: list[str]
    coverage: dict[str, float]         # sub-question -> evidence coverage
    missing_terms: dict[str, list[str]]
    grade: str                         # RELEVANT | AMBIGUOUS | MISSING
    draft: str
    citations: dict[str, str]          # "E1" -> chunk_id
    calculations: list[dict]
    verify: str                        # GROUNDED | HALLUCINATED | SIGN_ERROR
    unsupported: list[str]
    regenerations: int
    feedback: str
    final: dict[str, Any]
    trace: Annotated[list[str], operator.add]
