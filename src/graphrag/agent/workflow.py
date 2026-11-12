"""LangGraph StateGraph wiring for the self-corrective loop.

    route ─► retrieve ─► grade ──RELEVANT──────────────► generate ─► verify ──GROUNDED──► finalize
                ▲          │                               ▲           │
                │          ├─AMBIGUOUS/MISSING (≤N)─► rewrite          └─HALLUCINATED/SIGN_ERROR
                └──────────┘                               └──────────── regenerate (≤M)
                           └─MISSING after N retries ─► abstain
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import Nodes
from .state import RAGState


def build_workflow(nodes: Nodes):
    g = StateGraph(RAGState)
    g.add_node("route", nodes.route_and_decompose)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade", nodes.grade_documents)
    g.add_node("rewrite", nodes.rewrite_and_expand)
    g.add_node("generate", nodes.generate)
    g.add_node("verify", nodes.verify)
    g.add_node("regenerate", nodes.bump_regeneration)
    g.add_node("abstain", nodes.abstain)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", nodes.after_grade, {"generate": "generate", "rewrite": "rewrite", "abstain": "abstain"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", nodes.after_verify, {"finalize": "finalize", "regenerate": "regenerate"})
    g.add_edge("regenerate", "generate")
    g.add_edge("abstain", END)
    g.add_edge("finalize", END)
    return g.compile()
