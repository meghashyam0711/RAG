"""End-to-end tests of ingestion -> hybrid graph retrieval -> self-correction, in lite mode (no services, no LLM)."""

from pathlib import Path

import pytest

from graphrag.agent.nodes import Nodes
from graphrag.config import Settings
from graphrag.engine import GraphRAGEngine
from graphrag.models import Evidence

FIXTURE = Path(__file__).parent / "fixtures" / "acme_gateway_spec.md"


@pytest.fixture(scope="module")
def engine(tmp_path_factory, monkeypatch_module):
    d = tmp_path_factory.mktemp("graphrag")
    monkeypatch_module.setenv("GRAPHRAG_DATA_DIR", str(d))
    monkeypatch_module.setenv("GRAPHRAG_LLM_PROVIDER", "none")
    monkeypatch_module.setenv("GRAPHRAG_EMBEDDER", "hash")
    eng = GraphRAGEngine(Settings())
    eng.ingest(str(FIXTURE), "technical", doc_name="acme_gateway_spec")
    yield eng
    eng.close()


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_ingest_builds_graph_with_override_edge(engine):
    doc = next(iter(engine.registry.values()))
    assert doc["tables"] == 2 and doc["relations"].get("MODIFIES", 0) >= 1
    res = engine.ingest(str(FIXTURE), "technical", doc_name="acme_gateway_spec")
    assert res["status"].startswith("unchanged")


def test_multi_hop_override_is_retrieved(engine):
    res = engine.ask("What is max_requests_per_minute for Enterprise keys?")
    assert res["status"] == "verified"
    assert "6000" in res["answer"]
    assert any("Appendix B" in c["source"] for c in res["citations"])


def test_override_surfaces_even_when_query_omits_it(engine):
    # The query never mentions Enterprise/Appendix B; the graph must still surface the override.
    res = engine.ask("What is the default max_requests_per_minute?")
    assert "600" in res["answer"]
    assert "MODIFIES" in res["answer"] or any("Appendix B" in c["source"] for c in res["citations"])


def test_abstains_when_documents_are_silent(engine):
    res = engine.ask("What uptime SLA percentage does the gateway guarantee?")
    assert res["status"] == "abstained"
    assert res["retrieval_rounds"] == 3  # initial + 2 corrective retries


def test_parenthetical_negative_preserved(engine):
    res = engine.ask("What were net savings from caching in Q3 FY2025?")
    assert "(0.3)" in res["answer"]


def test_verifier_flags_sign_and_negation_errors(engine):
    ev = engine.retriever.retrieve("net savings from caching Q3 FY2025").evidence
    nodes: Nodes = engine.nodes
    table_idx = next(i for i, e in enumerate(ev, 1) if e.chunk.chunk_type == "table")
    state = {
        "draft": f"Net savings from caching in Q3 FY2025 were 0.3 million. [E{table_idx}]",
        "citations": {f"E{i}": e.chunk.chunk_id for i, e in enumerate(ev, 1)},
        "evidence": ev, "calculations": [],
    }
    assert nodes.verify(state)["verify"] == "SIGN_ERROR"

    auth = engine.retriever.retrieve("webhook signatures encrypted").evidence
    i = next(i for i, e in enumerate(auth, 1) if "HMAC" in e.chunk.text)
    state = {
        "draft": f"Webhook signatures use HMAC-SHA256 and are encrypted. [E{i}]",
        "citations": {f"E{k}": e.chunk.chunk_id for k, e in enumerate(auth, 1)},
        "evidence": auth, "calculations": [],
    }
    assert nodes.verify(state)["verify"] == "SIGN_ERROR"


def test_verifier_flags_invented_numbers(engine):
    ev = engine.retriever.retrieve("burst_size default").evidence
    i = next(i for i, e in enumerate(ev, 1) if "burst_size" in e.chunk.text)
    state = {
        "draft": f"The parameter burst_size defaults to 250. [E{i}]",
        "citations": {f"E{k}": e.chunk.chunk_id for k, e in enumerate(ev, 1)},
        "evidence": ev, "calculations": [],
    }
    assert nodes_verify(engine, state) == "HALLUCINATED"


def nodes_verify(engine, state):
    return engine.nodes.verify(state)["verify"]


def test_trace_entity_relations(engine):
    res = engine.trace_entity("max_requests_per_minute", max_hops=2)
    assert res["found"]
    rels = {(r["relation"], r["source"].split(":")[-1].split("/")[-1]) for r in res["relations"]}
    assert ("MODIFIES", "APPENDIX_B") in rels


def test_extract_verified_table_with_sandbox(engine):
    res = engine.extract_table("operational metrics requests served", compute="pct_change(1,840, 2,310)")
    assert res["found"] and res["units"] == "$ in millions"
    assert "(0.3)" in res["table_markdown"]
    assert res["calculation"]["result"] == pytest.approx(25.543478, rel=1e-6)
    assert res["calculation"]["all_operands_grounded"]


def test_doc_filter_errors_cleanly(engine):
    with pytest.raises(ValueError):
        engine.ask("anything", doc_filter="no_such_document_xyz")


@pytest.mark.parametrize("query", ["", "   \n\t ", "???"])
def test_contentless_query_is_not_reported_verified(engine, query):
    assert engine.ask(query)["status"] == "abstained"


def test_non_latin1_document_ingests_and_answers(engine, tmp_path):
    """Regression: the parsed-markdown copy was written with the locale encoding,
    so any document outside cp1252 crashed ingestion on Windows."""
    doc = tmp_path / "unicode_spec.md"
    doc.write_text(
        "# 速度仕様 — Throughput µ-spec \U0001f680\n\n"
        "## 1. 限度 (Limits)\n\n"
        "The parameter `queue_depth_max` defaults to 512 ≤ bound. "
        "Заметка: α-channel only.\n",
        encoding="utf-8",
    )
    res = engine.ingest(str(doc), "technical", doc_name="unicode_spec")
    assert "indexed" in res["status"] and res["chunks"] >= 1
    assert "512" in engine.ask("What is queue_depth_max?", doc_filter="unicode_spec")["answer"]


def test_empty_document_is_flagged_not_silently_indexed(engine, tmp_path):
    doc = tmp_path / "blank.md"
    doc.write_text("", encoding="utf-8")
    res = engine.ingest(str(doc), "general", doc_name="blank_doc")
    assert res["chunks"] == 0 and "no text was extracted" in res["status"]


def test_ingest_respects_allowed_dirs(engine, tmp_path, monkeypatch):
    doc = tmp_path / "outside.md"
    doc.write_text("# X\n\nThe parameter `p` defaults to 1.\n", encoding="utf-8")
    monkeypatch.setattr(engine.s, "allowed_ingest_dirs", [tmp_path / "elsewhere"])
    with pytest.raises(PermissionError):
        engine.ingest(str(doc), "general")
