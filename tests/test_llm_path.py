"""Exercise the LLM-backed nodes with a scripted stub model (no network)."""

from pathlib import Path

import pytest

from graphrag.config import Settings
from graphrag.engine import GraphRAGEngine
from graphrag.llm import LLM

FIXTURE = Path(__file__).parent / "fixtures" / "acme_gateway_spec.md"


class ScriptedLLM(LLM):
    """First draft inverts a sign (must be caught); second draft is correct and uses <calc>."""

    def __init__(self):
        super().__init__("stub", "stub", None)
        self.drafts = 0
        self.saw_feedback = False

    def structured(self, system, prompt, schema, max_tokens=2000):
        return None  # force deterministic fallbacks for decomposition/grading/extraction

    def complete(self, system, prompt, max_tokens=1500):
        self.drafts += 1
        if "failed verification" in prompt:
            self.saw_feedback = True
            return ('Net savings from caching were (0.3) in Q3 FY2025, down from 0.9 in Q3 FY2024. [E1] '
                    'Infrastructure cost grew <calc label="cost growth %">pct_change(4.2, 5.1)</calc> percent. [E1]')
        return "Net savings from caching were 0.3 in Q3 FY2025. [E1]"


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHRAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GRAPHRAG_LLM_PROVIDER", "none")
    monkeypatch.setenv("GRAPHRAG_EMBEDDER", "hash")
    eng = GraphRAGEngine(Settings())
    eng.ingest(str(FIXTURE), "technical")
    stub = ScriptedLLM()
    eng.nodes.llm = stub
    yield eng, stub
    eng.close()


def test_sign_error_triggers_regeneration_and_calc_is_sandboxed(engine):
    eng, stub = engine
    res = eng.ask("How did net savings from caching and infrastructure cost change between Q3 FY2024 and Q3 FY2025?")
    assert stub.drafts == 2 and stub.saw_feedback
    assert "(0.3)" in res["answer"]
    assert "**21.4286**" in res["answer"]
    assert res["calculations"][0]["result"] == pytest.approx(21.428571, rel=1e-6)
    assert any("regenerate" in t for t in res["trace"])
