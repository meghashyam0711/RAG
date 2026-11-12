import pytest

from graphrag.agent.sandbox import SandboxError, normalize_numbers, safe_eval
from graphrag.ingestion.chunker import chunk_blocks
from graphrag.ingestion.entity_resolver import EntityResolver, normalize_fiscal_period, section_key
from graphrag.ingestion.layout_parser import parse_markdown
from graphrag.models import BlockType


@pytest.mark.parametrize("text", ["Q3 '22", "Third Quarter 2022", "3Q22", "Q3 FY2022", "third fiscal quarter of 2022"])
def test_fiscal_period_canonicalization(text):
    assert normalize_fiscal_period(text) == "FY2022_Q3"


def test_fiscal_year_and_half():
    assert normalize_fiscal_period("fiscal 2024") == "FY2024"
    assert normalize_fiscal_period("FY25") == "FY2025"
    assert normalize_fiscal_period("H1 2023") == "FY2023_H1"


def test_section_keys():
    assert section_key("Appendix B: Enterprise Tier Overrides") == "APPENDIX_B"
    assert section_key("2.1 Default Limits by Endpoint") == "SECTION_2.1"
    assert section_key("Section 2") == "SECTION_2"
    assert section_key("Item 7A. Quantitative Disclosures") == "ITEM_7A"


def test_resolver_dedupes_fuzzy_and_scopes_sections():
    r = EntityResolver(threshold=90)
    a = r.resolve("Consolidated Net Income", "FinancialMetric")
    b = r.resolve("consolidated net income", "FinancialMetric")
    c = r.resolve("Consolidated Net Incomes", "FinancialMetric")
    assert a == b == c
    s1 = r.resolve("Appendix B", "Section", "doc1")
    s2 = r.resolve("APPENDIX B - Overrides", "Section", "doc1")
    s3 = r.resolve("Appendix B", "Section", "doc2")
    assert s1 == s2 and s1 != s3


def test_sandbox_accounting_and_safety():
    assert safe_eval("(0.3) + 1") == pytest.approx(0.7)
    assert safe_eval("pct_change(1,840, 2,310)") == pytest.approx(25.543478, rel=1e-6)
    assert safe_eval("$4.2 * 1,000") == pytest.approx(4200)
    assert safe_eval("10% * 50") == pytest.approx(5)
    assert normalize_numbers("abs((1,234))") == "abs((-1234))"
    for bad in ["__import__('os').system('id')", "open('x')", "(lambda: 1)()", "2 ** 10000", "[1,2][0]", "x + 1"]:
        with pytest.raises(SandboxError):
            safe_eval(bad)


def test_tables_are_atomic_and_carry_units():
    md = (
        "## Results\n\nAmounts below are $ in millions.\n\n"
        "| Metric | 2024 | 2025 |\n|---|---|---|\n| Net savings | 0.9 | (0.3) |\n\nMore text here.\n"
    )
    blocks = parse_markdown(md)
    tables = [b for b in blocks if b.type == BlockType.table]
    assert len(tables) == 1 and tables[0].units == "$ in millions" and "(0.3)" in tables[0].text
    chunks = chunk_blocks(blocks, doc_id="d", doc_name="d", domain="finance", chunk_tokens=50)
    t = [c for c in chunks if c.chunk_type == "table"][0]
    assert "| Metric | 2024 | 2025 |" in t.text and "Units: $ in millions" in t.header


def test_large_table_split_repeats_header():
    rows = "\n".join(f"| row_{i} | {i} |" for i in range(200))
    md = f"## T\n\n| name | value |\n|---|---|\n{rows}\n"
    chunks = chunk_blocks(parse_markdown(md), doc_id="d", doc_name="d", domain="general", chunk_tokens=120)
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) > 1
    assert all("| name | value |" in c.text for c in tables)
