"""MCP server exposing the Self-Corrective GraphRAG engine.

Transport: stdio by default (Claude Code, Cursor, Windsurf, Claude Desktop).
Set GRAPHRAG_TRANSPORT=streamable-http to serve remotely.

stdout is reserved for JSON-RPC on stdio, so all logging goes to stderr.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from typing import Literal

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

logging.basicConfig(
    stream=sys.stderr,
    level=os.environ.get("GRAPHRAG_LOG_LEVEL", "WARNING"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("graphrag.server")

INSTRUCTIONS = """\
Document-intelligence tools backed by a layout-aware, self-correcting GraphRAG index.
Use them instead of reading large PDFs / specs / filings into context:
1. ingest_complex_doc once per document (PDF, DOCX, HTML, Markdown).
2. ask_verified_docs for questions; answers carry [E#] citations and a status
   (verified / partially_verified / abstained). Treat "abstained" as "the documents do not say".
3. trace_entity_relations to see how a parameter / component / metric connects across sections and documents
   (e.g. a default in Section 2 overridden in Appendix B).
4. extract_verified_table for exact tables; pass `compute` for arithmetic instead of doing math yourself.
"""

mcp = MCPServer(name="graphrag-reader", instructions=INSTRUCTIONS, version="0.1.0")

_engine = None
_engine_lock = threading.Lock()


def get_engine():
    """Lazy init keeps server start-up instant (MCP clients time out slow handshakes)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                import atexit

                from .engine import GraphRAGEngine

                _engine = GraphRAGEngine()
                atexit.register(_engine.close)  # release the embedded Qdrant lock cleanly
    return _engine


async def _run(fn, *args, **kwargs):
    return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations=ToolAnnotations(title="Ingest complex document", read_only_hint=False, destructive_hint=False,
                                idempotent_hint=True, open_world_hint=False),
)
async def ingest_complex_doc(
    file_path: str,
    domain_tag: Literal["general", "technical", "finance", "legal"] = "general",
    doc_name: str | None = None,
    force: bool = False,
) -> str:
    """Parse and index a document so it can be queried with the other tools.

    Runs layout-aware parsing (Docling when installed: multi-column reading order, intact tables, unit headers),
    contextual chunking, ontology-constrained entity/relation extraction into the knowledge graph, and hybrid
    dense + BM25 indexing. Re-ingesting an unchanged file is a no-op unless force=true.

    Args:
        file_path: Absolute path to a .pdf, .docx, .pptx, .html, .md or .txt file.
        domain_tag: Selects the extraction ontology: technical (APIs, specs, datasheets), finance (10-K/10-Q),
            legal (contracts), or general.
        doc_name: Optional short name used in citations and doc filters (defaults to the file name).
        force: Re-index even if the file is unchanged.
    """
    path = os.path.expanduser(file_path)
    if not os.path.isfile(path):
        return f"Error: file not found: {file_path}"
    try:
        res = await _run(get_engine().ingest, path, domain_tag, doc_name, force)
    except Exception as exc:
        log.exception("ingest failed")
        return f"Error ingesting {file_path}: {type(exc).__name__}: {exc}"
    rels = ", ".join(f"{k}={v}" for k, v in sorted(res.get("relations", {}).items())) or "none"
    return (
        f"**{res['doc_name']}** ({res['doc_id']}): {res['status']}\n"
        f"- parser: {res['parser']} | chunks: {res['chunks']} (tables: {res['tables']}) | pages: {res.get('pages') or 'n/a'}\n"
        f"- graph: {res['entities']} entities; relations: {rels}\n"
        f"- use doc_filter=\"{res['doc_name']}\" to scope queries to this document"
    )


@mcp.tool(
    annotations=ToolAnnotations(title="Ask verified question", read_only_hint=True, open_world_hint=False),
)
async def ask_verified_docs(
    query: str,
    doc_filter: str | list[str] | None = None,
    response_format: Literal["markdown", "json"] = "markdown",
) -> str:
    """Answer a question from the indexed documents using the self-corrective GraphRAG loop.

    Pipeline: query decomposition -> hybrid dense/BM25/graph retrieval with RRF -> rerank -> CRAG evidence grading
    -> query rewrite + graph-hop expansion when evidence is missing (max 2 retries) -> grounded generation with a
    deterministic math sandbox -> hallucination, negation and accounting-sign verification.

    Returns ~5 cited chunks' worth of answer (not whole documents). Status meanings:
    verified = every claim is supported by cited evidence; partially_verified = some evidence missing or
    unsupported claims removed; abstained = the documents do not contain the answer (do not guess).

    Args:
        query: Natural-language question. Name parameters/sections verbatim when you know them.
        doc_filter: Optional document name/id (or list) to restrict the search.
        response_format: markdown (default, for reading) or json (structured, for programmatic use).
    """
    try:
        res = await _run(get_engine().ask, query, doc_filter)
    except ValueError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        log.exception("ask failed")
        return f"Error: {type(exc).__name__}: {exc}"
    if response_format == "json":
        return json.dumps(res, indent=2, default=str)
    return render_answer(res)


@mcp.tool(
    annotations=ToolAnnotations(title="Trace entity relations", read_only_hint=True, open_world_hint=False),
)
async def trace_entity_relations(entity_name: str, max_hops: int = 2, doc_filter: str | None = None) -> str:
    """Traverse the knowledge graph around an entity (parameter, API component, section, metric, party...).

    Shows where it is defined, which sections reference or override it, and what it depends on, with the
    citation for every edge. Useful for finding cross-references 100+ pages apart.

    Args:
        entity_name: Name as written, e.g. "max_requests_per_minute", "Appendix B", "Net Income", "Q3 FY2025".
        max_hops: Traversal depth, 1-4 (default 2).
        doc_filter: Optional document name/id to restrict edges to.
    """
    try:
        res = await _run(get_engine().trace_entity, entity_name, max_hops, doc_filter)
    except ValueError as exc:
        return f"Error: {exc}"
    if not res.get("found"):
        return res["message"]
    lines = [
        f"**{res['resolved_id']}** (type: {res['type']}; documents: {', '.join(res['documents'])}; "
        f"{res['reachable_entities']} entities within {max_hops} hop(s))",
    ]
    if res["aliases"]:
        lines.append(f"aliases: {', '.join(res['aliases'][:8])}")
    if res["alternatives"]:
        lines.append(f"other matches: {', '.join(res['alternatives'])}")
    lines += ["", "| hop | source | relation | target | cited at | evidence |", "|---|---|---|---|---|---|"]
    for r in res["relations"]:
        ev = (r["evidence"] or "").replace("|", "\\|")[:110]
        lines.append(f"| {r['hop']} | {r['source']} | {r['relation']} | {r['target']} | {r['cited_at'] or ''} | {ev} |")
    if res["truncated"]:
        lines.append("\n(truncated; lower max_hops or add doc_filter)")
    return "\n".join(lines)


@mcp.tool(
    annotations=ToolAnnotations(title="Extract verified table", read_only_hint=True, open_world_hint=False),
)
async def extract_verified_table(topic: str, doc_name: str | None = None, compute: str | None = None) -> str:
    """Fetch an exact, intact table (header row, units, parenthetical negatives preserved) and optionally compute.

    Args:
        topic: What the table is about, e.g. "default rate limits by endpoint" or "operating metrics Q3".
        doc_name: Optional document name/id.
        compute: Optional arithmetic over values from the table, evaluated in a deterministic sandbox (no LLM math).
            Accounting notation is understood: "(0.3)" is -0.3. Functions: pct_change(old,new), cagr(start,end,years),
            round, abs, min, max, avg, sqrt. Example: "pct_change(1,840, 2,310)".
    """
    try:
        res = await _run(get_engine().extract_table, topic, doc_name, compute)
    except ValueError as exc:
        return f"Error: {exc}"
    if not res.get("found"):
        return res["message"]
    badge = "VERIFIED MATCH" if res["verified"] else f"WEAK MATCH (unmatched: {', '.join(res['unmatched_terms'])})"
    out = [f"**{badge}**: {res['source']}" + (f" | units: {res['units']}" if res["units"] else ""),
           f"chunk_id: `{res['chunk_id']}`", "", res["table_markdown"]]
    calc = res.get("calculation")
    if calc:
        if calc["ok"]:
            grounded = "all operands found in table" if calc.get("all_operands_grounded") else \
                f"operands not in table: {[k for k, v in calc.get('operands_found_in_table', {}).items() if not v]}"
            out += ["", f"**Sandbox:** `{calc['expression']}` -> `{calc['normalized']}` = **{calc['result']}** ({grounded})"]
        else:
            out += ["", f"**Sandbox error:** {calc['error']}"]
    if res["other_candidates"]:
        out += ["", "Other candidate tables: " + "; ".join(f"{c['source']}" for c in res["other_candidates"])]
    return "\n".join(out)


@mcp.tool(annotations=ToolAnnotations(title="List indexed documents", read_only_hint=True, open_world_hint=False))
async def list_indexed_docs() -> str:
    """List indexed documents, graph size and which backends (embedder, reranker, LLM, stores) are active."""
    res = await _run(get_engine().list_documents)
    caps = res["capabilities"]
    lines = [f"backends: " + ", ".join(f"{k}={v}" for k, v in caps.items()),
             f"graph: {res['graph'].get('nodes')} nodes / {res['graph'].get('edges')} edges", ""]
    if not res["documents"]:
        lines.append("No documents indexed yet.")
    for d in res["documents"]:
        lines.append(f"- **{d['doc_name']}** ({d['domain']}, {d['parser']}): {d['chunks']} chunks, "
                     f"{d['tables']} tables, {d['entities']} entities - {d['path']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
STATUS_BADGE = {"verified": "VERIFIED", "partially_verified": "PARTIALLY VERIFIED",
                "unverified": "UNVERIFIED", "abstained": "ABSTAINED"}


def render_answer(res: dict) -> str:
    status = res.get("status", "?")
    lines = [f"**Status: {STATUS_BADGE.get(status, status)}** "
             f"({res.get('retrieval_rounds', 0)} retrieval round(s), {res.get('latency_ms', '?')} ms)", ""]
    lines.append(res.get("answer", ""))
    if status == "abstained":
        if res.get("missing"):
            lines.append(f"\nNot found in evidence: {', '.join(res['missing'][:10])}")
        if res.get("nearest_sources"):
            lines.append(f"Closest sections searched: {'; '.join(res['nearest_sources'])}")
    if res.get("calculations"):
        lines.append("\n**Calculations (deterministic sandbox)**")
        for c in res["calculations"]:
            lines.append(f"- {c.get('label') or 'calc'}: `{c['expression']}` = {c.get('result', c.get('error'))}")
    if res.get("citations"):
        lines.append("\n**Sources**")
        for c in res["citations"]:
            via = ", ".join(c.get("retrieved_by", []))
            lines.append(f"- [{c['ref']}] {c['source']} ({c['type']}; via {via}) `{c['chunk_id']}`")
    if res.get("unsupported_claims"):
        lines.append("\n**Removed by verifier** (not supported by evidence):")
        lines.extend(f"- {u}" for u in res["unsupported_claims"][:5])
    if res.get("trace"):
        lines.append("\n<details><summary>trace</summary>\n\n" + " -> ".join(res["trace"]) + "\n</details>")
    return "\n".join(lines)


def main() -> None:
    transport = os.environ.get("GRAPHRAG_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run("stdio")
    else:
        mcp.run(transport, host=os.environ.get("GRAPHRAG_HOST", "127.0.0.1"),
                port=int(os.environ.get("GRAPHRAG_PORT", "8765")))


if __name__ == "__main__":
    main()
