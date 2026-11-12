"""Spawn mcp_server.py over stdio and drive it with the official MCP client, like Claude Code does."""

import os
import sys
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "acme_gateway_spec.md"


def _text(result) -> str:
    return "\n".join(getattr(c, "text", "") for c in result.content)


def test_mcp_server_end_to_end(tmp_path):
    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(ROOT / "mcp_server.py")],
            env={**os.environ, "GRAPHRAG_DATA_DIR": str(tmp_path), "GRAPHRAG_LLM_PROVIDER": "none",
                 "GRAPHRAG_EMBEDDER": "hash"},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name: t for t in (await session.list_tools()).tools}
                assert {"ingest_complex_doc", "ask_verified_docs", "trace_entity_relations",
                        "extract_verified_table", "list_indexed_docs"} <= set(tools)
                assert tools["ask_verified_docs"].annotations.read_only_hint is True

                out = _text(await session.call_tool("ingest_complex_doc",
                                                    {"file_path": str(FIXTURE), "domain_tag": "technical"}))
                assert "indexed" in out and "MODIFIES" in out

                out = _text(await session.call_tool("ask_verified_docs",
                                                    {"query": "What is max_requests_per_minute for Enterprise keys?"}))
                assert "VERIFIED" in out and "6000" in out and "[E" in out

                out = _text(await session.call_tool("ask_verified_docs",
                                                    {"query": "What uptime SLA does the gateway guarantee?"}))
                assert "ABSTAINED" in out

                out = _text(await session.call_tool("trace_entity_relations",
                                                    {"entity_name": "max_requests_per_minute", "max_hops": 2}))
                assert "MODIFIES" in out

                out = _text(await session.call_tool("extract_verified_table",
                                                    {"topic": "default limits by endpoint timeout_ms",
                                                     "compute": "5000 / 1000"}))
                assert "| Endpoint |" in out and "= **5.0**" in out

                out = _text(await session.call_tool("ingest_complex_doc", {"file_path": "/nope.pdf"}))
                assert out.startswith("Error")

    anyio.run(run)
