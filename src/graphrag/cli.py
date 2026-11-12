"""`graphrag`: use the engine straight from a terminal (no coding agent needed), or install it into one.

  graphrag ingest spec.pdf --domain technical
  graphrag ask "What is max_requests_per_minute for Enterprise keys?"
  graphrag trace max_requests_per_minute --hops 2
  graphrag table "default limits by endpoint" --compute "pct_change(600, 6000)"
  graphrag docs
  graphrag chat                      # interactive Q&A session
  graphrag serve                     # run the MCP server (stdio, or --http)
  graphrag install cursor            # register with an agent/IDE (see `graphrag install --help`)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anyio


def _quiet_logs() -> None:
    os.environ.setdefault("GRAPHRAG_LOG_LEVEL", "ERROR")
    import logging

    logging.getLogger().setLevel(os.environ["GRAPHRAG_LOG_LEVEL"])


def _server():
    _quiet_logs()
    from . import server

    return server


def _call(coro_fn, *args):
    return anyio.run(lambda: coro_fn(*args))


def _print(text: str, plain: bool) -> None:
    if plain:
        import re

        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"</?details>|<summary>.*?</summary>", "", text)
    print(text)


# ---------------------------------------------------------------------------
def cmd_ingest(a) -> int:
    s = _server()
    rc = 0
    import glob

    files: list[str] = []
    for f in a.files:  # expand wildcards ourselves: PowerShell/cmd pass "*.pdf" through literally
        matches = sorted(glob.glob(os.path.expanduser(f))) if any(c in f for c in "*?[") else [f]
        if not matches:
            print(f"Error: no files match {f}")
            rc = 1
        files.extend(matches)
    for f in files:
        out = _call(s.ingest_complex_doc, str(Path(f).expanduser().resolve()), a.domain, a.name, a.force)
        _print(out, a.plain)
        rc |= out.startswith("Error")
    return rc


def cmd_ask(a) -> int:
    s = _server()
    query = " ".join(a.question)
    out = _call(s.ask_verified_docs, query, a.doc, "json" if a.json else "markdown")
    _print(out, a.plain)
    return 0 if not out.startswith("Error") else 1


def cmd_trace(a) -> int:
    s = _server()
    _print(_call(s.trace_entity_relations, a.entity, a.hops, a.doc), a.plain)
    return 0


def cmd_table(a) -> int:
    s = _server()
    _print(_call(s.extract_verified_table, " ".join(a.topic), a.doc, a.compute), a.plain)
    return 0


def cmd_docs(a) -> int:
    s = _server()
    _print(_call(s.list_indexed_docs), a.plain)
    return 0


def cmd_chat(a) -> int:
    s = _server()
    print("GraphRAG chat. Ask questions about your indexed documents.")
    print("Commands: :ingest <file> [domain] | :trace <entity> | :table <topic> | :docs | :doc <name|*> | :quit\n")
    doc = a.doc
    while True:
        try:
            line = input(f"graphrag{'[' + doc + ']' if doc else ''}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            return 0
        try:
            if line.startswith(":ingest "):
                parts = line.split()
                out = _call(s.ingest_complex_doc, str(Path(parts[1]).expanduser().resolve()),
                            parts[2] if len(parts) > 2 else "general", None, False)
            elif line.startswith(":trace "):
                out = _call(s.trace_entity_relations, line[7:].strip(), 2, doc)
            elif line.startswith(":table "):
                out = _call(s.extract_verified_table, line[7:].strip(), doc, None)
            elif line == ":docs":
                out = _call(s.list_indexed_docs)
            elif line.startswith(":doc"):
                arg = line[4:].strip()
                doc = None if arg in ("", "*") else arg
                out = f"document filter: {doc or 'all documents'}"
            else:
                out = _call(s.ask_verified_docs, line, doc, "markdown")
        except Exception as exc:  # keep the session alive
            out = f"Error: {exc}"
        _print(out, a.plain)
        print()


_SAMPLE_DOC = """# Doctor Sample Spec
<!-- page: 2 -->
## 2. Rate Limiting
The parameter `max_requests_per_minute` defaults to 600. See Appendix B for Enterprise overrides.
<!-- page: 9 -->
## Appendix B: Enterprise Overrides
This appendix supersedes the default limits in Section 2. For Enterprise keys `max_requests_per_minute` is set to 6000.
"""


def cmd_doctor(a) -> int:
    """Self-test in a throwaway index: imports, ingest, multi-hop answer, abstention, client configs."""
    import shutil
    import tempfile

    ok = True

    def check(name: str, passed: bool, hint: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + ("" if passed or not hint else f"  -> {hint}"))

    check(f"Python {sys.version.split()[0]}", sys.version_info >= (3, 11), "need Python 3.11+ (uv sync installs it)")
    check("uv on PATH", bool(shutil.which("uv")), "install uv, then open a NEW terminal")
    tmp = tempfile.mkdtemp(prefix="graphrag-doctor-")
    os.environ["GRAPHRAG_DATA_DIR"] = tmp
    try:
        from .config import Settings
        from .engine import GraphRAGEngine

        eng = GraphRAGEngine(Settings())
        doc = Path(tmp) / "doctor_sample.md"
        doc.write_text(_SAMPLE_DOC, encoding="utf-8")
        check("ingest", eng.ingest(str(doc), "technical")["status"] == "indexed")
        r = eng.ask("What is max_requests_per_minute for Enterprise keys?")
        check("multi-hop verified answer", r["status"] == "verified" and "6000" in r["answer"], str(r.get("status")))
        r = eng.ask("What uptime SLA is guaranteed?")
        check("abstains when docs are silent", r["status"] == "abstained")
        caps = eng.capabilities()
        print(f"       backends: embedder={caps['embedder']}, llm={caps['llm']}, graph={caps['graph']}")
        eng.close()
    except Exception as exc:
        check("engine", False, f"{type(exc).__name__}: {exc}  (run `uv sync` in the repo folder)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    from . import installer

    launch = installer.resolve_launch()
    print(f"       MCP launch command: {launch.command} {' '.join(launch.args)}")
    for client in installer.CLIENTS:
        path = installer.config_path(client, "user", Path.cwd())
        if path and path.exists():
            found = installer.SERVER_NAME in path.read_text(encoding="utf-8", errors="replace")
            print(f"       {client}: {'configured' if found else 'not configured'} ({path})")
    print("\nAll checks passed." if ok else "\nSome checks failed; see hints above.")
    return 0 if ok else 1


def cmd_serve(a) -> int:
    if a.http:
        os.environ["GRAPHRAG_TRANSPORT"] = "streamable-http"
        os.environ["GRAPHRAG_PORT"] = str(a.port)
        os.environ["GRAPHRAG_HOST"] = a.host
        print(f"MCP server on http://{a.host}:{a.port}/mcp", file=sys.stderr)
    from .server import main

    main()
    return 0


def cmd_install(a) -> int:
    from . import installer

    if a.print_all:
        print(installer.snippets(a.launcher, a.data_dir))
        return 0
    if not a.client:
        print("usage: graphrag install <client> | all | --print-all\nclients: " + ", ".join(installer.CLIENTS))
        return 2
    targets = installer.detect_clients() if a.client == "all" else [a.client]
    if a.client == "all":
        print(f"detected: {', '.join(targets) or 'none'}")
    rc = 0
    for client in targets:
        scope = a.scope
        try:
            if a.uninstall:
                msg = installer.uninstall(client, scope, Path(a.project_dir) if a.project_dir else None)
            else:
                msg = installer.install(client, scope, Path(a.project_dir) if a.project_dir else None,
                                        a.launcher, a.data_dir, a.print)
            print(f"[{client}] {msg}")
        except Exception as exc:
            print(f"[{client}] skipped: {exc}")
            rc = 1
    if not a.print and not a.uninstall:
        print("\nRestart the client (or reload its MCP servers) to pick up graphrag-reader.")
    return rc


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    from .installer import CLIENTS

    p = argparse.ArgumentParser(prog="graphrag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plain", action="store_true", help="strip markdown from output")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="parse and index one or more documents")
    s.add_argument("files", nargs="+")
    s.add_argument("--domain", default="general", choices=["general", "technical", "finance", "legal"])
    s.add_argument("--name", help="document name used in citations (single file only)")
    s.add_argument("--force", action="store_true", help="re-index even if unchanged")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("ask", help="ask a verified question")
    s.add_argument("question", nargs="+")
    s.add_argument("--doc", help="restrict to a document name/id")
    s.add_argument("--json", action="store_true", help="structured JSON output")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("trace", help="show knowledge-graph relations around an entity")
    s.add_argument("entity")
    s.add_argument("--hops", type=int, default=2)
    s.add_argument("--doc")
    s.set_defaults(fn=cmd_trace)

    s = sub.add_parser("table", help="extract an exact table, optionally compute over it")
    s.add_argument("topic", nargs="+")
    s.add_argument("--doc")
    s.add_argument("--compute", help='arithmetic, e.g. "pct_change(1,840, 2,310)"')
    s.set_defaults(fn=cmd_table)

    s = sub.add_parser("docs", help="list indexed documents and active backends")
    s.set_defaults(fn=cmd_docs)

    s = sub.add_parser("chat", help="interactive question-answering session")
    s.add_argument("--doc")
    s.set_defaults(fn=cmd_chat)

    s = sub.add_parser("doctor", help="self-test the install (uses a temporary index)")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("serve", help="run the MCP server")
    s.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("install", help="register the MCP server with an agent / IDE",
                       description="Clients: " + ", ".join(CLIENTS) + ", or 'all' (every detected client).")
    s.add_argument("client", nargs="?", choices=CLIENTS + ["all"])
    s.add_argument("--scope", choices=["user", "project"], default="user",
                   help="user = all projects (default); project = config file in --project-dir")
    s.add_argument("--project-dir", help="project root for --scope project (default: current directory)")
    s.add_argument("--launcher", choices=["auto", "uv", "script", "python"], default="auto")
    s.add_argument("--data-dir", help="index location to pin in the config (GRAPHRAG_DATA_DIR)")
    s.add_argument("--print", action="store_true", help="show what would be written; change nothing")
    s.add_argument("--print-all", action="store_true", help="print copy-paste config for every client")
    s.add_argument("--uninstall", action="store_true", help="remove graphrag-reader from the client")
    s.set_defaults(fn=cmd_install)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; documents contain any Unicode
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
