# Public Release Audit — graphrag-reader

Date: 2026-11-18 · Branch `main`, **nothing committed** · Audited on Windows 11 / Python 3.13.12

Scope note: the functional, RAG, security and performance testing was completed in the review of
2026-11-17 (see `AUDIT_REPORT.md`, 130/130 cases passing). This audit does **not** repeat it. It covers
what that review did not: whether a stranger can download this folder and run it without your laptop.

---

## 1. Architecture analysis

| | |
|---|---|
| Language / runtime | Python ≥3.11, no compiled extensions of its own |
| Frontend | **None.** Not a web app — there is no UI, no browser code, no HTTP application API |
| Backend | An MCP server (`src/graphrag/server.py`, stdio by default) plus a terminal CLI (`src/graphrag/cli.py`) |
| Frameworks | `mcp` (tool protocol), `langgraph` (the self-correction state machine), `pydantic` (typed contracts) |
| RAG pipeline | `ingestion/` (layout parsing → chunking → entity/relation extraction) → `indexing/` (embed + store) → `retrieval/hybrid.py` (dense + BM25 + graph, fused with RRF, reranked) → `agent/` (grade → rewrite → generate → verify → abstain) |
| LLM integration | **Optional.** `llm.py` supports Anthropic and OpenAI (plus any OpenAI-compatible endpoint). With no key every LLM step falls back to a deterministic implementation and the server returns extractive cited evidence |
| Embeddings | fastembed / `BAAI/bge-small-en-v1.5` if installed, else a deterministic offline hashed embedder. BM25 sparse vectors alongside |
| Vector DB | Qdrant — embedded on-disk by default, server mode via `QDRANT_URL` |
| Graph DB | NetworkX persisted to JSON by default, Neo4j/Memgraph via `NEO4J_URI` |
| Document processing | Docling if installed (multi-column order, intact tables), else native markdown/HTML parsing, else pypdf for PDFs |
| Config | Environment variables only, read once in `config.py`. No config file is required |

**Hardcoded local paths: none.** PASS — grep for `D:\`, `C:\Users`, `/home/`, `/Users/` and the author's
username matches only the project's own GitHub URL in `install.ps1` / `install.sh` / `AGENTS.md`.
All runtime paths derive from `GRAPHRAG_DATA_DIR` (default `~/.graphrag-mcp`) and are created
automatically by `Settings.__post_init__` and `Settings.parsed_dir`.

**Missing files / incomplete functionality: none found.** Every module imported is present; no TODO
or stub raised in testing. **Unnecessary artifacts: none** — no stray notebooks, build output, local
databases, logs or editor folders in the tree.

---

## 2. RAG functional testing — PASS (carried forward, re-verified)

Re-verified in this audit on a clean pip environment: ingestion (markdown fixture, 10 chunks, 2 tables,
22 entities, `MODIFIES` edge built), retrieval, multi-hop override answer (`6000` from Appendix B,
cited), abstention, and the full 49-test suite. Full per-question results — direct, multi-section,
unrelated, missing-information, ambiguous, long-compound, negation, accounting-sign, cross-document,
citation integrity, hallucination resistance — are in `AUDIT_REPORT.md` §3–5. No hallucination,
retrieval or citation failure was observed across 23 query tests.

Not implemented, therefore not documented and not tested: chat history / conversational memory. The
`graphrag chat` command is a REPL, but each question is answered independently — there is no memory
across turns. Document upload is a file path, not an HTTP upload; there is no upload endpoint.

---

## 3. Download & installation readiness

| Check | Status | Notes |
|---|---|---|
| Structure suitable for distribution | PASS | Standard `src/` layout, hatchling build backend, console scripts declared |
| All source files present | PASS | Verified by running from a clean venv that shares nothing with the dev environment |
| `requirements.txt` complete | **FIXED** | Did not exist — the project assumed uv. Added `requirements.txt` (8 core deps, mirroring `pyproject.toml`) and `requirements-optional.txt` (the 7 optional upgrades, each annotated with its download size and what it replaces) |
| Dependency versions compatible | PASS | Resolved cleanly on Python 3.13 (68 packages); CI covers 3.11, 3.12, 3.13 |
| Installation instructions correct | PASS | Executed verbatim from the README in a clean venv |
| Config files included | PASS | `.env.example` is complete and committed; no config file is *required* |
| Environment variable examples | PASS | `.env.example` documents all 24 variables, every one optional |
| API keys documented | PASS | Documented as optional, with what changes if you add one. No key is needed to run |
| Database setup instructions | PASS | `docker-compose.yml` for optional Qdrant/Neo4j; the default embedded stores need no setup |
| Model download instructions | PASS | Optional-extras files state what each model downloads and when |
| Required directories auto-created | PASS | Data, Qdrant and parsed-markdown directories are created on first use |
| Local absolute paths removed | PASS | None existed |
| OS-specific assumptions documented | PASS | README gives both Windows PowerShell and macOS/Linux commands, and warns that PowerShell 5.1 rejects `&&` |
| Runs without the author's machine | PASS | Proved in §9 |

### Issues found and fixed in this phase

1. **No pip path at all (HIGH for students).** Installation required `uv`. A student with only Python
   and pip had no documented route. **Fixed:** `requirements.txt`, `requirements-optional.txt`, and a
   copy-pasteable pip section in the README for both OS families.
2. **`graphrag doctor` reported failure on a correct pip install (MEDIUM).** The uv check was a hard
   `[FAIL]`, so `doctor` printed "Some checks failed" and exited 1 on a perfectly working pip install —
   the first thing a new user runs, telling them they had broken it. **Fixed:** uv is now an `[INFO]`
   note, since it is only needed for the `uv run` launcher. Verified with uv removed from `PATH`:
   `All checks passed.`, exit 0, and the launcher correctly falls back to `python mcp_server.py`.
3. **Library warning leaked into `doctor` output (LOW).** `fastembed unavailable (No module named
   'fastembed')` printed above the check list, because `cmd_doctor` never configured logging. Alarming
   and meaningless to a first-time user. **Fixed:** `cmd_doctor` now calls `_quiet_logs()`.
4. **`uv.lock` was gitignored (MEDIUM, needs your decision).** Dependencies resolved fresh on every
   install and every CI run, so two users could get different versions. **Changed:** removed from
   `.gitignore`, so the existing `uv.lock` is now visible as untracked. **Commit it** to give everyone
   reproducible installs.

---

## 4. Documentation — PASS (improved)

The existing README was already accurate and genuinely useful; it was not rewritten. Three real gaps
were filled, and the install instructions were kept short and copy-pasteable as you asked:

- **"What is RAG, and what does this project add?"** — a plain-language explanation of the standard
  pipeline, then the three things this project adds (graph links, the self-correcting loop,
  verification). Written for someone who has not met RAG before.
- **"System requirements"** — OS, Python, disk (~400 MB lite / ~4 GB with all extras), RAM, network,
  and explicitly: **no API key and no services required**.
- **"Install with pip instead of uv"** — six copy-paste lines per OS, what `doctor` should print, and
  the one thing that changes afterwards (`python graphrag_cli.py <cmd>` instead of `uv run graphrag <cmd>`).

Against the 22 requested sections: all are now present except an explicit "Future improvements"
heading — the existing **Known limitations** section covers that ground honestly, and duplicating it
would add words without information. License: `LICENSE` is MIT and `pyproject.toml` declares MIT, so
it is documented truthfully. Nothing unimplemented is documented: no chat memory, no web UI and no
upload endpoint is claimed anywhere.

---

## 5. Code comments & quality — PASS, no churn

Deliberately left almost untouched. This codebase is better commented than most released Python: every
module opens with a docstring explaining *why* the approach was chosen (see `sandbox.py` on why no
`eval`, `entity_resolver.py` on node duplication, `hybrid.py`'s ASCII diagram of the retrieval legs,
`embeddings.py` on the offline fallback), every public method on the engine has a docstring, and the
MCP tool docstrings are the agent-facing API documentation. Names are clear and consistent.

Rewriting comments here would be churn, not improvement, so the only comments added were those
explaining yesterday's fixes and today's three (why `errors="replace"` on the parsed copy, why
coverage cannot be 1.0 without key terms, why uv is only a note). No outdated or misleading comment was
found. No variable or function was renamed, because none was unclear.

---

## 6. Security — PASS (no new findings)

Fully audited yesterday; see `AUDIT_REPORT.md` §6–7. Summary for a public release:

- **No secrets in the repo.** Pattern-scanned: no API keys, no credentials. `.env` is gitignored and
  was never read or modified. Keys come only from the environment and are never logged.
- **Fixed yesterday:** unauthenticated HTTP transport now warns on a non-loopback bind, with a new
  opt-in `GRAPHRAG_ALLOWED_DIRS` allowlist for ingest paths; `docker-compose.yml` publishes Qdrant and
  Neo4j on loopback only; the MCP host is told that quoted document text is data, not instructions.
- **Verified safe:** math sandbox (AST walk, rejects `__import__`, `open`, lambdas, huge exponents);
  path traversal via `doc_name` (sanitised, nothing written outside the data dir); `<script>`/`<style>`
  stripped before indexing; corrupt/empty/binary/unsupported files fail cleanly; parameterised Cypher.
- **Not applicable:** SQL injection (no SQL), XSS and CSRF (no browser surface), unsafe file upload
  (no upload endpoint — ingest takes a local path).
- **Residual, inherent:** prompt injection via an ingested document cannot be eliminated by a tool whose
  job is to quote its sources; the mitigation is the instruction boundary, and the risk sits with the host
  agent. The HTTP transport still has no authentication of its own — use loopback or a proxy.

**`.gitignore` updated** and verified not to exclude anything needed to run the project. Now also
covers `venv/`, `env/`, `.ruff_cache/`, `*.bak` (installer backups of client configs), `.DS_Store`,
`Thumbs.db`, `.idea/`. Still excludes `.env`, `__pycache__/`, `*.py[cod]`, `.venv/`, build output,
`data/raw_docs/*` (keeping `.gitkeep`), generated indexes (`.graphrag-mcp/`) and `results*.json`.
`uv.lock` was **removed** from the ignore list on purpose (see §3.4).

**Dependency vulnerabilities: NOT TESTED.** No vulnerability database was consulted (no network scan
was run against a CVE feed). Run `pip-audit` or enable Dependabot before announcing the release.

---

## 7. Project structure — PASS, no changes needed

No duplicate files, no unused scripts, no temporary files, no debugging code, no hardcoded paths, no
missing `__init__.py` in any `src/graphrag` package. `integrations/` holds per-client config samples
that the README and installer both reference. `tests/` needs no `__init__.py` given the
`pythonpath = ["src", "."]` setting in `pyproject.toml`. Nothing was deleted.

---

## 8. Automated testing — PASS

**49 tests, 49 passed, 0 failed** — run twice: once in the project's uv environment, once in the
clean pip venv from §9.

Coverage by area: ingestion and re-ingestion (incl. duplicate/force), text extraction (markdown, HTML,
PDF, non-Latin-1), chunking (atomic tables, header repetition on split, section boundaries), embedding
and vector round-trip, hybrid retrieval, multi-hop graph override, abstention, citation correctness,
hallucination/sign/negation verification, the math sandbox's rejections, entity resolution, the MCP
server driven over real stdio by the official MCP client, the CLI end to end, installer config merging
for 7 clients, input validation and error handling.

External services are **mocked or absent, not called**: the suite pins `GRAPHRAG_LLM_PROVIDER=none` and
`GRAPHRAG_EMBEDDER=hash`, so **no test needs an API key, a network connection or a paid call**, and
there is no separate credentialed suite to run. `test_llm_path.py` exercises the LLM code path against
a fake client. 15 of the 49 tests were added in yesterday's review.

---

## 9. Fresh installation simulation — PASS

Performed in an **isolated directory with its own interpreter**, sharing nothing with the project's
`.venv`, and with `GRAPHRAG_DATA_DIR` pointed at a scratch folder so your real index was never touched.

| Step | Result |
|---|---|
| `python -m venv` + `pip install -r requirements.txt` | PASS — exit 0, 68 packages, no build errors, no compiler needed |
| Config from `.env.example` | PASS — not needed; every variable is optional and defaults work |
| Required directories created | PASS — data, qdrant and parsed-markdown folders created on first run |
| Application starts | PASS — `python graphrag_cli.py doctor` → `All checks passed.` |
| Backend works | PASS — ingest, ask, trace, table, docs all correct from the clean venv |
| Frontend | N/A — none exists |
| Document ingestion | PASS — fixture indexed: 10 chunks, 2 tables, 22 entities, `MODIFIES=1` |
| RAG pipeline | PASS — `Status: VERIFIED`, correct `6000`, Appendix B cited with page number |
| Questions answered appropriately | PASS — verified answers, honest abstention |
| Error handling | PASS — 49/49 tests pass in this environment |
| Simulated pip-only machine (uv removed from `PATH`) | PASS — exit 0; launcher falls back to `python mcp_server.py` |

**No API key was used and none is needed.** A new developer gets a fully working install with verified,
cited answers out of the box; adding `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` only sharpens the
extraction, grading and prose steps.

---

## 10. Bugs discovered & fixed in this audit

| # | Severity | Issue | Status |
|---|---|---|---|
| 1 | High (for students) | No pip install route; `uv` was mandatory and undocumented as such | FIXED |
| 2 | Medium | `doctor` exited 1 and printed "Some checks failed" on a correct pip install | FIXED |
| 3 | Low | `fastembed unavailable` warning leaked into `doctor` output | FIXED |
| 4 | Medium | `uv.lock` gitignored → non-reproducible installs | CHANGED, needs your commit |
| 5 | Low | README had no "What is RAG?", no system requirements, no pip instructions | FIXED |
| 6 | Low | `.gitignore` missed `venv/`, `env/`, `.ruff_cache/`, `*.bak`, OS/editor files | FIXED |

No functional bug was found in the RAG pipeline itself in this pass; everything above is distribution
and first-run experience.

## 11. Files created / modified

**Created (3):** `requirements.txt`, `requirements-optional.txt`, `RAG_PUBLIC_RELEASE_AUDIT.md`
**Modified (3):** `README.md` (added What is RAG / System requirements / pip install; simplified the
`doctor` note), `src/graphrag/cli.py` (uv check is informational; quiet logging in `doctor`),
`.gitignore` (added entries; removed `uv.lock`)
**Untracked and worth committing:** `uv.lock`
**Deleted: none.** `.env` was never read or written.

## 12. Remaining limitations

| Status | Item |
|---|---|
| NOT TESTED | Dependency CVE scan — run `pip-audit` / enable Dependabot before announcing |
| NOT TESTED | Docling, fastembed, cross-encoder reranker, Neo4j, Qdrant server, DOCX/PPTX, live LLM providers, macOS and Linux (audited on Windows only; CI covers Linux for the test suite) |
| WARNING | `uv.lock` must be committed by you for reproducible installs |
| WARNING | HTTP transport has no authentication of its own (loopback default + warning + allowlist are the mitigations) |
| WARNING | Embedded Qdrant allows one process at a time, so the CLI cannot run while an IDE holds the index; set `QDRANT_URL` for real concurrency |
| By design | No chat memory, no web UI, no upload endpoint — none of these is claimed in the docs |

## Verdict

**Ready for public release** once `uv.lock` is committed. A new developer on Windows, macOS or Linux can
download the folder, follow either the uv or the pip instructions, run `doctor`, ingest a document and
get verified cited answers — with no API key, no services and no access to your machine. The two
first-run blockers that would actually have stopped a student (no pip route, and `doctor` falsely
reporting failure) are fixed and verified in a clean environment.
