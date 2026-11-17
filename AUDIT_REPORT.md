# Audit & Test Report — graphrag-reader (self-corrective-graphrag)

Date: 2026-11-17 · Branch: `main` (no commits made) · Platform: Windows 11, Python 3.13.12, uv
Mode exercised: **lite** (embedded Qdrant, NetworkX graph, hashed embedder, lexical reranker, no LLM)

---

## 1. Project architecture overview

There is **no frontend, no HTTP API for application traffic, and no SQL database**. The project is a
local document-intelligence engine exposed two ways:

| Layer | Implementation |
|---|---|
| Entry points | `mcp_server.py` (MCP stdio/HTTP), `graphrag_cli.py` → `src/graphrag/cli.py` (terminal CLI) |
| MCP tools | `src/graphrag/server.py`: `ingest_complex_doc`, `ask_verified_docs`, `trace_entity_relations`, `extract_verified_table`, `list_indexed_docs` |
| Facade | `src/graphrag/engine.py` (`GraphRAGEngine`) + `config.py` (all config from env vars) |
| Ingestion | `layout_parser.py` (Docling → markdown/HTML native → pypdf fallback) → `chunker.py` (section-aware, tables atomic, metadata header) → `graph_extractor.py` (deterministic structural + optional LLM, ontology-constrained) → `entity_resolver.py` (fiscal-period/section/fuzzy canonicalisation) |
| Vector store | `indexing/qdrant_store.py` — Qdrant, embedded on-disk by default, server via `QDRANT_URL`; named dense vector + BM25 sparse vector per chunk |
| Embeddings | `indexing/embeddings.py` — fastembed (optional) or deterministic hashed fallback; BM25 sparse encoder with Qdrant-side IDF |
| Graph store | `indexing/graph_store.py` — NetworkX persisted as JSON (default) or Neo4j/Memgraph |
| Retrieval | `retrieval/hybrid.py` — dense + BM25 + entity-linked k-hop graph leg, fused with RRF, reranked (`reranker.py`: cross-encoder or lexical), with override (`MODIFIES`) evidence pinned |
| Agent loop | `agent/workflow.py` (LangGraph) over `agent/nodes.py`: route/decompose → retrieve → CRAG grade → rewrite+hop-expand (≤2) → generate → Self-RAG verify → regenerate (≤1) → finalize / abstain |
| Math | `agent/sandbox.py` — AST-walking evaluator, no `eval`/`exec` |
| Installer | `installer.py` — writes MCP config for 7 clients (JSON/JSONC/TOML merge, `.bak` backups) |
| Evaluation | `evaluation/run_benchmark.py` — naive vs hybrid vs full ablation |

External services are **all optional**; every component degrades to an offline deterministic fallback.
Secrets come only from env vars (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `QDRANT_API_KEY`, `NEO4J_PASSWORD`).

---

## 2. Features tested

Startup/shutdown · MCP stdio handshake and all 5 tools · streamable-HTTP transport · CLI
(`ingest`/`ask`/`trace`/`table`/`docs`/`chat`/`doctor`/`serve`/`install`) · ingestion of MD, TXT, HTML,
PDF · text extraction and preprocessing · chunking (incl. atomic + split tables) · embedding generation
· vector upsert/search/retrieve/scroll/delete · hybrid dense+BM25+graph retrieval with RRF · reranking ·
CRAG grading, query rewrite, hop expansion · extractive generation · hallucination / negation /
accounting-sign verification · abstention · citations · entity resolution and graph traversal · table
extraction + deterministic compute · doc filters · duplicate/force re-ingest · client-config installer ·
evaluation benchmark · error handling and graceful degradation.

**NOT TESTED** (optional extras not installed, no network/keys used): Docling parser, fastembed dense
embeddings, cross-encoder reranker, Anthropic/OpenAI LLM paths beyond the existing mocked unit test,
Qdrant server mode, Neo4j/Memgraph graph store, DOCX/PPTX ingestion, FinanceBench dataset,
dependency CVE scanning (no vulnerability database consulted).

---

## 3–5. Test case totals

| Suite | Executed | Passed | Failed |
|---|---|---|---|
| `pytest` (repo suite, after fixes) | **49** | 49 | 0 |
| Functional / RAG / edge-case harness | **58** | 58 | 0 |
| Security probes | **14** | 14 | 0 |
| Manual CLI / transport / lock / benchmark | **9** | 9 | 0 |
| **Total** | **130** | **130** | **0** |

Baseline before fixes: `pytest` 33 passed / **2 failed**; harness 56 passed / **2 failed**.

### RAG pipeline results (Phase 3)

| Question class | Result |
|---|---|
| Directly answered (`default burst_size`) | PASS — `verified`, correct value, cited |
| Multi-section / multi-hop (Enterprise override in Appendix B, 123 pages from the default) | PASS — `verified`, 6000, Appendix B cited |
| Override surfaced when the query never names it | PASS — graph `MODIFIES` edge pinned into evidence |
| Unrelated to the corpus ("capital of France") | PASS — `abstained` after 3 rounds |
| Information genuinely missing (uptime SLA) | PASS — `abstained`, missing terms + nearest sections reported |
| Ambiguous ("what about limits") | PASS — answers, no fabrication |
| Long compound question (3 sections + a negative condition) | PASS — `partially_verified`, correct DELETE carve-out |
| Negation ("are webhook signatures encrypted?") | PASS — preserves "not encrypted" |
| Accounting sign | PASS — `(0.3)` preserved, not rendered as 0.3 |
| Cross-document (2 specs sharing a parameter name) | PASS — no bleed; `doc_filter` isolates correctly |
| Citation integrity | PASS — every `[E#]` in the answer maps to a returned citation, and the cited chunk really contains the claimed number |
| Hallucination resistance | PASS — invented number → `HALLUCINATED`, claim stripped; ungrounded `compute` operands flagged `all_operands_grounded=false` |

No hallucination, no irrelevant-retrieval failure and no missing-citation case was observed in 23 query tests.

---

## 6–7. Bugs identified and fixed

All items marked **FIXED** were fixed, re-tested and are covered by new automated tests.

### High

**H1 — Unauthenticated HTTP transport + unrestricted file read.** `GRAPHRAG_TRANSPORT=streamable-http`
serves all tools with **no authentication** (verified: an unauthenticated `initialize` returned 200), and
`ingest_complex_doc` accepts any absolute path the process can read, whose content is then retrievable
via `ask_verified_docs`. On loopback (the default) this is the intended local design; bound to a routable
interface it is a remote arbitrary-file-read and exfiltration primitive.
**FIXED (defence in depth):** a loud stderr warning when binding a non-loopback host, and a new opt-in
`GRAPHRAG_ALLOWED_DIRS` allowlist enforced in `engine.ingest` (unset = previous behaviour, so nothing
breaks). Documented in `.env.example`.

**H2 — Ingestion crashes on any document outside the locale encoding (Windows).**
`engine.py` wrote the parsed-markdown copy with `write_text()` and no encoding, so on Windows (cp1252)
ingesting a document containing CJK, Cyrillic, Greek, emoji or most maths symbols raised
`UnicodeEncodeError` and aborted the ingest. Reproduced. Same class of defect in the registry JSON and
the NetworkX graph JSON.
**FIXED:** explicit `encoding="utf-8"` on every read/write; `errors="replace"` for the parsed copy, since a
document may contain any code point. Regression test ingests a Japanese/Cyrillic/Greek/emoji spec and
answers a question from it.

### Medium

**M1 — A contentless question was reported as `verified`.** `coverage()` returned `1.0` when the question
contained no key terms, so `""`, `"   "` and `"???"` graded `RELEVANT` and returned
`Status: VERIFIED` with arbitrary document sentences — the worst possible failure mode for a tool whose
contract is "trust the status". **FIXED:** no key terms → coverage `0.0`; plus an explicit blank-query
guard in `engine.ask`. All three now `abstained`.

**M2 — Document text reaches the host agent with no data/instruction boundary.** Confirmed by probe: a
document containing "IGNORE ALL PREVIOUS INSTRUCTIONS…" is quoted verbatim in tool output (correct for
an extractive engine — but unframed). The loop itself was not subverted (the injected doc produced
`abstained` for its own topic, and no instruction was ever acted on). **FIXED (mitigation):** the server
`instructions` now tell the host that quoted document text is data to report on, never instructions.

**M3 — Two tests could not run on Windows.** `test_terminal_cli_end_to_end` and `test_doctor_passes` built
a subprocess env from scratch with `PATH=/usr/bin:/bin` and `HOME`; without `SystemRoot` winsock fails
to load (`WinError 10106`) and without `USERPROFILE` `Path.home()` raises. The CLI end-to-end and doctor
coverage was therefore dead on the platform the README documents first. **FIXED:** shared `_clean_env`
helper inherits `os.environ` and pins only what the test controls.

**M4 — Installer corrupts/crashes on non-ASCII client configs.** `installer.py` read and wrote
`mcp.json` / `config.toml` with the locale encoding; a config containing a non-ASCII path (common
outside en-US) would mangle or raise. **FIXED:** `encoding="utf-8"` on all six call sites.

**M5 — `docker-compose.yml` published unauthenticated datastores on all interfaces.** Qdrant (no API key)
and Neo4j (dev password `graphrag-dev`) were mapped as `6333:6333` / `7687:7687`, i.e. `0.0.0.0` —
anyone on the LAN could read or overwrite the index and the knowledge graph. **FIXED:** ports bound to
`127.0.0.1`, with a comment stating neither service is authenticated and what to set before exposing them.

**M6 — Degenerate configuration silently accepted or failed opaquely.** `GRAPHRAG_FINAL_K=0` made every
answer abstain with no explanation; `GRAPHRAG_CHUNK_TOKENS=-5` was accepted; `GRAPHRAG_VECTOR_K=abc`
raised a bare `invalid literal for int()` naming no variable. **FIXED:** range-checked parsing that names
the offending variable, with minimums per setting.

### Low

| ID | Issue | Status |
|---|---|---|
| L1 | Empty/unparseable document indexed as plain `indexed` with 0 chunks and no warning | **FIXED** — status now says no text was extracted |
| L2 | `trace_entity_relations`, `extract_verified_table`, `list_indexed_docs` lacked the broad `except` the other tools have → raw traceback instead of a readable tool error | **FIXED** |
| L3 | `ingest` read the whole file into memory for its SHA-256 | **FIXED** — streamed in 1 MiB blocks |
| L4 | Embedded Qdrant's exclusive lock produced a generic "already accessed" error when the CLI ran against the index an IDE's MCP server held | **FIXED** — message now names the cause and the three ways out |
| L5 | `evaluation/run_benchmark.py` read/wrote JSON with the locale encoding | **FIXED** |
| L6 | `uv.lock` is in `.gitignore`, so CI resolves fresh dependency versions on every run (supply-chain drift; no reproducible builds) | **REPORTED** — committing the lock file is your call |
| L7 | `installer._backup` overwrites the previous `.bak` on each run | **REPORTED** |
| L8 | Benchmark's `naive`/`hybrid` configs score `numeric_accuracy` against the concatenated chunk text, so they trivially "match" any number present — the ablation flatters the weak baselines (`naive 1.0` vs `full 1.0` on the sample set) | **REPORTED** — methodology, not a defect in the app |

### Verified secure (no action needed)

- **Math sandbox:** AST walk, no `eval`/`exec`. `__import__('os').system(...)`, `open(...)`, lambdas,
  subscripts, names, `2**10000` all rejected (`disallowed syntax: Call/Name`); magnitude and exponent bounded.
- **Path traversal:** `doc_name="../../../../pwned"` is sanitised by the `doc_id` slug; no file was written
  outside the data directory.
- **HTML ingestion:** `<script>`/`<style>` bodies and CSS are stripped before indexing — no script tag,
  `alert(` or stylesheet content reached the index.
- **Secrets:** no hardcoded credentials anywhere in the repo (pattern-scanned); `.env` is gitignored and was
  never read or modified; keys are only read from the environment and never logged — logs carry paths and
  exception types only. `capabilities()` exposes provider/model names, not keys.
- **SQL injection:** not applicable — no SQL anywhere. Qdrant filters are built from typed model objects, and
  the Cypher in `Neo4jGraphStore` is fully parameterised (no string interpolation of user data).
- **XSS / CSRF:** not applicable — no browser-facing surface. Output is markdown consumed by an agent.
- **Malicious / corrupt files:** empty PDF, truncated PDF, binary-garbage `.md`, unsupported extension,
  missing path and a directory path all fail cleanly with typed errors; nothing crashes the process.
- **Query-side injection:** SQL-ish, path-ish, 12 000-token and mixed-script queries all handled; the engine
  abstains or answers normally.

---

## 8. Performance

Measured in lite mode on this machine (hash embedder + lexical reranker — no model inference in the path).

| Metric | Result |
|---|---|
| Engine init (cold) | 52–53 ms |
| Ingest 2.6 KB / 10-chunk markdown | 47 ms |
| Ingest 131 KB / 133-chunk markdown (148 entities) | 0.74 s |
| Ingest 4-page PDF via pypdf | 258 ms (1.97 s on the first call, pypdf import) |
| `ask` latency, warm | 2–20 ms; benchmark p50 3.4 ms / p95 8.0 ms |
| `ask` after indexing 7 documents / 133-chunk corpus | 5 ms, still correct |
| Pathological 12 000-token query | 147 ms |
| 8 concurrent `ask` calls (threads) | 0.04 s total, 0 errors |
| 3 concurrent `ingest` + 3 concurrent `ask` | 0 errors (`engine._lock` serialises writes) |

Bottlenecks and scaling notes:

- **FIXED:** whole-file read for the content hash (a 500 MB filing no longer lands in RAM).
- **WARNING — startup is O(graph size):** `GraphRAGEngine.__init__` loads every entity into the resolver,
  and `NetworkXGraphStore` loads the whole graph JSON. Fine at hundreds of entities; move to Neo4j
  (`NEO4J_URI`) before tens of thousands.
- **WARNING — each ingest rewrites the whole graph JSON** and recomputes Louvain communities over the
  entire graph. It is once per document, not per chunk, so it is acceptable now, but ingest cost grows with
  total corpus size, not document size.
- **WARNING — embedded Qdrant allows one process at a time.** Running the CLI while an IDE holds the index
  is impossible by design; the error now explains it. Use `QDRANT_URL` for genuine concurrency.
- **NOT TESTED:** latency with fastembed and the cross-encoder reranker, where model inference (not this
  code) dominates.

---

## 9. Files modified (13 — no files deleted, no `.env` touched, nothing committed)

| File | Change |
|---|---|
| `src/graphrag/config.py` | range-checked env parsing with named errors; `GRAPHRAG_ALLOWED_DIRS` + `check_ingest_path()` |
| `src/graphrag/engine.py` | utf-8 encodings; streamed SHA-256; ingest-path allowlist; blank-query guard; empty-document warning |
| `src/graphrag/server.py` | injection-boundary note in `instructions`; error handling on 3 tools; non-loopback bind warning |
| `src/graphrag/agent/nodes.py` | `coverage()` returns 0.0 for a question with no key terms |
| `src/graphrag/indexing/qdrant_store.py` | actionable message for the embedded-store lock |
| `src/graphrag/indexing/graph_store.py` | utf-8 graph JSON read/write |
| `src/graphrag/installer.py` | utf-8 config read/write (6 sites) |
| `evaluation/run_benchmark.py` | utf-8 JSON read/write |
| `docker-compose.yml` | loopback port bindings + security note |
| `.env.example` | documents `GRAPHRAG_ALLOWED_DIRS` and the unauthenticated HTTP transport |
| `tests/test_cli_installer.py` | `_clean_env` helper — tests run on Windows |
| `tests/test_units.py` | +9 tests: config validation, allowlist, coverage floor |
| `tests/test_pipeline.py` | +6 tests: contentless queries, non-Latin-1 document, empty document, allowlist |

---

## 10. Remaining issues

| Severity | Issue |
|---|---|
| Medium (accepted design) | The HTTP transport still has no authentication of its own. The warning and `GRAPHRAG_ALLOWED_DIRS` are mitigations; a routable deployment needs an authenticating reverse proxy. Adding auth to the transport is a feature decision, not a bug fix, so it was left to you. |
| Medium (inherent) | Prompt injection via ingested documents cannot be eliminated by a retrieval tool that must quote its sources. The server now tells the host that quoted text is data; the residual risk sits with the host agent. |
| Low | `uv.lock` untracked (L6) and `.bak` overwriting (L7) — both left as-is pending your decision. |
| Low | Benchmark ablation methodology (L8). |
| Not tested | Docling, fastembed, cross-encoder, Neo4j, Qdrant server, DOCX/PPTX, live LLM providers, dependency CVEs. |

---

## 11. Final project health assessment

**PASS — healthy.** The architecture is coherent and unusually disciplined for a RAG project: every
external dependency is optional with a deterministic fallback, the retrieval loop is genuinely
self-correcting, arithmetic never goes through the model, and the verifier demonstrably strips invented
numbers and catches inverted signs and negations. Abstention works and is honest — which is the property
that matters most in a tool an agent is told to trust.

The defects found were concentrated at the edges rather than in the core: Windows encoding handling (which
broke ingestion outright for non-Western documents), one genuine correctness bug where a contentless
question was labelled `verified`, missing input validation on configuration, and deployment hardening of
the optional HTTP transport and docker-compose stack. All of those are fixed, with 15 new tests pinning
them. 130 of 130 test cases now pass, and the pre-existing behaviour is unchanged (all original tests
still pass; the full pipeline, CLI, MCP stdio server and benchmark were each re-verified after the fixes).

The two items worth your attention next are authentication for the HTTP transport (if you intend to use it
off-host) and committing `uv.lock` for reproducible dependency resolution.
