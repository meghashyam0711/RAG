# Agent instructions: graphrag-reader

This repo is an MCP server and CLI for verified, cited answers from large documents (PDF, DOCX, HTML, Markdown).

## If the user asks you to install or set it up

Run exactly one of these. Each installs uv if needed, gets the code, installs dependencies, runs a self-test and registers the MCP server with every detected agent/IDE:

- Windows (PowerShell): `irm https://raw.githubusercontent.com/meghashyam0711/RAG/main/install.ps1 | iex`
- macOS / Linux: `curl -LsSf https://raw.githubusercontent.com/meghashyam0711/RAG/main/install.sh | sh`

To register only yourself, set `GRAPHRAG_CLIENTS` first (`codex`, `cursor`, `vscode`, `antigravity`, `claude-code`, `windsurf`, `claude-desktop`), for example in PowerShell: `$env:GRAPHRAG_CLIENTS="codex"`.

If you are already inside this repo: `uv sync`, then `uv run graphrag doctor`, then `uv run graphrag install <client>`.

Rules:
- On Windows PowerShell 5.1, never chain commands with `&&`. Run one command per call.
- If `uv` or `git` is "not recognized" right after installing it, reload PATH:
  `$env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")`
- Success = `uv run graphrag doctor` prints "All checks passed." Tell the user to restart their agent/IDE.

## Using the tools (saves tokens)

Do not read large PDFs or specs into context. Instead:
1. `ingest_complex_doc` once per file (`domain_tag`: technical, finance, legal or general).
2. `ask_verified_docs` for questions. Trust `ABSTAINED` (the documents don't say it), and don't guess.
3. `extract_verified_table` for exact tables. Do math with its `compute` argument, not in your head.
4. `trace_entity_relations` to find where a parameter is defined or overridden across sections.

Leave `verbose=false` (the default) unless debugging retrieval.

## Development

`uv sync --extra dev`, then `uv run pytest -q`. Code is in `src/graphrag/` (`server.py` holds the MCP tools, `cli.py` the terminal CLI, `installer.py` the client configs).
