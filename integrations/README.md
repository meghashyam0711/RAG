# Per-client MCP configs

The easy way is `graphrag install <client>` (or `graphrag install all`), which writes these for you with the correct
absolute path and keeps your existing servers. These files are for manual setup: replace
`/ABSOLUTE/PATH/TO/self-corrective-graphrag` with the full path of your clone of
https://github.com/meghashyam0711/RAG, e.g. `C:/Users/<name>/RAG` on Windows (use forward slashes or `\\` in JSON)
or `/home/<name>/RAG` on macOS/Linux.

| Client | File here | Copy to (user / all projects) | Copy to (project) | Top-level key |
|---|---|---|---|---|
| Claude Code | `claude-code.mcp.json` | `claude mcp add graphrag-reader -- uv run --directory <repo> mcp_server.py` | `.mcp.json` | `mcpServers` |
| Cursor | `cursor.mcp.json` | `~/.cursor/mcp.json` | `.cursor/mcp.json` | `mcpServers` |
| VS Code (Copilot agent mode) | `vscode.mcp.json` | `code --add-mcp '{...}'` or *MCP: Add Server* | `.vscode/mcp.json` | **`servers`** + `"type": "stdio"` |
| Codex (CLI / IDE ext / app) | `codex.config.toml` | `~/.codex/config.toml` or `codex mcp add graphrag-reader -- uv run --directory <repo> mcp_server.py` | `.codex/config.toml` (trusted projects) | `[mcp_servers.<name>]` (TOML) |
| Antigravity (IDE / CLI / app) | `antigravity.mcp_config.json` | `~/.gemini/config/mcp_config.json` (Windows: `%USERPROFILE%\.gemini\config\`) | `.agents/mcp_config.json` | `mcpServers` |
| Windsurf | `windsurf.mcp_config.json` | `~/.codeium/windsurf/mcp_config.json` | - | `mcpServers` |
| Claude Desktop | `claude-desktop.json` | macOS `~/Library/Application Support/Claude/claude_desktop_config.json`, Windows `%APPDATA%\Claude\claude_desktop_config.json` | - | `mcpServers` |

Remote/shared server (`graphrag serve --http`): most clients take `"url": "http://host:8765/mcp"`; Antigravity
uses `"serverUrl"` instead (see `antigravity.remote.mcp_config.json`); VS Code uses `"type": "http", "url": ...`;
Codex uses `codex mcp add graphrag-reader --url http://host:8765/mcp`.

No `uv`? Use `"command": "python", "args": ["<repo>/mcp_server.py"]` with the Python that has the deps installed,
or `pip install .` and use `"command": "graphrag-mcp", "args": []`.
