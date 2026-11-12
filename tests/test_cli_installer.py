import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from graphrag import installer

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "acme_gateway_spec.md"


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("USERPROFILE", str(h))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(installer.shutil, "which", lambda _x: None)  # no client CLIs, no uv -> deterministic
    return h


@pytest.mark.parametrize("client,rel,key", [
    ("cursor", ".cursor/mcp.json", "mcpServers"),
    ("antigravity", ".gemini/config/mcp_config.json", "mcpServers"),
    ("windsurf", ".codeium/windsurf/mcp_config.json", "mcpServers"),
])
def test_user_json_clients_merge(home, client, rel, key):
    path = home / rel
    path.parent.mkdir(parents=True)
    path.write_text('{\n  // comment\n  "mcpServers": {"other": {"command": "keep"}},\n}\n')  # JSONC
    installer.install(client, "user")
    data = json.loads(path.read_text())
    assert data[key]["other"] == {"command": "keep"}
    entry = data[key][installer.SERVER_NAME]
    assert entry["command"] == sys.executable and entry["args"][0].endswith("mcp_server.py")
    assert path.with_suffix(".json.bak").exists()


def test_antigravity_updates_legacy_file_when_present(home):
    legacy = home / ".gemini/antigravity/mcp_config.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("{}")
    installer.install("antigravity", "user")
    assert installer.SERVER_NAME in json.loads(legacy.read_text())["mcpServers"]


def test_codex_toml_merge_idempotent_and_uninstall(home):
    cfg = home / ".codex/config.toml"
    cfg.parent.mkdir()
    cfg.write_text('model = "gpt-5"\n\n[mcp_servers.other]\ncommand = "x"\n')
    installer.install("codex", "user", data_dir="/data/idx")
    installer.install("codex", "user", data_dir="/data/idx")
    parsed = tomllib.loads(cfg.read_text())
    srv = parsed["mcp_servers"][installer.SERVER_NAME]
    assert parsed["model"] == "gpt-5" and "other" in parsed["mcp_servers"]
    assert srv["tool_timeout_sec"] == 900 and srv["env"]["GRAPHRAG_DATA_DIR"] == "/data/idx"
    assert cfg.read_text().count(f"[mcp_servers.{installer.SERVER_NAME}]") == 1
    installer.uninstall("codex", "user")
    parsed = tomllib.loads(cfg.read_text())
    assert installer.SERVER_NAME not in parsed["mcp_servers"] and "other" in parsed["mcp_servers"]


def test_project_scope_files(home, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    for c in ["claude-code", "cursor", "vscode", "codex", "antigravity"]:
        installer.install(c, "project", proj)
    assert installer.SERVER_NAME in json.loads((proj / ".mcp.json").read_text())["mcpServers"]
    assert installer.SERVER_NAME in json.loads((proj / ".cursor/mcp.json").read_text())["mcpServers"]
    vs = json.loads((proj / ".vscode/mcp.json").read_text())["servers"][installer.SERVER_NAME]
    assert vs["type"] == "stdio"
    assert installer.SERVER_NAME in tomllib.loads((proj / ".codex/config.toml").read_text())["mcp_servers"]
    assert installer.SERVER_NAME in json.loads((proj / ".agents/mcp_config.json").read_text())["mcpServers"]


def test_cli_based_clients_dry_run(home):
    out = installer.install("claude-code", "user", dry_run=True)
    assert out.startswith("$ claude mcp add --scope user graphrag-reader --")
    out = installer.install("vscode", "user", dry_run=True)
    assert "code --add-mcp" in out and '"type": "stdio"' in out
    with pytest.raises(RuntimeError):
        installer.install("claude-code", "user")  # `claude` not on PATH


def test_snippets_cover_all_clients(home):
    s = installer.snippets()
    for marker in ["claude mcp add", "codex mcp add", "[mcp_servers.graphrag-reader]", "code --add-mcp",
                   '"servers"', '"mcpServers"', "Antigravity"]:
        assert marker in s


def test_terminal_cli_end_to_end(tmp_path):
    env = {"GRAPHRAG_DATA_DIR": str(tmp_path / "idx"), "GRAPHRAG_EMBEDDER": "hash", "GRAPHRAG_LLM_PROVIDER": "none",
           "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    run = lambda *a, **kw: subprocess.run([sys.executable, str(ROOT / "graphrag_cli.py"), *a],
                                          capture_output=True, text=True, env=env, **kw)
    r = run("ingest", str(FIXTURE), "--domain", "technical")
    assert r.returncode == 0 and "indexed" in r.stdout
    r = run("--plain", "ask", "What is max_requests_per_minute for Enterprise keys?")
    assert "VERIFIED" in r.stdout and "6000" in r.stdout and "**" not in r.stdout
    assert "Traceback" not in r.stderr
    r = run("ask", "--json", "What is the default burst_size?")
    assert json.loads(r.stdout)["status"] == "verified"
    r = run("trace", "Appendix B", "--hops", "1")
    assert "MODIFIES" in r.stdout
    r = run("chat", input=":docs\nWhat uptime SLA does the gateway guarantee?\n:quit\n")
    assert "acme_gateway_spec" in r.stdout and "ABSTAINED" in r.stdout
