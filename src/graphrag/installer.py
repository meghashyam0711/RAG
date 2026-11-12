"""Register the MCP server with coding agents and IDEs.

    graphrag install claude-code | cursor | vscode | codex | antigravity | windsurf | claude-desktop | all
                     [--scope user|project] [--project-dir DIR] [--print] [--launcher uv|script|python]

Every write merges into the existing config (other servers are preserved) and
leaves a ``.bak`` copy of the previous file. ``--print`` only shows what would
be written.

Config formats (verified Nov 2026):
  Claude Code     CLI `claude mcp add` (user) | .mcp.json  {"mcpServers": {...}} (project)
  Cursor          ~/.cursor/mcp.json | .cursor/mcp.json     {"mcpServers": {...}}
  VS Code         CLI `code --add-mcp` (user) | .vscode/mcp.json {"servers": {name: {"type": "stdio", ...}}}
  Codex           ~/.codex/config.toml | .codex/config.toml  [mcp_servers.<name>]  (CLI, IDE ext. and app share it)
  Antigravity     ~/.gemini/config/mcp_config.json | .agents/mcp_config.json  {"mcpServers": {...}}
                  (legacy ~/.gemini/antigravity/mcp_config.json also updated if present)
  Windsurf        ~/.codeium/windsurf/mcp_config.json         {"mcpServers": {...}}
  Claude Desktop  OS-specific claude_desktop_config.json      {"mcpServers": {...}}
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SERVER_NAME = "graphrag-reader"
REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENTS = ["claude-code", "cursor", "vscode", "codex", "antigravity", "windsurf", "claude-desktop"]


@dataclass
class Launch:
    command: str
    args: list[str]
    env: dict[str, str]


def resolve_launch(launcher: str = "auto", data_dir: str | None = None) -> Launch:
    """How the client should start the server.

    uv     -> uv run --directory <repo> mcp_server.py   (git clone + uv; recommended)
    script -> graphrag-mcp                               (pip / uv tool / pipx install)
    python -> <this python> <repo>/mcp_server.py         (no uv available)
    """
    env = {"GRAPHRAG_DATA_DIR": data_dir} if data_dir else {}
    in_repo = (REPO_ROOT / "mcp_server.py").exists()
    if launcher == "auto":
        if in_repo and shutil.which("uv"):
            launcher = "uv"
        elif shutil.which("graphrag-mcp"):
            launcher = "script"
        else:
            launcher = "python"
    if launcher == "uv":
        if not in_repo:
            raise SystemExit("--launcher uv needs the git checkout (mcp_server.py not found).")
        # Absolute path: GUI apps (VS Code, Cursor, Antigravity, Claude Desktop) often start with a PATH
        # that predates the uv install, so a bare "uv" fails with "command not found".
        return Launch(shutil.which("uv") or "uv", ["run", "--directory", str(REPO_ROOT), "mcp_server.py"], env)
    if launcher == "script":
        exe = shutil.which("graphrag-mcp") or "graphrag-mcp"
        return Launch(exe, [], env)
    if in_repo:
        return Launch(sys.executable, [str(REPO_ROOT / "mcp_server.py")], env)
    return Launch(sys.executable, ["-m", "graphrag.server"], env)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _claude_desktop_path() -> Path:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    if system == "Windows":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming")) / "Claude/claude_desktop_config.json"
    return Path.home() / ".config/Claude/claude_desktop_config.json"


def config_path(client: str, scope: str, project_dir: Path) -> Path | None:
    home = Path.home()
    table = {
        ("claude-code", "project"): project_dir / ".mcp.json",
        ("cursor", "user"): home / ".cursor/mcp.json",
        ("cursor", "project"): project_dir / ".cursor/mcp.json",
        ("vscode", "project"): project_dir / ".vscode/mcp.json",
        ("codex", "user"): Path(os.environ.get("CODEX_HOME", home / ".codex")) / "config.toml",
        ("codex", "project"): project_dir / ".codex/config.toml",
        ("antigravity", "user"): home / ".gemini/config/mcp_config.json",
        ("antigravity", "project"): project_dir / ".agents/mcp_config.json",
        ("windsurf", "user"): home / ".codeium/windsurf/mcp_config.json",
        ("claude-desktop", "user"): _claude_desktop_path(),
    }
    return table.get((client, scope))


def detect_clients() -> list[str]:
    home = Path.home()
    found = []
    checks = {
        "claude-code": shutil.which("claude") or (home / ".claude").exists(),
        "cursor": shutil.which("cursor") or (home / ".cursor").exists(),
        "vscode": shutil.which("code") or (home / ".vscode").exists(),
        "codex": shutil.which("codex") or (home / ".codex").exists(),
        "antigravity": shutil.which("antigravity") or shutil.which("agy") or (home / ".gemini").exists(),
        "windsurf": shutil.which("windsurf") or (home / ".codeium/windsurf").exists(),
        "claude-desktop": _claude_desktop_path().parent.exists(),
    }
    for client, ok in checks.items():
        if ok:
            found.append(client)
    return found


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _entry_mcpservers(l: Launch) -> dict:
    e: dict = {"command": l.command, "args": l.args}
    if l.env:
        e["env"] = l.env
    return e


def _entry_vscode(l: Launch) -> dict:
    return {"type": "stdio", **_entry_mcpservers(l)}


def _backup(path: Path) -> None:
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))


def _load_json(path: Path) -> dict:
    if not path.exists() or not path.read_text().strip():
        return {}
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # VS Code / Cursor allow comments and trailing commas (JSONC).
        stripped = re.sub(r"(?m)^\s*//.*$", "", text)
        stripped = re.sub(r",(\s*[}\]])", r"\1", stripped)
        return json.loads(stripped)


def merge_json(path: Path, key: str, entry: dict, dry_run: bool) -> str:
    data = _load_json(path)
    data.setdefault(key, {})[SERVER_NAME] = entry
    text = json.dumps(data, indent=2) + "\n"
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        _backup(path)
        path.write_text(text)
    return text


def _toml_str(s: str) -> str:
    return json.dumps(s)  # JSON string escaping is valid TOML basic-string escaping


def codex_toml_block(l: Launch) -> str:
    lines = [
        f'[mcp_servers.{SERVER_NAME}]',
        f"command = {_toml_str(l.command)}",
        "args = [" + ", ".join(_toml_str(a) for a in l.args) + "]",
        "startup_timeout_sec = 60",
        "tool_timeout_sec = 900   # ingesting a large PDF can take minutes",
    ]
    if l.env:
        lines += ["", f"[mcp_servers.{SERVER_NAME}.env]"] + [f"{k} = {_toml_str(v)}" for k, v in l.env.items()]
    return "\n".join(lines) + "\n"


def merge_toml(path: Path, block: str, dry_run: bool) -> str:
    """Replace any existing [mcp_servers.graphrag-reader] (and sub-tables) with `block`, keep everything else."""
    text = path.read_text() if path.exists() else ""
    out, skipping = [], False
    header_re = re.compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*(#.*)?$")
    for line in text.splitlines():
        m = header_re.match(line)
        if m:
            name = m.group(1).replace('"', "").replace("'", "")
            skipping = name == f"mcp_servers.{SERVER_NAME}" or name.startswith(f"mcp_servers.{SERVER_NAME}.")
        if not skipping:
            out.append(line)
    body = "\n".join(out).rstrip()
    new = (body + "\n\n" if body else "") + block
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        _backup(path)
        path.write_text(new)
    return new


def _run(cmd: list[str], dry_run: bool) -> str:
    shown = " ".join(_shell_quote(c) for c in cmd)
    if dry_run:
        return f"$ {shown}"
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{shown}\n{res.stderr.strip() or res.stdout.strip()}")
    return f"$ {shown}\n{res.stdout.strip()}"


def _shell_quote(s: str) -> str:
    return s if re.fullmatch(r"[\w@%+=:,./\-]+", s) else "'" + s.replace("'", "'\\''") + "'"


# ---------------------------------------------------------------------------
def install(client: str, scope: str = "user", project_dir: Path | None = None, launcher: str = "auto",
            data_dir: str | None = None, dry_run: bool = False) -> str:
    project_dir = (project_dir or Path.cwd()).resolve()
    l = resolve_launch(launcher, data_dir)

    if client == "claude-code" and scope == "user":
        if shutil.which("claude") or dry_run:
            cmd = ["claude", "mcp", "add", "--scope", "user"]
            for k, v in l.env.items():
                cmd += ["-e", f"{k}={v}"]
            return _run(cmd + [SERVER_NAME, "--", l.command, *l.args], dry_run)
        raise RuntimeError("`claude` CLI not found. Install Claude Code, or use --scope project to write .mcp.json.")

    if client == "vscode" and scope == "user":
        if shutil.which("code") or dry_run:
            payload = json.dumps({"name": SERVER_NAME, **_entry_vscode(l)})
            return _run(["code", "--add-mcp", payload], dry_run)
        raise RuntimeError("`code` CLI not found. Run VS Code 'Shell Command: Install code command in PATH', "
                           "or use --scope project to write .vscode/mcp.json.")

    path = config_path(client, scope, project_dir)
    if path is None:
        raise RuntimeError(f"{client} has no {scope}-scope config; try --scope {'project' if scope == 'user' else 'user'}.")

    if client == "codex":
        text = merge_toml(path, codex_toml_block(l), dry_run)
        note = "" if scope == "user" else "\n(Codex loads project config only for trusted projects.)"
        return f"{'would write' if dry_run else 'wrote'} {path}{note}\n\n{text if dry_run else codex_toml_block(l)}"

    key = "servers" if client == "vscode" else "mcpServers"
    entry = _entry_vscode(l) if client == "vscode" else _entry_mcpservers(l)
    text = merge_json(path, key, entry, dry_run)
    msgs = [f"{'would write' if dry_run else 'wrote'} {path}"]
    if client == "antigravity" and scope == "user":
        legacy = Path.home() / ".gemini/antigravity/mcp_config.json"
        if legacy.exists():
            merge_json(legacy, "mcpServers", entry, dry_run)
            msgs.append(f"{'would update' if dry_run else 'updated'} legacy {legacy}")
    if dry_run:
        msgs.append("\n" + text)
    return "\n".join(msgs)


def uninstall(client: str, scope: str = "user", project_dir: Path | None = None) -> str:
    project_dir = (project_dir or Path.cwd()).resolve()
    if client == "claude-code" and scope == "user":
        return _run(["claude", "mcp", "remove", "--scope", "user", SERVER_NAME], False)
    if client == "vscode" and scope == "user":
        return "Remove it from VS Code: Command Palette -> 'MCP: List Servers' -> graphrag-reader -> Uninstall."
    path = config_path(client, scope, project_dir)
    if not path or not path.exists():
        return f"nothing to remove ({path} not found)"
    _backup(path)
    if client == "codex":
        text = merge_toml(path, "", dry_run=True).rstrip() + "\n"
        path.write_text(text)
    else:
        data = _load_json(path)
        data.get("servers" if client == "vscode" else "mcpServers", {}).pop(SERVER_NAME, None)
        path.write_text(json.dumps(data, indent=2) + "\n")
    return f"removed {SERVER_NAME} from {path}"


def snippets(launcher: str = "auto", data_dir: str | None = None) -> str:
    """Copy-paste configs for every client (used by `graphrag install --print-all`)."""
    l = resolve_launch(launcher, data_dir)
    std = json.dumps({"mcpServers": {SERVER_NAME: _entry_mcpservers(l)}}, indent=2)
    vsc = json.dumps({"servers": {SERVER_NAME: _entry_vscode(l)}}, indent=2)
    cmd = " ".join(_shell_quote(x) for x in [l.command, *l.args])
    return f"""\
# Claude Code
claude mcp add {SERVER_NAME} -- {cmd}

# Codex (CLI, IDE extension and desktop app share ~/.codex/config.toml)
codex mcp add {SERVER_NAME} -- {cmd}
# or append to ~/.codex/config.toml:
{codex_toml_block(l)}
# VS Code (GitHub Copilot agent mode)
code --add-mcp {_shell_quote(json.dumps({"name": SERVER_NAME, **_entry_vscode(l)}))}
# or .vscode/mcp.json:
{vsc}

# Cursor (~/.cursor/mcp.json), Antigravity (~/.gemini/config/mcp_config.json),
# Windsurf (~/.codeium/windsurf/mcp_config.json), Claude Desktop (claude_desktop_config.json):
{std}
"""
