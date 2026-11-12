#!/bin/sh
# GraphRAG one-line installer for macOS / Linux.
#   curl -LsSf https://raw.githubusercontent.com/meghashyam0711/RAG/main/install.sh | sh
# Options (env vars):
#   GRAPHRAG_HOME=install folder (default ~/RAG)   GRAPHRAG_CLIENTS=all|none|codex,cursor (default all)
#   GRAPHRAG_REPO=git URL or local path (default https://github.com/meghashyam0711/RAG)
set -eu

REPO="${GRAPHRAG_REPO:-https://github.com/meghashyam0711/RAG}"
DIR="${GRAPHRAG_HOME:-$HOME/RAG}"
CLIENTS="${GRAPHRAG_CLIENTS:-all}"
step() { printf '\033[36m==> %s\033[0m\n' "$1"; }
has() { command -v "$1" >/dev/null 2>&1; }
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 1. uv (also provides Python 3.11+)
if ! has uv; then
  step "Installing uv"
  if has curl; then curl -LsSf https://astral.sh/uv/install.sh | sh; else wget -qO- https://astral.sh/uv/install.sh | sh; fi
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  has uv || { echo "uv installed but not on PATH; open a new terminal and re-run." >&2; exit 1; }
fi

# 2. Code: git clone, or tarball when git is missing
if [ -f "$DIR/pyproject.toml" ]; then
  step "Found existing install in $DIR"
  if has git && [ -d "$DIR/.git" ]; then git -C "$DIR" pull --ff-only --quiet || true; fi
else
  if [ -d "$DIR" ] && [ -n "$(ls -A "$DIR" 2>/dev/null)" ]; then
    echo "$DIR exists but is not this project. Remove it or set GRAPHRAG_HOME." >&2; exit 1
  fi
  if has git; then
    step "Cloning $REPO into $DIR"
    git clone --depth 1 --quiet "$REPO" "$DIR"
  else
    step "git not found; downloading archive"
    tmp=$(mktemp -d)
    url="$(printf '%s' "$REPO" | sed 's#/*$##; s#\.git$##')/archive/refs/heads/main.tar.gz"
    if has curl; then curl -LsSf "$url" | tar xz -C "$tmp"; else wget -qO- "$url" | tar xz -C "$tmp"; fi
    rm -rf "$DIR"; mv "$tmp"/* "$DIR"; rm -rf "$tmp"
  fi
fi

# 3. Dependencies + self-test
cd "$DIR"
step "Installing dependencies (uv sync)"
uv sync --quiet
step "Self-test"
uv run graphrag doctor

# 4. Register with agents / IDEs
if [ "$CLIENTS" != "none" ]; then
  for c in $(printf '%s' "$CLIENTS" | tr ',' ' '); do
    step "Registering MCP server: $c"
    uv run graphrag install "$c" || true
  done
fi

printf '\n\033[32mDone. Installed in %s\033[0m\n' "$DIR"
echo "Restart your agent/IDE to load 'graphrag-reader'. Terminal use:"
echo "  cd \"$DIR\" && uv run graphrag ingest /path/to/file.pdf --domain technical"
echo "  uv run graphrag ask \"your question\""
