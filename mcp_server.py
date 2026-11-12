"""Entry point: `uv run --directory /path/to/self-corrective-graphrag mcp_server.py`

Thin wrapper so the server runs straight from a git clone without installing
the package. Equivalent to the `graphrag-mcp` console script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from graphrag.server import main  # noqa: E402

if __name__ == "__main__":
    main()
