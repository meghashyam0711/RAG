"""Runtime configuration, read once from environment variables.

Every external service is optional. With no env vars set the engine runs in
"lite" mode: embedded Qdrant on disk, a NetworkX graph persisted as JSON,
hashed embeddings, a lexical reranker and heuristic graders. Each component
upgrades automatically when its dependency / credential is present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))  # type: ignore[arg-type]


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))  # type: ignore[arg-type]


@dataclass
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(
            _env("GRAPHRAG_DATA_DIR", str(Path.home() / ".graphrag-mcp"))  # type: ignore[arg-type]
        ).expanduser()
    )

    # --- Vector store -----------------------------------------------------
    qdrant_url: str | None = field(default_factory=lambda: _env("QDRANT_URL"))
    qdrant_api_key: str | None = field(default_factory=lambda: _env("QDRANT_API_KEY"))
    collection: str = field(default_factory=lambda: _env("GRAPHRAG_COLLECTION", "graphrag_chunks"))  # type: ignore[arg-type]

    # --- Graph store ------------------------------------------------------
    neo4j_uri: str | None = field(default_factory=lambda: _env("NEO4J_URI"))
    neo4j_user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))  # type: ignore[arg-type]
    neo4j_password: str | None = field(default_factory=lambda: _env("NEO4J_PASSWORD"))

    # --- Models -----------------------------------------------------------
    dense_model: str = field(default_factory=lambda: _env("GRAPHRAG_DENSE_MODEL", "BAAI/bge-small-en-v1.5"))  # type: ignore[arg-type]
    reranker_model: str = field(default_factory=lambda: _env("GRAPHRAG_RERANKER", "BAAI/bge-reranker-v2-m3"))  # type: ignore[arg-type]
    # "auto" picks the best available; "hash" forces the offline embedder.
    embedder: str = field(default_factory=lambda: _env("GRAPHRAG_EMBEDDER", "auto"))  # type: ignore[arg-type]

    # LLM: provider in {"anthropic", "openai", "none"}; "auto" picks by available key.
    llm_provider: str = field(default_factory=lambda: _env("GRAPHRAG_LLM_PROVIDER", "auto"))  # type: ignore[arg-type]
    llm_model: str | None = field(default_factory=lambda: _env("GRAPHRAG_LLM_MODEL"))
    openai_base_url: str | None = field(default_factory=lambda: _env("OPENAI_BASE_URL"))

    # --- Chunking ---------------------------------------------------------
    chunk_tokens: int = field(default_factory=lambda: _env_int("GRAPHRAG_CHUNK_TOKENS", 800))
    chunk_overlap: float = field(default_factory=lambda: _env_float("GRAPHRAG_CHUNK_OVERLAP", 0.10))

    # --- Retrieval / agent ------------------------------------------------
    vector_k: int = field(default_factory=lambda: _env_int("GRAPHRAG_VECTOR_K", 15))
    fused_k: int = field(default_factory=lambda: _env_int("GRAPHRAG_FUSED_K", 10))
    final_k: int = field(default_factory=lambda: _env_int("GRAPHRAG_FINAL_K", 5))
    graph_hops: int = field(default_factory=lambda: _env_int("GRAPHRAG_GRAPH_HOPS", 1))
    max_graph_hops: int = field(default_factory=lambda: _env_int("GRAPHRAG_MAX_GRAPH_HOPS", 2))
    max_retries: int = field(default_factory=lambda: _env_int("GRAPHRAG_MAX_RETRIES", 2))
    max_regenerations: int = field(default_factory=lambda: _env_int("GRAPHRAG_MAX_REGENERATIONS", 1))
    entity_match_threshold: float = field(default_factory=lambda: _env_float("GRAPHRAG_ENTITY_THRESHOLD", 92.0))

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def graph_path(self) -> Path:
        return self.data_dir / "graph.json"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "documents.json"

    @property
    def parsed_dir(self) -> Path:
        p = self.data_dir / "parsed_markdown"
        p.mkdir(parents=True, exist_ok=True)
        return p
