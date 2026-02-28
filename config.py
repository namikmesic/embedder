"""Configuration loading for the Git RAG MCP server."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None  # type: ignore[assignment]

from models import EmbedderConfig, EmbedderProvider, PipelineConfig, StoreBackend, StoreConfig


CONFIG_FILE_ENV = "GIT_RAG_CONFIG"
DEFAULT_CONFIG_PATH = Path("./pipeline.yaml")
DATA_DIR = Path(os.environ.get("GIT_RAG_DATA_DIR", "./data"))


def load_config(config_path: Optional[Path] = None) -> PipelineConfig:
    """Load pipeline configuration from YAML file and/or environment variables.

    Priority: env vars > YAML file > defaults.
    """
    config = PipelineConfig()

    # Try loading from YAML
    path = config_path or (Path(os.environ[CONFIG_FILE_ENV]) if CONFIG_FILE_ENV in os.environ else DEFAULT_CONFIG_PATH)
    if path.exists() and yaml is not None:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        config = PipelineConfig(**raw)
    elif path.exists() and path.suffix == ".json":
        with open(path) as f:
            raw = json.load(f)
        config = PipelineConfig(**raw)

    # Override with env vars
    if os.environ.get("GIT_RAG_EMBEDDER_PROVIDER"):
        config.embedder.provider = EmbedderProvider(os.environ["GIT_RAG_EMBEDDER_PROVIDER"])
    if os.environ.get("GIT_RAG_EMBEDDER_MODEL"):
        config.embedder.model_name = os.environ["GIT_RAG_EMBEDDER_MODEL"]
    if os.environ.get("OPENAI_API_KEY"):
        config.embedder.openai_api_key = os.environ["OPENAI_API_KEY"]
    if os.environ.get("GIT_RAG_STORE_BACKEND"):
        config.store.backend = StoreBackend(os.environ["GIT_RAG_STORE_BACKEND"])
    if os.environ.get("GIT_RAG_STORE_PERSIST_DIR"):
        config.store.persist_dir = os.environ["GIT_RAG_STORE_PERSIST_DIR"]
    if os.environ.get("GIT_RAG_REPOS_DIR"):
        config.repos_dir = os.environ["GIT_RAG_REPOS_DIR"]

    # Ensure data directories exist
    Path(config.repos_dir).mkdir(parents=True, exist_ok=True)
    Path(config.store.persist_dir).mkdir(parents=True, exist_ok=True)

    return config
