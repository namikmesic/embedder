from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None  # type: ignore[assignment]


def _parse_bool(value: str) -> bool:
    return value.lower() in ("1", "true", "yes")


class EmbedderProvider(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"


class EmbedderConfig(BaseModel):
    provider: EmbedderProvider = EmbedderProvider.SENTENCE_TRANSFORMERS
    model: str = "BAAI/bge-base-en-v1.5"
    openai_api_key: Optional[str] = None
    batch_size: int = Field(default=64, ge=1, le=2048)


class StoreConfig(BaseModel):
    pg_dsn: str = "postgresql://postgres:postgres@localhost:5432/vectordb"
    table_name: str = "chunks"
    ef_search: int = Field(default=40, ge=1, le=1000)
    iterative_scan: bool = False
    hybrid_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    pool_min_size: int = Field(default=1, ge=1, le=100)
    pool_max_size: int = Field(default=10, ge=1, le=100)


class PipelineConfig(BaseModel):
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)


CONFIG_FILE_ENV = "VECTORDB_CONFIG"
DEFAULT_CONFIG_PATH = Path("./pipeline.yaml")

_ENV_MAP: list[tuple[str, str, type]] = [
    ("VECTORDB_EMBEDDER_PROVIDER", "embedder.provider", EmbedderProvider),
    ("VECTORDB_EMBEDDER_MODEL", "embedder.model", str),
    ("OPENAI_API_KEY", "embedder.openai_api_key", str),
    ("DATABASE_URL", "store.pg_dsn", str),
    ("VECTORDB_PG_DSN", "store.pg_dsn", str),
    ("VECTORDB_TABLE_NAME", "store.table_name", str),
    ("VECTORDB_EF_SEARCH", "store.ef_search", int),
    ("VECTORDB_ITERATIVE_SCAN", "store.iterative_scan", _parse_bool),
    ("VECTORDB_HYBRID_ALPHA", "store.hybrid_alpha", float),
    ("VECTORDB_POOL_MIN_SIZE", "store.pool_min_size", int),
    ("VECTORDB_POOL_MAX_SIZE", "store.pool_max_size", int),
]


def _apply_env_overrides(config: PipelineConfig) -> None:
    for env_var, dotted_path, converter in _ENV_MAP:
        value = os.environ.get(env_var)
        if not value:
            continue
        parts = dotted_path.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], converter(value))


def load_config(config_path: Optional[Path] = None) -> PipelineConfig:
    """Priority: env vars > YAML file > defaults."""
    config = PipelineConfig()

    path = config_path or (Path(os.environ[CONFIG_FILE_ENV]) if CONFIG_FILE_ENV in os.environ else DEFAULT_CONFIG_PATH)
    if path.exists() and yaml is not None:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        config = PipelineConfig(**raw)
    elif path.exists() and path.suffix == ".json":
        with open(path) as f:
            raw = json.load(f)
        config = PipelineConfig(**raw)

    _apply_env_overrides(config)

    return config
