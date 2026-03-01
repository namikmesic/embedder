from __future__ import annotations

import os
from enum import Enum

from pydantic import BaseModel, Field


def _parse_bool(value: str) -> bool:
    return value.lower() in ("1", "true", "yes")


class EmbedderProvider(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"


class VectorPrecision(str, Enum):
    FLOAT32 = "vector"
    FLOAT16 = "halfvec"


class EmbedderConfig(BaseModel):
    provider: EmbedderProvider = EmbedderProvider.SENTENCE_TRANSFORMERS
    model: str = "BAAI/bge-base-en-v1.5"
    openai_api_key: str | None = None
    batch_size: int = Field(default=64, ge=1, le=2048)


class KnowledgebaseDefaults(BaseModel):
    """Defaults for new KBs. Overridable per-KB at creation time."""
    embedder_provider: EmbedderProvider = EmbedderProvider.SENTENCE_TRANSFORMERS
    embedder_model: str = "BAAI/bge-base-en-v1.5"
    precision: VectorPrecision = VectorPrecision.FLOAT16
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    hnsw_m: int = Field(default=16, ge=2, le=100)
    hnsw_ef_construction: int = Field(default=64, ge=8, le=1000)
    tsv_language: str = "english"
    # Mutable query-time settings
    ef_search: int = Field(default=40, ge=1, le=1000)
    iterative_scan: bool = False
    hybrid_alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    candidate_multiplier: int = Field(default=5, ge=1, le=100)


class StoreConfig(BaseModel):
    pg_dsn: str = "postgresql://postgres:postgres@localhost:5432/vectordb"
    pool_min_size: int = Field(default=1, ge=1, le=100)
    pool_max_size: int = Field(default=10, ge=1, le=100)


class ObjectStoreConfig(BaseModel):
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "vectordb-chunks"
    secure: bool = False


class PipelineConfig(BaseModel):
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    object_store: ObjectStoreConfig = Field(default_factory=ObjectStoreConfig)
    kb_defaults: KnowledgebaseDefaults = Field(default_factory=KnowledgebaseDefaults)


_ENV_MAP: list[tuple[str, str, type]] = [
    # Embedder
    ("VECTORDB_EMBEDDER_PROVIDER", "embedder.provider", EmbedderProvider),
    ("VECTORDB_EMBEDDER_MODEL", "embedder.model", str),
    ("OPENAI_API_KEY", "embedder.openai_api_key", str),
    # Store (PostgreSQL) — infrastructure only
    ("DATABASE_URL", "store.pg_dsn", str),
    ("VECTORDB_PG_DSN", "store.pg_dsn", str),
    ("VECTORDB_POOL_MIN_SIZE", "store.pool_min_size", int),
    ("VECTORDB_POOL_MAX_SIZE", "store.pool_max_size", int),
    # Knowledgebase defaults (query-time settings for new KBs)
    ("VECTORDB_EF_SEARCH", "kb_defaults.ef_search", int),
    ("VECTORDB_ITERATIVE_SCAN", "kb_defaults.iterative_scan", _parse_bool),
    ("VECTORDB_HYBRID_ALPHA", "kb_defaults.hybrid_alpha", float),
    ("VECTORDB_CANDIDATE_MULTIPLIER", "kb_defaults.candidate_multiplier", int),
    ("VECTORDB_HNSW_M", "kb_defaults.hnsw_m", int),
    ("VECTORDB_HNSW_EF_CONSTRUCTION", "kb_defaults.hnsw_ef_construction", int),
    ("VECTORDB_TSV_LANGUAGE", "kb_defaults.tsv_language", str),
    # Object Store (MinIO)
    ("VECTORDB_MINIO_ENDPOINT", "object_store.endpoint", str),
    ("VECTORDB_MINIO_ACCESS_KEY", "object_store.access_key", str),
    ("VECTORDB_MINIO_SECRET_KEY", "object_store.secret_key", str),
    ("VECTORDB_MINIO_BUCKET", "object_store.bucket", str),
    ("VECTORDB_MINIO_SECURE", "object_store.secure", _parse_bool),
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


def load_config() -> PipelineConfig:
    """Priority: env vars > defaults."""
    config = PipelineConfig()
    _apply_env_overrides(config)
    return config
