from __future__ import annotations

import logging
import re
from datetime import datetime
from uuid import UUID

import asyncpg
from pgvector.asyncpg import register_vector
from pydantic import BaseModel

import uuid_utils

from config import (
    DistanceMetric,
    EmbedderConfig,
    EmbedderProvider,
    KnowledgebaseDefaults,
    ObjectStoreConfig,
    StoreConfig,
    VectorPrecision,
)
from embedding import Embedder, EmbedderCache
from object_store import ObjectStore
from store import PgVectorStore

logger = logging.getLogger(__name__)

# Valid KB name: lowercase alphanumeric, hyphens, underscores, 1-63 chars.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")

# PostgreSQL built-in text search configurations.
_VALID_TSV_LANGUAGES = frozenset({
    "simple", "arabic", "armenian", "basque", "catalan", "danish", "dutch",
    "english", "finnish", "french", "german", "greek", "hindi", "hungarian",
    "indonesian", "irish", "italian", "lithuanian", "nepali", "norwegian",
    "portuguese", "romanian", "russian", "serbian", "spanish", "swedish",
    "tamil", "turkish", "yiddish",
})

# --------------------------------------------------------------------------
# Operator lookups
# --------------------------------------------------------------------------

_OPS_CLASS: dict[tuple[VectorPrecision, DistanceMetric], str] = {
    (VectorPrecision.FLOAT16, DistanceMetric.COSINE): "halfvec_cosine_ops",
    (VectorPrecision.FLOAT16, DistanceMetric.L2): "halfvec_l2_ops",
    (VectorPrecision.FLOAT16, DistanceMetric.INNER_PRODUCT): "halfvec_ip_ops",
    (VectorPrecision.FLOAT32, DistanceMetric.COSINE): "vector_cosine_ops",
    (VectorPrecision.FLOAT32, DistanceMetric.L2): "vector_l2_ops",
    (VectorPrecision.FLOAT32, DistanceMetric.INNER_PRODUCT): "vector_ip_ops",
}

_DISTANCE_OP: dict[DistanceMetric, str] = {
    DistanceMetric.COSINE: "<=>",
    DistanceMetric.L2: "<->",
    DistanceMetric.INNER_PRODUCT: "<#>",
}


# --------------------------------------------------------------------------
# Knowledgebase record (mirrors the registry table row)
# --------------------------------------------------------------------------

class KnowledgebaseRecord(BaseModel):
    id: UUID
    name: str
    table_name: str
    embedder_provider: str
    embedder_model: str
    dimension: int
    precision: str = "halfvec"
    distance_metric: str = "cosine"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    tsv_language: str = "english"
    ef_search: int = 40
    iterative_scan: bool = False
    hybrid_alpha: float = 1.0
    candidate_multiplier: int = 5
    created_at: datetime | None = None


# --------------------------------------------------------------------------
# DDL builder
# --------------------------------------------------------------------------

def build_table_ddl(
    table: str,
    dimension: int,
    precision: VectorPrecision,
    distance_metric: DistanceMetric,
    hnsw_m: int,
    hnsw_ef_construction: int,
) -> str:
    """Generate complete DDL for a knowledgebase table.

    Includes: CREATE TABLE, all indexes (HNSW, GIN, B-tree, unique),
    RLS policy, and per-table autovacuum tuning.
    """
    vec_type = precision.value  # "halfvec" or "vector"
    ops_class = _OPS_CLASS[(precision, distance_metric)]

    return f"""
-- Table
CREATE TABLE IF NOT EXISTS {table} (
    id            UUID         NOT NULL PRIMARY KEY,
    tenant_id     UUID         NOT NULL,
    source_id     TEXT         NOT NULL,
    embedding     {vec_type}({dimension}) NOT NULL,
    content_hash  BYTEA        NOT NULL,
    tsv           tsvector     NOT NULL,
    latest        BOOLEAN      NOT NULL DEFAULT true
);

-- HNSW vector index (set maintenance_work_mem for build performance)
SET LOCAL maintenance_work_mem = '512MB';
CREATE INDEX IF NOT EXISTS idx_{table}_hnsw
    ON {table} USING hnsw (embedding {ops_class})
    WITH (m = {hnsw_m}, ef_construction = {hnsw_ef_construction})
    WHERE latest;

-- Supporting indexes
CREATE INDEX IF NOT EXISTS idx_{table}_tenant
    ON {table} (tenant_id) WHERE latest;

CREATE INDEX IF NOT EXISTS idx_{table}_tsv
    ON {table} USING gin (tsv) WHERE latest;

CREATE INDEX IF NOT EXISTS idx_{table}_tenant_source
    ON {table} (tenant_id, source_id) WHERE latest;

CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_tenant_source_uniq
    ON {table} (tenant_id, source_id) WHERE latest;

CREATE INDEX IF NOT EXISTS idx_{table}_hash
    ON {table} (content_hash);

-- Row-Level Security
ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;
ALTER TABLE {table} FORCE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE tablename = '{table}' AND policyname = 'tenant_isolation'
    ) THEN
        CREATE POLICY tenant_isolation ON {table}
            USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
            WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);
    END IF;
END
$$;

-- Autovacuum tuning for high-write vector tables
ALTER TABLE {table} SET (autovacuum_vacuum_scale_factor = 0.01);
"""


# --------------------------------------------------------------------------
# KnowledgebaseManager
# --------------------------------------------------------------------------

class KnowledgebaseManager:
    """Central manager for knowledgebases.

    Holds a shared asyncpg pool, ObjectStore, and EmbedderCache.
    Builds per-KB PgVectorStore instances on demand and caches them.
    """

    def __init__(
        self,
        store_config: StoreConfig,
        object_store_config: ObjectStoreConfig,
        embedder_config: EmbedderConfig,
        kb_defaults: KnowledgebaseDefaults | None = None,
    ) -> None:
        self._store_config = store_config
        self._openai_api_key = embedder_config.openai_api_key
        self._kb_defaults = kb_defaults or KnowledgebaseDefaults()
        self._pool: asyncpg.Pool | None = None
        self.object_store = ObjectStore(object_store_config)
        self._embedder_cache = EmbedderCache()
        # Cache: name -> (record, store, embedder)
        self._cache: dict[str, tuple[KnowledgebaseRecord, PgVectorStore, Embedder]] = {}

    async def initialize(self) -> None:
        """Create the shared pool and initialize the object store."""
        self._pool = await asyncpg.create_pool(
            dsn=self._store_config.pg_dsn,
            min_size=self._store_config.pool_min_size,
            max_size=self._store_config.pool_max_size,
            init=_init_pg_connection,
        )

        # Verify the knowledgebases registry table exists
        async with self._pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                "WHERE table_name = 'knowledgebases')"
            )
            if not exists:
                raise RuntimeError(
                    "Table 'knowledgebases' does not exist. Run init.sql first "
                    "(docker compose will do this automatically)."
                )

        await self.object_store.initialize()
        logger.info("KnowledgebaseManager initialized")

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("KnowledgebaseManager not initialized")
        return self._pool

    async def create_kb(
        self,
        name: str,
        embedder_provider: EmbedderProvider | None = None,
        embedder_model: str | None = None,
        precision: VectorPrecision | None = None,
        distance_metric: DistanceMetric | None = None,
        hnsw_m: int | None = None,
        hnsw_ef_construction: int | None = None,
        tsv_language: str | None = None,
        ef_search: int | None = None,
        iterative_scan: bool | None = None,
        hybrid_alpha: float | None = None,
        candidate_multiplier: int | None = None,
        openai_api_key: str | None = None,
    ) -> KnowledgebaseRecord:
        """Create a new knowledgebase with its own PG table."""
        if not _NAME_RE.match(name):
            raise ValueError(
                f"Invalid KB name '{name}'. Must be 1-63 chars, lowercase alphanumeric, "
                "hyphens, underscores, starting with alphanumeric."
            )

        pool = self._require_pool()
        defaults = self._kb_defaults

        # Resolve settings: explicit arg > kb_defaults
        provider = embedder_provider or defaults.embedder_provider
        model = embedder_model or defaults.embedder_model
        prec = precision or defaults.precision
        dist = distance_metric or defaults.distance_metric
        m = hnsw_m if hnsw_m is not None else defaults.hnsw_m
        ef_con = hnsw_ef_construction if hnsw_ef_construction is not None else defaults.hnsw_ef_construction
        lang = tsv_language or defaults.tsv_language
        efs = ef_search if ef_search is not None else defaults.ef_search
        iscan = iterative_scan if iterative_scan is not None else defaults.iterative_scan
        ha = hybrid_alpha if hybrid_alpha is not None else defaults.hybrid_alpha
        cm = candidate_multiplier if candidate_multiplier is not None else defaults.candidate_multiplier

        # Validate resolved values (explicit overrides bypass KnowledgebaseDefaults)
        if not (2 <= m <= 100):
            raise ValueError(f"hnsw_m must be 2-100, got {m}")
        if not (8 <= ef_con <= 1000):
            raise ValueError(f"hnsw_ef_construction must be 8-1000, got {ef_con}")
        if not (1 <= efs <= 1000):
            raise ValueError(f"ef_search must be 1-1000, got {efs}")
        if not (0.0 <= ha <= 1.0):
            raise ValueError(f"hybrid_alpha must be 0.0-1.0, got {ha}")
        if not (1 <= cm <= 100):
            raise ValueError(f"candidate_multiplier must be 1-100, got {cm}")
        if lang not in _VALID_TSV_LANGUAGES:
            raise ValueError(
                f"Invalid tsv_language '{lang}'. Must be one of: "
                + ", ".join(sorted(_VALID_TSV_LANGUAGES))
            )

        # Instantiate embedder to discover dimension
        emb_config = EmbedderConfig(
            provider=provider,
            model=model,
            openai_api_key=openai_api_key or self._openai_api_key,
        )
        embedder = self._embedder_cache.get_or_create(emb_config)
        dimension = embedder.dimension

        # Generate table name and KB ID
        kb_id = UUID(int=uuid_utils.uuid7().int)
        table_name = f"kb_{kb_id.hex}"

        # Build DDL and execute within a transaction
        ddl = build_table_ddl(
            table=table_name,
            dimension=dimension,
            precision=prec,
            distance_metric=dist,
            hnsw_m=m,
            hnsw_ef_construction=ef_con,
        )

        async with pool.acquire() as conn:
            async with conn.transaction():
                # Insert registry row first (catches duplicate name via UNIQUE constraint)
                try:
                    await conn.execute(
                        """INSERT INTO knowledgebases
                           (id, name, table_name, embedder_provider, embedder_model,
                            dimension, precision, distance_metric, hnsw_m,
                            hnsw_ef_construction, tsv_language, ef_search,
                            iterative_scan, hybrid_alpha, candidate_multiplier)
                           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)""",
                        kb_id, name, table_name,
                        provider.value, model, dimension,
                        prec.value, dist.value, m, ef_con, lang,
                        efs, iscan, ha, cm,
                    )
                except asyncpg.UniqueViolationError:
                    raise ValueError(f"Knowledgebase '{name}' already exists")

                # Create the KB table + indexes
                await conn.execute(ddl)

        record = KnowledgebaseRecord(
            id=kb_id,
            name=name,
            table_name=table_name,
            embedder_provider=provider.value,
            embedder_model=model,
            dimension=dimension,
            precision=prec.value,
            distance_metric=dist.value,
            hnsw_m=m,
            hnsw_ef_construction=ef_con,
            tsv_language=lang,
            ef_search=efs,
            iterative_scan=iscan,
            hybrid_alpha=ha,
            candidate_multiplier=cm,
        )

        logger.info("Created knowledgebase '%s' -> table '%s' (dim=%d)", name, table_name, dimension)
        return record

    async def get_kb(self, name: str) -> tuple[KnowledgebaseRecord, PgVectorStore, Embedder]:
        """Get or build the (record, store, embedder) triple for a KB."""
        if name in self._cache:
            return self._cache[name]

        pool = self._require_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM knowledgebases WHERE name = $1", name
            )

        if row is None:
            raise ValueError(f"Knowledgebase '{name}' not found")

        record = KnowledgebaseRecord(**dict(row))

        # Build PgVectorStore with per-KB settings
        dist = DistanceMetric(record.distance_metric)
        store = PgVectorStore(
            pool=pool,
            table=record.table_name,
            vec_type=record.precision,
            distance_op=_DISTANCE_OP[dist],
            tsv_language=record.tsv_language,
            ef_search=record.ef_search,
            iterative_scan=record.iterative_scan,
            hybrid_alpha=record.hybrid_alpha,
            candidate_multiplier=record.candidate_multiplier,
        )

        # Get or create embedder from cache
        emb_config = EmbedderConfig(
            provider=EmbedderProvider(record.embedder_provider),
            model=record.embedder_model,
            openai_api_key=self._openai_api_key,
        )
        embedder = self._embedder_cache.get_or_create(emb_config)

        self._cache[name] = (record, store, embedder)
        return record, store, embedder

    async def list_kbs(self) -> list[KnowledgebaseRecord]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM knowledgebases ORDER BY created_at"
            )
        return [KnowledgebaseRecord(**dict(row)) for row in rows]

    async def close(self) -> None:
        await self._embedder_cache.close_all()
        await self.object_store.close()
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._cache.clear()


async def _init_pg_connection(conn: asyncpg.Connection) -> None:
    """Pool init callback — register pgvector types on each new connection."""
    await register_vector(conn)
