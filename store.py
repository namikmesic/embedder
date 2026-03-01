from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector
from pydantic import BaseModel, Field

from config import StoreConfig
from embedding import Embedder

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class VectorStore(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def add(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadatas: list[dict[str, Any]],
        texts: list[str],
        created_ats: Optional[list[datetime]] = None,
    ) -> None: ...

    @abstractmethod
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
        query_text: str = "",
    ) -> list[SearchResult]: ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> int: ...

    @abstractmethod
    async def delete_by_metadata(self, filter_key: str, filter_value: str) -> int: ...

    @abstractmethod
    async def count(self) -> int: ...

    @abstractmethod
    async def close(self) -> None: ...


class PgVectorStore(VectorStore):

    def __init__(self, config: StoreConfig, dimension: int, model_version: str = "") -> None:
        self._config = config
        self._dimension = dimension
        self._model_version = model_version
        self._pool: asyncpg.Pool | None = None

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    async def initialize(self) -> None:
        # Run schema DDL before creating the pool, because register_vector
        # (the pool init callback) needs the vector extension to already exist.
        conn = await asyncpg.connect(dsn=self._config.pg_dsn)
        table = self._config.table_name
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id         TEXT PRIMARY KEY,
                    embedding  vector({self._dimension}),
                    text       TEXT NOT NULL,
                    metadata   JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_embedding_hnsw
                    ON {table} USING hnsw (embedding vector_cosine_ops)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_metadata_gin
                    ON {table} USING gin (metadata)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_created_at
                    ON {table} USING btree (created_at)
            """)
            await conn.execute(f"""
                ALTER TABLE {table}
                ADD COLUMN IF NOT EXISTS model_version TEXT
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_model_version
                    ON {table} USING btree (model_version)
            """)
            await conn.execute(f"""
                ALTER TABLE {table}
                ADD COLUMN IF NOT EXISTS tsv tsvector
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_tsv_gin
                    ON {table} USING gin (tsv)
            """)
        finally:
            await conn.close()

        self._pool = await asyncpg.create_pool(
            dsn=self._config.pg_dsn,
            min_size=self._config.pool_min_size,
            max_size=self._config.pool_max_size,
            init=self._init_connection,
        )

        logger.info(
            "PgVectorStore initialized: table=%s, dim=%d",
            table,
            self._dimension,
        )

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Store not initialized")
        return self._pool

    async def add(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadatas: list[dict[str, Any]],
        texts: list[str],
        created_ats: Optional[list[datetime]] = None,
    ) -> None:
        if len(ids) == 0:
            return

        pool = self._require_pool()
        table = self._config.table_name

        if created_ats:
            async with pool.acquire() as conn:
                await conn.executemany(
                    f"""
                    INSERT INTO {table} (id, embedding, text, metadata, created_at, model_version, tsv)
                    VALUES ($1, $2, $3, $4::jsonb, $5, $6, to_tsvector('english', $3))
                    ON CONFLICT (id) DO UPDATE
                        SET embedding     = EXCLUDED.embedding,
                            text          = EXCLUDED.text,
                            metadata      = EXCLUDED.metadata,
                            created_at    = EXCLUDED.created_at,
                            model_version = EXCLUDED.model_version,
                            tsv           = EXCLUDED.tsv
                    """,
                    [
                        (ids[i], vectors[i].tolist(), texts[i], json.dumps(metadatas[i]), created_ats[i], self._model_version)
                        for i in range(len(ids))
                    ],
                )
        else:
            async with pool.acquire() as conn:
                await conn.executemany(
                    f"""
                    INSERT INTO {table} (id, embedding, text, metadata, model_version, tsv)
                    VALUES ($1, $2, $3, $4::jsonb, $5, to_tsvector('english', $3))
                    ON CONFLICT (id) DO UPDATE
                        SET embedding     = EXCLUDED.embedding,
                            text          = EXCLUDED.text,
                            metadata      = EXCLUDED.metadata,
                            model_version = EXCLUDED.model_version,
                            tsv           = EXCLUDED.tsv
                    """,
                    [
                        (ids[i], vectors[i].tolist(), texts[i], json.dumps(metadatas[i]), self._model_version)
                        for i in range(len(ids))
                    ],
                )

        logger.info("Added %d vectors to table '%s'", len(ids), table)

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
        query_text: str = "",
    ) -> list[SearchResult]:
        pool = self._require_pool()
        table = self._config.table_name
        alpha = self._config.hybrid_alpha

        conditions: list[str] = []
        args: list[Any] = [query_vector.tolist()]
        param_idx = 2  # $1 is the vector

        if self._model_version:
            conditions.append(f"model_version = ${param_idx}")
            args.append(self._model_version)
            param_idx += 1

        if filters:
            conditions.append(f"metadata @> ${param_idx}::jsonb")
            args.append(json.dumps(filters))
            param_idx += 1

        if after:
            conditions.append(f"created_at >= ${param_idx}")
            args.append(after)
            param_idx += 1

        if before:
            conditions.append(f"created_at <= ${param_idx}")
            args.append(before)
            param_idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        use_hybrid = alpha < 1.0 and query_text

        if use_hybrid:
            # Over-fetch vector candidates, then re-rank with blended score.
            text_param_idx = param_idx
            args.append(query_text)
            param_idx += 1
            limit_param_idx = param_idx
            args.append(top_k)
            candidate_limit = top_k * 5

            query = f"""
                WITH candidates AS (
                    SELECT id, text, metadata, created_at, tsv,
                           1 - (embedding <=> $1::vector) AS vec_score
                    FROM {table}
                    {where}
                    ORDER BY embedding <=> $1::vector
                    LIMIT {candidate_limit}
                )
                SELECT id, text, metadata, created_at,
                       {alpha} * vec_score
                       + {1.0 - alpha} * COALESCE(ts_rank(tsv, plainto_tsquery('english', ${text_param_idx})), 0)
                       AS score
                FROM candidates
                ORDER BY score DESC
                LIMIT ${limit_param_idx}
            """
        else:
            query = f"""
                SELECT id, text, metadata, created_at,
                       1 - (embedding <=> $1::vector) AS score
                FROM {table}
                {where}
                ORDER BY embedding <=> $1::vector
                LIMIT ${param_idx}
            """
            args.append(top_k)

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL hnsw.ef_search = {self._config.ef_search}")
                if self._config.iterative_scan:
                    await conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
                rows = await conn.fetch(query, *args)

        return [
            SearchResult(
                chunk_id=row["id"],
                text=row["text"],
                score=float(row["score"]),
                metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"]),
                created_at=row["created_at"].isoformat() if row["created_at"] else None,
            )
            for row in rows
        ]

    async def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0

        pool = self._require_pool()
        table = self._config.table_name

        async with pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {table} WHERE id = ANY($1)", ids
            )

        deleted = int(result.split()[-1])
        logger.info("Deleted %d vectors by ID from table '%s'", deleted, table)
        return deleted

    async def delete_by_metadata(self, filter_key: str, filter_value: str) -> int:
        pool = self._require_pool()
        table = self._config.table_name
        filter_json = json.dumps({filter_key: filter_value})

        async with pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {table} WHERE metadata @> $1::jsonb",
                filter_json,
            )

        deleted = int(result.split()[-1])
        logger.info(
            "Deleted %d vectors where %s=%s from table '%s'",
            deleted,
            filter_key,
            filter_value,
            table,
        )
        return deleted

    async def count(self) -> int:
        pool = self._require_pool()
        table = self._config.table_name

        async with pool.acquire() as conn:
            return await conn.fetchval(f"SELECT count(*) FROM {table}")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


async def create_store(config: StoreConfig, embedder: Embedder) -> VectorStore:
    store = PgVectorStore(config=config, dimension=embedder.dimension, model_version=embedder.model_id)
    await store.initialize()
    return store
