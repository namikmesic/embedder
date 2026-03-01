from __future__ import annotations

import logging
from datetime import datetime
from typing import NamedTuple
from uuid import UUID

import asyncpg
import numpy as np
from pydantic import BaseModel

import uuid_utils

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    chunk_id: UUID
    source_id: str
    score: float


class UpsertResult(NamedTuple):
    wrote: bool
    prior_id: UUID | None  # existing id if unchanged; old version id if updated


class PgVectorStore:

    def __init__(
        self,
        pool: asyncpg.Pool,
        table: str,
        vec_type: str = "halfvec",
        distance_op: str = "<=>",
        tsv_language: str = "english",
        ef_search: int = 40,
        iterative_scan: bool = False,
        hybrid_alpha: float = 1.0,
        candidate_multiplier: int = 5,
    ) -> None:
        self._pool = pool
        self._table = table
        self._vec_type = vec_type
        self._distance_op = distance_op
        self._tsv_language = tsv_language
        self._ef_search = ef_search
        self._iterative_scan = iterative_scan
        self._hybrid_alpha = hybrid_alpha
        self._candidate_multiplier = candidate_multiplier

    def _score_expr(self, vec_ref: str) -> str:
        """Build a SQL score expression. Cosine: 1 - distance; L2/IP: negated."""
        if self._distance_op == "<=>":
            return f"1 - (embedding {self._distance_op} {vec_ref})"
        return f"-(embedding {self._distance_op} {vec_ref})"

    @staticmethod
    async def _set_tenant(conn: asyncpg.Connection, tenant_id: UUID) -> None:
        await conn.execute("SELECT set_config('app.tenant_id', $1, true)", str(tenant_id))

    @staticmethod
    def _dt_to_uuid7(dt: datetime) -> UUID:
        """Convert a datetime to a UUIDv7 boundary for time-range filtering."""
        ms = int(dt.timestamp() * 1000)
        return UUID(int=uuid_utils.uuid7(timestamp=ms).int)

    async def check_hash(
        self, tenant_id: UUID, source_id: str, content_hash: bytes
    ) -> UUID | None:
        """Optimistic read-only check: return existing chunk id if hash matches, else None."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._set_tenant(conn, tenant_id)
                return await conn.fetchval(
                    f"SELECT id FROM {self._table} "
                    "WHERE tenant_id = $1 AND source_id = $2 AND latest = true "
                    "AND content_hash = $3",
                    tenant_id,
                    source_id,
                    content_hash,
                )

    async def upsert(
        self,
        chunk_id: UUID,
        tenant_id: UUID,
        source_id: str,
        embedding: np.ndarray,
        content_hash: bytes,
        text_for_tsv: str,
    ) -> UpsertResult:
        """Insert or update a chunk.

        Returns UpsertResult(wrote, prior_id):
        - (False, existing_id) if content_hash matches existing latest row (no-op)
        - (True, None) if this is a new source_id
        - (True, old_id) if updated (old row set to latest=false)
        """
        vec_list = embedding.tolist()

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._set_tenant(conn, tenant_id)

                existing = await conn.fetchrow(
                    f"SELECT id, content_hash FROM {self._table} "
                    "WHERE tenant_id = $1 AND source_id = $2 AND latest = true "
                    "FOR UPDATE",
                    tenant_id,
                    source_id,
                )

                if existing is not None:
                    if existing["content_hash"] == content_hash:
                        return UpsertResult(False, existing["id"])

                    old_id: UUID = existing["id"]
                    await conn.execute(
                        f"UPDATE {self._table} SET latest = false WHERE id = $1 AND tenant_id = $2",
                        old_id,
                        tenant_id,
                    )
                else:
                    old_id = None

                await conn.execute(
                    f"INSERT INTO {self._table} "
                    "(id, tenant_id, source_id, embedding, content_hash, tsv, latest) "
                    f"VALUES ($1, $2, $3, $4::{self._vec_type}, $5, "
                    f"to_tsvector('{self._tsv_language}', $6), true)",
                    chunk_id,
                    tenant_id,
                    source_id,
                    vec_list,
                    content_hash,
                    text_for_tsv,
                )

                return UpsertResult(True, old_id)

    async def search(
        self,
        query_vector: np.ndarray,
        tenant_id: UUID,
        top_k: int = 10,
        after: datetime | None = None,
        before: datetime | None = None,
        query_text: str = "",
    ) -> list[SearchResult]:
        alpha = self._hybrid_alpha

        conditions = ["latest = true", "tenant_id = $2"]
        args: list = [query_vector.tolist(), tenant_id]
        param_idx = 3

        if after:
            conditions.append(f"id >= ${param_idx}")
            args.append(self._dt_to_uuid7(after))
            param_idx += 1

        if before:
            conditions.append(f"id <= ${param_idx}")
            args.append(self._dt_to_uuid7(before))
            param_idx += 1

        where = f"WHERE {' AND '.join(conditions)}"

        vec_cast = f"$1::{self._vec_type}"
        score_expr = self._score_expr(vec_cast)
        order_expr = f"embedding {self._distance_op} {vec_cast}"

        use_hybrid = alpha < 1.0 and query_text

        if use_hybrid:
            text_param_idx = param_idx
            args.append(query_text)
            param_idx += 1

            alpha_param_idx = param_idx
            args.append(alpha)
            param_idx += 1

            one_minus_alpha_param_idx = param_idx
            args.append(1.0 - alpha)
            param_idx += 1

            candidate_limit_param_idx = param_idx
            args.append(top_k * self._candidate_multiplier)
            param_idx += 1

            limit_param_idx = param_idx
            args.append(top_k)

            query = f"""
                WITH candidates AS (
                    SELECT id, source_id, tsv,
                           {score_expr} AS vec_score
                    FROM {self._table}
                    {where}
                    ORDER BY {order_expr}
                    LIMIT ${candidate_limit_param_idx}
                )
                SELECT id, source_id,
                       ${alpha_param_idx} * vec_score
                       + ${one_minus_alpha_param_idx} * COALESCE(ts_rank(tsv, plainto_tsquery('{self._tsv_language}', ${text_param_idx})), 0)
                       AS score
                FROM candidates
                ORDER BY score DESC
                LIMIT ${limit_param_idx}
            """
        else:
            limit_param_idx = param_idx
            args.append(top_k)

            query = f"""
                SELECT id, source_id,
                       {score_expr} AS score
                FROM {self._table}
                {where}
                ORDER BY {order_expr}
                LIMIT ${limit_param_idx}
            """

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._set_tenant(conn, tenant_id)
                await conn.execute(f"SET LOCAL hnsw.ef_search = {int(self._ef_search)}")
                if self._iterative_scan:
                    await conn.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
                rows = await conn.fetch(query, *args)

        return [
            SearchResult(
                chunk_id=row["id"],
                source_id=row["source_id"],
                score=float(row["score"]),
            )
            for row in rows
        ]

    async def delete_by_source(self, tenant_id: UUID, source_id: str) -> list[UUID]:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._set_tenant(conn, tenant_id)
                rows = await conn.fetch(
                    f"DELETE FROM {self._table} WHERE tenant_id = $1 AND source_id = $2 RETURNING id",
                    tenant_id,
                    source_id,
                )

        ids = [row["id"] for row in rows]
        logger.info("Deleted %d chunks for source_id=%s tenant=%s", len(ids), source_id, tenant_id)
        return ids

    async def delete(self, tenant_id: UUID, ids: list[UUID]) -> int:
        if not ids:
            return 0

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._set_tenant(conn, tenant_id)
                result = await conn.execute(
                    f"DELETE FROM {self._table} WHERE id = ANY($1) AND tenant_id = $2",
                    ids,
                    tenant_id,
                )

        deleted = int(result.split()[-1])
        logger.info("Deleted %d chunks by ID for tenant=%s", deleted, tenant_id)
        return deleted

    async def count(self, tenant_id: UUID) -> int:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._set_tenant(conn, tenant_id)
                return await conn.fetchval(
                    f"SELECT count(*) FROM {self._table} WHERE latest = true AND tenant_id = $1",
                    tenant_id,
                )
