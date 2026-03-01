from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator
from uuid import UUID

import uuid_utils
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from config import DistanceMetric, EmbedderProvider, VectorPrecision, load_config
from knowledgebase import KnowledgebaseManager
from object_store import ChunkContent

logger = logging.getLogger(__name__)

MAX_TOP_K = 200


def _parse_utc(iso: str) -> datetime:
    """Parse ISO 8601 string, defaulting to UTC if no timezone."""
    dt = datetime.fromisoformat(iso)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class VectorDBService:
    """Orchestration layer between MCP tools and knowledgebases.

    Encapsulates all business logic so it can be tested without MCP.
    """

    def __init__(self, manager: KnowledgebaseManager) -> None:
        self.manager = manager

    @classmethod
    async def create(cls) -> VectorDBService:
        """Build from env-var config. Used by the MCP server entry point."""
        config = load_config()
        manager = KnowledgebaseManager(
            store_config=config.store,
            object_store_config=config.object_store,
            embedder_config=config.embedder,
            kb_defaults=config.kb_defaults,
        )
        await manager.initialize()
        return cls(manager)

    async def store_chunk(
        self,
        text: str,
        source_id: str,
        tenant_id: str,
        content_date: str,
        title: str = "",
        knowledgebase: str = "default",
    ) -> dict:
        record, store, embedder = await self.manager.get_kb(knowledgebase)
        kb_id = str(record.id)

        token_count = embedder.count_tokens(text)
        if token_count > embedder.max_tokens:
            raise ValueError(
                f"Chunk too large ({token_count} tokens, "
                f"max {embedder.max_tokens} for {embedder.model_id})"
            )

        tid = UUID(tenant_id)
        content_hash = hashlib.sha256(text.encode()).digest()

        # Fast path: if content hasn't changed, skip embedding + MinIO entirely.
        existing_id = await store.check_hash(tid, source_id, content_hash)
        if existing_id is not None:
            return {"status": "unchanged", "chunk_id": str(existing_id)}

        dt = _parse_utc(content_date)
        content_ts_ms = int(dt.timestamp() * 1000)
        chunk_id = UUID(int=uuid_utils.uuid7(timestamp=content_ts_ms).int)

        vector = await embedder.embed_texts([text])

        # Write content to MinIO first (pre-flight).
        await self.manager.object_store.put(
            tid, chunk_id, ChunkContent(text=text, title=title), kb_id
        )

        result = await store.upsert(
            chunk_id=chunk_id,
            tenant_id=tid,
            source_id=source_id,
            embedding=vector[0],
            content_hash=content_hash,
            text_for_tsv=text,
        )

        # Race: another writer matched the hash between check_hash and upsert
        if not result.wrote:
            return {"status": "unchanged", "chunk_id": str(result.prior_id)}

        status = "updated" if result.prior_id is not None else "created"
        return {"chunk_id": str(chunk_id), "status": status}

    async def search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 10,
        after: str = "",
        before: str = "",
        knowledgebase: str = "default",
    ) -> list[dict]:
        if top_k > MAX_TOP_K:
            raise ValueError(f"top_k must be <= {MAX_TOP_K}, got {top_k}")

        record, store, embedder = await self.manager.get_kb(knowledgebase)
        kb_id = str(record.id)
        tid = UUID(tenant_id)

        after_dt = _parse_utc(after) if after else None
        before_dt = _parse_utc(before) if before else None

        query_vector = await embedder.embed_query(query)

        results = await store.search(
            query_vector,
            tenant_id=tid,
            top_k=top_k,
            after=after_dt,
            before=before_dt,
            query_text=query,
        )

        if not results:
            return []

        chunk_ids = [r.chunk_id for r in results]
        from_minio = await self.manager.object_store.get_many(tid, chunk_ids, kb_id)

        output = []
        for r in results:
            content = from_minio.get(r.chunk_id)
            if content is None:
                logger.warning("Chunk %s not found in MinIO", r.chunk_id)
                continue

            output.append({
                "chunk_id": str(r.chunk_id),
                "source_id": r.source_id,
                "score": r.score,
                "text": content.text,
                "title": content.title,
            })

        return output

    async def delete_chunks(
        self,
        source_id: str,
        tenant_id: str,
        knowledgebase: str = "default",
    ) -> dict:
        record, store, _ = await self.manager.get_kb(knowledgebase)
        kb_id = str(record.id)
        tid = UUID(tenant_id)

        deleted_ids = await store.delete_by_source(tid, source_id)

        if deleted_ids:
            await self.manager.object_store.delete_many(tid, deleted_ids, kb_id)

        return {"source_id": source_id, "chunks_deleted": len(deleted_ids)}

    async def count_chunks(
        self,
        tenant_id: str,
        knowledgebase: str = "default",
    ) -> dict:
        _, store, _ = await self.manager.get_kb(knowledgebase)
        tid = UUID(tenant_id)
        total = await store.count(tid)
        return {"total_chunks": total}

    async def create_knowledgebase(
        self,
        name: str,
        embedder_provider: str = "",
        embedder_model: str = "",
        precision: str = "",
        distance_metric: str = "",
        hnsw_m: int | None = None,
        hnsw_ef_construction: int | None = None,
        tsv_language: str = "",
        ef_search: int | None = None,
        iterative_scan: bool | None = None,
        hybrid_alpha: float | None = None,
        candidate_multiplier: int | None = None,
    ) -> dict:
        record = await self.manager.create_kb(
            name=name,
            embedder_provider=EmbedderProvider(embedder_provider) if embedder_provider else None,
            embedder_model=embedder_model or None,
            precision=VectorPrecision(precision) if precision else None,
            distance_metric=DistanceMetric(distance_metric) if distance_metric else None,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            tsv_language=tsv_language or None,
            ef_search=ef_search,
            iterative_scan=iterative_scan,
            hybrid_alpha=hybrid_alpha,
            candidate_multiplier=candidate_multiplier,
        )
        return {
            "name": record.name,
            "id": str(record.id),
            "table_name": record.table_name,
            "dimension": record.dimension,
            "embedder_model": record.embedder_model,
            "status": "created",
        }

    async def list_knowledgebases(self) -> list[dict]:
        records = await self.manager.list_kbs()
        return [
            {
                "name": r.name,
                "id": str(r.id),
                "table_name": r.table_name,
                "embedder_provider": r.embedder_provider,
                "embedder_model": r.embedder_model,
                "dimension": r.dimension,
                "precision": r.precision,
                "distance_metric": r.distance_metric,
                "tsv_language": r.tsv_language,
                "ef_search": r.ef_search,
                "hybrid_alpha": r.hybrid_alpha,
            }
            for r in records
        ]

    async def close(self) -> None:
        await self.manager.close()


# ---------------------------------------------------------------------------
# Global service + lazy init
# ---------------------------------------------------------------------------

_service: VectorDBService | None = None
_init_lock = asyncio.Lock()


async def _get_service() -> VectorDBService:
    global _service
    async with _init_lock:
        if _service is None:
            _service = await VectorDBService.create()
    return _service


# ---------------------------------------------------------------------------
# FastMCP lifespan — ensures all connections are closed on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    yield {}
    if _service is not None:
        await _service.close()


mcp = FastMCP("vectordb", lifespan=app_lifespan)


# ---------------------------------------------------------------------------
# MCP Tool wrappers (thin — all logic lives in VectorDBService)
# ---------------------------------------------------------------------------

async def _run_tool(
    fn: Callable[[VectorDBService], Awaitable[dict | list]],
    error_msg: str,
) -> str:
    """Shared error handling for every MCP tool wrapper."""
    try:
        svc = await _get_service()
        return json.dumps(await fn(svc))
    except ToolError:
        raise
    except ValueError as e:
        raise ToolError(f"Invalid input: {e}") from e
    except Exception as e:
        raise ToolError(f"{error_msg}: {type(e).__name__}") from e


@mcp.tool()
async def store_chunk(
    text: str,
    source_id: str,
    tenant_id: str,
    content_date: str,
    title: str = "",
    knowledgebase: str = "default",
) -> str:
    """Embed text and store it in the vector database for later retrieval.

    Rejects chunks exceeding the model's token limit (measured via the
    model's own tokenizer). Aim for 256-1024 tokens per chunk.
    """
    return await _run_tool(
        lambda svc: svc.store_chunk(text, source_id, tenant_id, content_date, title, knowledgebase),
        "Failed to store chunk",
    )


@mcp.tool()
async def search(
    query: str,
    tenant_id: str,
    top_k: int = 10,
    after: str = "",
    before: str = "",
    knowledgebase: str = "default",
) -> str:
    """Semantic search across stored knowledge chunks.

    When hybrid_alpha < 1.0, results blend vector similarity with BM25.
    """
    return await _run_tool(
        lambda svc: svc.search(query, tenant_id, top_k, after, before, knowledgebase),
        "Failed to search",
    )


@mcp.tool()
async def delete_chunks(
    source_id: str,
    tenant_id: str,
    knowledgebase: str = "default",
) -> str:
    """Delete all stored chunks matching a source_id for a given tenant."""
    return await _run_tool(
        lambda svc: svc.delete_chunks(source_id, tenant_id, knowledgebase),
        "Failed to delete chunks",
    )


@mcp.tool()
async def count_chunks(
    tenant_id: str,
    knowledgebase: str = "default",
) -> str:
    """Return the total number of stored chunks."""
    return await _run_tool(
        lambda svc: svc.count_chunks(tenant_id, knowledgebase),
        "Failed to count chunks",
    )


@mcp.tool()
async def create_knowledgebase(
    name: str,
    embedder_model: str = "",
    embedder_provider: str = "",
    precision: str = "",
    distance_metric: str = "",
    hnsw_m: int | None = None,
    hnsw_ef_construction: int | None = None,
    tsv_language: str = "",
    ef_search: int | None = None,
    iterative_scan: bool | None = None,
    hybrid_alpha: float | None = None,
    candidate_multiplier: int | None = None,
) -> str:
    """Create a new knowledgebase with its own table and embedding model."""
    return await _run_tool(
        lambda svc: svc.create_knowledgebase(
            name, embedder_provider, embedder_model, precision,
            distance_metric, hnsw_m, hnsw_ef_construction, tsv_language,
            ef_search, iterative_scan, hybrid_alpha, candidate_multiplier,
        ),
        "Failed to create knowledgebase",
    )


@mcp.tool()
async def list_knowledgebases() -> str:
    """List all knowledgebases and their configuration."""
    return await _run_tool(
        lambda svc: svc.list_knowledgebases(),
        "Failed to list knowledgebases",
    )


if __name__ == "__main__":
    mcp.run()
