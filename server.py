from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from config import PipelineConfig, load_config
from embedding import Embedder, create_embedder
from store import VectorStore, create_store


class VectorDBService:
    """Orchestration layer between MCP tools and the store/embedder.

    Encapsulates all business logic so it can be tested without MCP.
    """

    def __init__(self) -> None:
        self._config: PipelineConfig | None = None
        self._embedder: Embedder | None = None
        self._store: VectorStore | None = None

    async def initialize(self) -> None:
        """Load config, create embedder and store. Idempotent."""
        if self._config is not None:
            return
        self._config = load_config()
        self._embedder = create_embedder(self._config.embedder)
        self._store = await create_store(self._config.store, self._embedder)

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            raise RuntimeError("Service not initialized")
        return self._embedder

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            raise RuntimeError("Service not initialized")
        return self._store

    async def store_chunk(self, text: str, created_at: str, title: str = "") -> dict:
        token_count = self.embedder.count_tokens(text)
        if token_count > self.embedder.max_tokens:
            raise ValueError(
                f"Chunk too large ({token_count} tokens, "
                f"max {self.embedder.max_tokens} for {self.embedder.model_id})"
            )

        chunk_id = uuid.uuid4().hex[:16]
        vector = await self.embedder.embed_texts([text])

        dt = datetime.fromisoformat(created_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        metadata = {}
        if title:
            metadata["title"] = title

        await self.store.add([chunk_id], vector, [metadata], [text], created_ats=[dt])

        return {"chunk_id": chunk_id}

    async def search(
        self, query: str, top_k: int = 10, after: str = "", before: str = ""
    ) -> list[dict]:
        query_vector = await self.embedder.embed_query(query)

        after_dt = None
        if after:
            after_dt = datetime.fromisoformat(after)
            if after_dt.tzinfo is None:
                after_dt = after_dt.replace(tzinfo=timezone.utc)

        before_dt = None
        if before:
            before_dt = datetime.fromisoformat(before)
            if before_dt.tzinfo is None:
                before_dt = before_dt.replace(tzinfo=timezone.utc)

        results = await self.store.search(
            query_vector, top_k=top_k, after=after_dt, before=before_dt,
            query_text=query,
        )

        return [
            {"chunk_id": r.chunk_id, "text": r.text, "score": r.score, "metadata": r.metadata, "created_at": r.created_at}
            for r in results
        ]

    async def delete_chunks(self, title: str) -> dict:
        deleted = await self.store.delete_by_metadata("title", title)
        return {"title": title, "chunks_deleted": deleted}

    async def count_chunks(self) -> int:
        return await self.store.count()

    async def close(self) -> None:
        if self._store:
            await self._store.close()


# ---------------------------------------------------------------------------
# Global service + lazy init
# ---------------------------------------------------------------------------

_service: VectorDBService | None = None
_init_lock = asyncio.Lock()


async def _get_service() -> VectorDBService:
    global _service
    async with _init_lock:
        if _service is None:
            _service = VectorDBService()
            await _service.initialize()
    return _service


# ---------------------------------------------------------------------------
# FastMCP lifespan — ensures the asyncpg pool is closed on shutdown
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

@mcp.tool()
async def store_chunk(text: str, created_at: str, title: str = "") -> str:
    """Embed text and store it in the vector database for later retrieval.

    Rejects chunks exceeding the model's token limit (measured via the
    model's own tokenizer). Aim for 256–1024 tokens per chunk.
    """
    try:
        svc = await _get_service()
        return json.dumps(await svc.store_chunk(text, created_at, title))
    except ToolError:
        raise
    except ValueError as e:
        raise ToolError(f"Invalid input: {e}") from e
    except Exception as e:
        raise ToolError(f"Failed to store chunk: {type(e).__name__}") from e


@mcp.tool()
async def search(query: str, top_k: int = 10, after: str = "", before: str = "") -> str:
    """Semantic search across stored knowledge chunks.

    When hybrid_alpha < 1.0, results blend vector similarity with BM25.
    """
    try:
        svc = await _get_service()
        return json.dumps(await svc.search(query, top_k, after, before))
    except ToolError:
        raise
    except ValueError as e:
        raise ToolError(f"Invalid input: {e}") from e
    except Exception as e:
        raise ToolError(f"Failed to search: {type(e).__name__}") from e


@mcp.tool()
async def delete_chunks(title: str) -> str:
    """Delete all stored chunks matching a title."""
    try:
        svc = await _get_service()
        return json.dumps(await svc.delete_chunks(title))
    except ToolError:
        raise
    except ValueError as e:
        raise ToolError(f"Invalid input: {e}") from e
    except Exception as e:
        raise ToolError(f"Failed to delete chunks: {type(e).__name__}") from e


@mcp.tool()
async def count_chunks() -> str:
    """Return the total number of stored chunks."""
    try:
        svc = await _get_service()
        total = await svc.count_chunks()
        return json.dumps({"total_chunks": total})
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Failed to count chunks: {type(e).__name__}") from e


if __name__ == "__main__":
    mcp.run()
