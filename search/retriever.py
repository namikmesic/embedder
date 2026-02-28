from __future__ import annotations

import logging

from domain.document import SearchResult
from embedding.base import Embedder
from stores.base import VectorStore

logger = logging.getLogger(__name__)


class Retriever:

    def __init__(self, embedder: Embedder, store: VectorStore):
        self._embedder = embedder
        self._store = store

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        repo_ids: list[str] | None = None,
        file_types: list[str] | None = None,
    ) -> list[SearchResult]:
        query_vector = await self._embedder.embed_query(query)

        filters = None
        if repo_ids or file_types:
            filters = {}
            if repo_ids:
                filters["repo_id"] = repo_ids
            if file_types:
                filters["file_extension"] = file_types

        return await self._store.search(query_vector, top_k=top_k, filters=filters)
