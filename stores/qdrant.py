from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from domain.document import SearchResult
from stores.base import VectorStore

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100


class QdrantStore(VectorStore):
    _UUID_NAMESPACE = uuid.UUID("a3e2b0c1-4f5d-6e7a-8b9c-0d1e2f3a4b5c")

    def __init__(
        self,
        dimension: int,
        persist_dir: str = "./data/qdrant",
        collection_name: str = "git_rag",
    ):
        self._dimension = dimension
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name

        self._client = QdrantClient(path=str(self._persist_dir))
        self._ensure_collection()

    @classmethod
    def _to_uuid(cls, chunk_id: str) -> str:
        return str(uuid.uuid5(cls._UUID_NAMESPACE, chunk_id))

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection_name in collections:
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=self._dimension,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            "Created Qdrant collection '%s' (dim=%d) at %s",
            self._collection_name,
            self._dimension,
            self._persist_dir,
        )

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Filter:
        must: list[FieldCondition] = []
        for key, value in filters.items():
            if isinstance(value, list):
                must.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                must.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=must)

    async def add(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadatas: list[dict[str, Any]],
        texts: list[str],
    ) -> None:
        if len(ids) == 0:
            return

        for start in range(0, len(ids), _BATCH_SIZE):
            end = min(start + _BATCH_SIZE, len(ids))
            points = [
                PointStruct(
                    id=self._to_uuid(ids[i]),
                    vector=vectors[i].tolist(),
                    payload={**metadatas[i], "_text": texts[i], "_chunk_id": ids[i]},
                )
                for i in range(start, end)
            ]
            self._client.upsert(collection_name=self._collection_name, points=points)

        logger.info(
            "Added %d vectors to Qdrant collection '%s' (total: %d)",
            len(ids),
            self._collection_name,
            self._client.count(collection_name=self._collection_name).count,
        )

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        qf = self._build_filter(filters) if filters else None

        hits = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector.tolist(),
            query_filter=qf,
            limit=top_k,
            with_payload=True,
        ).points

        results: list[SearchResult] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            text = payload.pop("_text", "")
            chunk_id = payload.pop("_chunk_id", str(hit.id))
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=text,
                    score=hit.score,
                    metadata=payload,
                )
            )
        return results

    async def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0
        uuid_ids = [self._to_uuid(cid) for cid in ids]
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=uuid_ids,
        )
        logger.info("Deleted %d vectors by ID from Qdrant", len(ids))
        return len(ids)

    async def delete_by_metadata(self, filter_key: str, filter_value: str) -> int:
        filt = self._build_filter({filter_key: filter_value})
        count = self._client.count(
            collection_name=self._collection_name,
            count_filter=filt,
            exact=True,
        ).count

        if count == 0:
            return 0

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=filt,
        )
        logger.info("Deleted %d vectors where %s=%s", count, filter_key, filter_value)
        return count

    async def count(self) -> int:
        return self._client.count(
            collection_name=self._collection_name, exact=True
        ).count

    async def save(self) -> None:
        pass

    async def load(self) -> None:
        pass
