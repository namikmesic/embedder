from __future__ import annotations

import asyncio
import io
import json
import logging
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from miniopy_async import Minio

from config import ObjectStoreConfig

logger = logging.getLogger(__name__)


@dataclass
class ChunkContent:
    text: str
    title: str = ""

    def to_json_bytes(self) -> bytes:
        return json.dumps({"text": self.text, "title": self.title}).encode()

    @classmethod
    def from_json_bytes(cls, data: bytes) -> ChunkContent:
        obj = json.loads(data)
        return cls(text=obj["text"], title=obj.get("title", ""))


class ObjectStore:

    def __init__(self, config: ObjectStoreConfig) -> None:
        self._config = config
        self._client: Minio | None = None

    async def initialize(self) -> None:
        self._client = Minio(
            self._config.endpoint,
            access_key=self._config.access_key,
            secret_key=self._config.secret_key,
            secure=self._config.secure,
        )
        bucket = self._config.bucket
        if not await self._client.bucket_exists(bucket):
            await self._client.make_bucket(bucket)
            logger.info("Created MinIO bucket: %s", bucket)
        logger.info("ObjectStore initialized: endpoint=%s bucket=%s", self._config.endpoint, bucket)

    def _key(self, tenant_id: UUID, kb_id: str, chunk_id: UUID) -> str:
        return f"{tenant_id}/{kb_id}/{chunk_id}.json"

    async def put(self, tenant_id: UUID, chunk_id: UUID, content: ChunkContent, kb_id: str) -> None:
        client: Minio = self._client  # type: ignore[assignment]
        data = content.to_json_bytes()
        await client.put_object(
            self._config.bucket,
            self._key(tenant_id, kb_id, chunk_id),
            io.BytesIO(data),
            length=len(data),
            content_type="application/json",
        )

    async def get(self, tenant_id: UUID, chunk_id: UUID, kb_id: str) -> ChunkContent:
        client: Minio = self._client  # type: ignore[assignment]
        response = await client.get_object(
            self._config.bucket,
            self._key(tenant_id, kb_id, chunk_id),
        )
        try:
            data = await response.read()
        finally:
            response.close()
            await response.release()
        return ChunkContent.from_json_bytes(data)

    async def _gather_per_chunk(
        self,
        chunk_ids: list[UUID],
        coros: list[Coroutine[Any, Any, Any]],
        action: str,
    ) -> list[tuple[UUID, Any]]:
        """Run one coroutine per chunk, log failures, return (id, result) for successes."""
        results = await asyncio.gather(*coros, return_exceptions=True)
        successes: list[tuple[UUID, Any]] = []
        for cid, result in zip(chunk_ids, results):
            if isinstance(result, Exception):
                logger.warning("Failed to %s chunk %s in MinIO: %s", action, cid, result)
            else:
                successes.append((cid, result))
        return successes

    async def get_many(self, tenant_id: UUID, chunk_ids: list[UUID], kb_id: str) -> dict[UUID, ChunkContent]:
        if not chunk_ids:
            return {}
        pairs = await self._gather_per_chunk(
            chunk_ids,
            [self.get(tenant_id, cid, kb_id) for cid in chunk_ids],
            "fetch",
        )
        return dict(pairs)

    async def delete(self, tenant_id: UUID, chunk_id: UUID, kb_id: str) -> None:
        client: Minio = self._client  # type: ignore[assignment]
        await client.remove_object(
            self._config.bucket,
            self._key(tenant_id, kb_id, chunk_id),
        )

    async def delete_many(self, tenant_id: UUID, chunk_ids: list[UUID], kb_id: str) -> None:
        if not chunk_ids:
            return
        await self._gather_per_chunk(
            chunk_ids,
            [self.delete(tenant_id, cid, kb_id) for cid in chunk_ids],
            "delete",
        )

    async def close(self) -> None:
        # Minio client is stateless HTTP — no persistent connection to close
        self._client = None
