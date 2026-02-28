from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from embedding.base import Embedder

logger = logging.getLogger(__name__)


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 64,
    ):
        import httpx

        self._model_name = model_name
        self._api_key = api_key
        self._batch_size = batch_size
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
        dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self._dimension = dim_map.get(model_name, 1536)
        logger.info("OpenAI embedder: %s (dim=%d)", model_name, self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    async def _embed_batch(self, texts: list[str]) -> np.ndarray:
        response = await self._client.post(
            "/embeddings",
            json={"input": texts, "model": self._model_name},
        )
        response.raise_for_status()
        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return np.array(embeddings, dtype=np.float32)

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            emb = await self._embed_batch(batch)
            all_embeddings.append(emb)
        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, self._dimension))

    async def embed_query(self, query: str) -> np.ndarray:
        result = await self._embed_batch([query])
        return result[0]
