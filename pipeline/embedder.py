"""Embedder: abstract interface + SentenceTransformer and OpenAI implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from models import EmbedderConfig, EmbedderProvider

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (N, dim) array."""
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns (dim,) array."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class SentenceTransformerEmbedder(Embedder):
    """Local embedding using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Loaded SentenceTransformer model: %s (dim=%d)", model_name, self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        # sentence-transformers is sync; run in batches
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            emb = self._model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.append(np.array(emb))
        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, self._dimension))

    async def embed_query(self, query: str) -> np.ndarray:
        emb = self._model.encode([query], show_progress_bar=False, normalize_embeddings=True)
        return np.array(emb[0])


class OpenAIEmbedder(Embedder):
    """Embedding via OpenAI API."""

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
        # Dimensions for common models
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


def create_embedder(config: EmbedderConfig) -> Embedder:
    """Factory function to create the appropriate embedder."""
    if config.provider == EmbedderProvider.SENTENCE_TRANSFORMERS:
        return SentenceTransformerEmbedder(
            model_name=config.model_name,
            batch_size=config.batch_size,
        )
    elif config.provider == EmbedderProvider.OPENAI:
        if not config.openai_api_key:
            raise ValueError("OpenAI API key required when using OpenAI embedder. Set OPENAI_API_KEY env var.")
        return OpenAIEmbedder(
            model_name=config.model_name,
            api_key=config.openai_api_key,
            batch_size=config.batch_size,
        )
    else:
        raise ValueError(f"Unknown embedder provider: {config.provider}")
