from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from config import EmbedderConfig, EmbedderProvider

logger = logging.getLogger(__name__)


class Embedder(ABC):

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def max_tokens(self) -> int: ...

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

    async def close(self) -> None:
        """Release resources. Override in subclasses that hold clients."""


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        self._model = SentenceTransformer(model_name)
        self._model_id = model_name
        self._batch_size = batch_size
        self._dimension = self._model.get_sentence_embedding_dimension()
        self._max_tokens: int = self._model.max_seq_length
        logger.info("Loaded SentenceTransformer model: %s (dim=%d, max_tokens=%d)", model_name, self._dimension, self._max_tokens)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer.encode(text))

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            emb = await asyncio.to_thread(
                self._model.encode, batch, show_progress_bar=False, normalize_embeddings=True
            )
            all_embeddings.append(np.array(emb))
        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, self._dimension))

    async def embed_query(self, query: str) -> np.ndarray:
        emb = await asyncio.to_thread(
            self._model.encode, [query], show_progress_bar=False, normalize_embeddings=True
        )
        return np.array(emb[0])


class OpenAIEmbedder(Embedder):
    _DIM_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    _MAX_TOKENS_MAP = {
        "text-embedding-3-small": 8191,
        "text-embedding-3-large": 8191,
        "text-embedding-ada-002": 8191,
    }

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None, batch_size: int = 64):
        import httpx

        self._model_id = model_name
        self._batch_size = batch_size
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
        self._dimension = self._DIM_MAP.get(model_name, 1536)
        self._encoding: Any = None
        logger.info("OpenAI embedder: %s (dim=%d)", model_name, self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def max_tokens(self) -> int:
        return self._MAX_TOKENS_MAP.get(self._model_id, 8191)

    def count_tokens(self, text: str) -> int:
        if self._encoding is None:
            import tiktoken
            try:
                self._encoding = tiktoken.encoding_for_model(self._model_id)
            except KeyError:
                self._encoding = tiktoken.get_encoding("cl100k_base")
        return len(self._encoding.encode(text))

    async def _embed_batch(self, texts: list[str]) -> np.ndarray:
        response = await self._client.post("/embeddings", json={"input": texts, "model": self._model_id})
        response.raise_for_status()
        embeddings = [item["embedding"] for item in response.json()["data"]]
        return np.array(embeddings, dtype=np.float32)

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            emb = await self._embed_batch(texts[i : i + self._batch_size])
            all_embeddings.append(emb)
        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, self._dimension))

    async def embed_query(self, query: str) -> np.ndarray:
        return (await self._embed_batch([query]))[0]

    async def close(self) -> None:
        await self._client.aclose()


def create_embedder(config: EmbedderConfig) -> Embedder:
    if config.provider == EmbedderProvider.SENTENCE_TRANSFORMERS:
        return SentenceTransformerEmbedder(model_name=config.model, batch_size=config.batch_size)
    elif config.provider == EmbedderProvider.OPENAI:
        if not config.openai_api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")
        return OpenAIEmbedder(model_name=config.model, api_key=config.openai_api_key, batch_size=config.batch_size)
    raise ValueError(f"Unknown embedder provider: {config.provider}")


class EmbedderCache:
    """Cache embedders by (provider, model) key. Not thread-safe — expects external lock."""

    def __init__(self) -> None:
        self._cache: dict[tuple[EmbedderProvider, str], Embedder] = {}

    def get_or_create(self, config: EmbedderConfig) -> Embedder:
        key = (config.provider, config.model)
        if key not in self._cache:
            self._cache[key] = create_embedder(config)
        return self._cache[key]

    async def close_all(self) -> None:
        for emb in self._cache.values():
            await emb.close()
        self._cache.clear()
