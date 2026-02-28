from __future__ import annotations

import logging

import numpy as np

from embedding.base import Embedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(Embedder):
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
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            emb = self._model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.append(np.array(emb))
        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, self._dimension))

    async def embed_query(self, query: str) -> np.ndarray:
        emb = self._model.encode([query], show_progress_bar=False, normalize_embeddings=True)
        return np.array(emb[0])
