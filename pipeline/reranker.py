"""Reranker: cross-encoder reranking for improved precision."""

from __future__ import annotations

import logging
from typing import Optional

from models import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model: Optional[object] = None

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

            self._model = CrossEncoder(self._model_name)
            logger.info("Loaded reranker model: %s", self._model_name)

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_n: Optional[int] = None,
    ) -> list[SearchResult]:
        """Re-score candidates using a cross-encoder and return top-N.

        Args:
            query: The original query text.
            candidates: List of SearchResult from the retriever.
            top_n: Number of results to return (None = return all, re-sorted).

        Returns:
            Reranked list of SearchResult with updated scores.
        """
        if not candidates:
            return []

        self._load_model()

        # Prepare (query, text) pairs for the cross-encoder
        pairs = [(query, result.text) for result in candidates]
        scores = self._model.predict(pairs)  # type: ignore[union-attr]

        # Update scores and sort
        reranked = []
        for result, score in zip(candidates, scores):
            reranked.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=float(score),
                    metadata={**result.metadata, "retriever_score": result.score},
                )
            )

        reranked.sort(key=lambda r: r.score, reverse=True)

        if top_n is not None:
            reranked = reranked[:top_n]

        return reranked
