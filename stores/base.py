"""Abstract base class for vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from models import SearchResult


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    async def add(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadatas: list[dict[str, Any]],
        texts: list[str],
    ) -> None:
        """Add vectors with associated metadata and text.

        Args:
            ids: Unique identifiers for each vector.
            vectors: (N, dim) array of embedding vectors.
            metadatas: List of metadata dicts, one per vector.
            texts: Original text for each vector.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            query_vector: (dim,) query embedding.
            top_k: Number of results to return.
            filters: Optional metadata filters. Supports exact match
                (``{"key": "value"}``) and list-of-values / IN semantics
                (``{"key": ["v1", "v2"]}``). Multiple keys are AND-ed.

        Returns:
            List of SearchResult sorted by relevance (highest first).
        """
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> int:
        """Delete vectors by their IDs. Returns count of deleted."""
        ...

    @abstractmethod
    async def delete_by_metadata(self, filter_key: str, filter_value: str) -> int:
        """Delete all vectors matching a metadata filter. Returns count."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return total number of vectors in the store."""
        ...

    @abstractmethod
    async def save(self) -> None:
        """Persist the store to disk."""
        ...

    @abstractmethod
    async def load(self) -> None:
        """Load the store from disk."""
        ...
