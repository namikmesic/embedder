from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from domain.document import SearchResult


class VectorStore(ABC):

    @abstractmethod
    async def add(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict[str, Any]], texts: list[str]) -> None: ...

    @abstractmethod
    async def search(self, query_vector: np.ndarray, top_k: int = 10, filters: Optional[dict[str, Any]] = None) -> list[SearchResult]: ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> int: ...

    @abstractmethod
    async def delete_by_metadata(self, filter_key: str, filter_value: str) -> int: ...

    @abstractmethod
    async def count(self) -> int: ...

    @abstractmethod
    async def save(self) -> None: ...

    @abstractmethod
    async def load(self) -> None: ...
