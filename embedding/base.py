from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...
