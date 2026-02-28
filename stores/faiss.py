from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import faiss  # type: ignore[import-untyped]
import numpy as np

from domain.document import SearchResult
from stores.base import VectorStore

logger = logging.getLogger(__name__)


class FAISSStore(VectorStore):
    def __init__(self, dimension: int, persist_dir: str = "./data/faiss"):
        self._dimension = dimension
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._index = faiss.IndexFlatIP(dimension)
        self._ids: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._texts: list[str] = []

    @property
    def _index_path(self) -> Path:
        return self._persist_dir / "index.faiss"

    @property
    def _meta_path(self) -> Path:
        return self._persist_dir / "metadata.json"

    async def add(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadatas: list[dict[str, Any]],
        texts: list[str],
    ) -> None:
        if len(ids) == 0:
            return

        assert vectors.shape == (len(ids), self._dimension), (
            f"Expected shape ({len(ids)}, {self._dimension}), got {vectors.shape}"
        )

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = (vectors / norms).astype(np.float32)

        self._index.add(normalized)
        self._ids.extend(ids)
        self._metadatas.extend(metadatas)
        self._texts.extend(texts)

        logger.info("Added %d vectors to FAISS store (total: %d)", len(ids), self._index.ntotal)

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        if self._index.ntotal == 0:
            return []

        qv = query_vector.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv = qv / norm

        search_k = min(top_k * 5, self._index.ntotal) if filters else min(top_k, self._index.ntotal)
        scores, indices = self._index.search(qv, search_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing
                continue

            meta = self._metadatas[idx]

            if filters:
                match = all(meta.get(k) == v for k, v in filters.items())
                if not match:
                    continue

            results.append(
                SearchResult(
                    chunk_id=self._ids[idx],
                    text=self._texts[idx],
                    score=float(score),
                    metadata=meta,
                )
            )

            if len(results) >= top_k:
                break

        return results

    async def delete(self, ids: list[str]) -> int:
        id_set = set(ids)
        keep_indices = [i for i, cid in enumerate(self._ids) if cid not in id_set]
        deleted = len(self._ids) - len(keep_indices)

        if deleted == 0:
            return 0

        self._rebuild_index(keep_indices)
        logger.info("Deleted %d vectors from FAISS store", deleted)
        return deleted

    async def delete_by_metadata(self, filter_key: str, filter_value: str) -> int:
        keep_indices = [
            i for i, meta in enumerate(self._metadatas) if meta.get(filter_key) != filter_value
        ]
        deleted = len(self._ids) - len(keep_indices)

        if deleted == 0:
            return 0

        self._rebuild_index(keep_indices)
        logger.info("Deleted %d vectors where %s=%s", deleted, filter_key, filter_value)
        return deleted

    def _rebuild_index(self, keep_indices: list[int]) -> None:
        if not keep_indices:
            self._index = faiss.IndexFlatIP(self._dimension)
            self._ids = []
            self._metadatas = []
            self._texts = []
            return

        vectors = np.array([self._index.reconstruct(i) for i in keep_indices], dtype=np.float32)

        self._ids = [self._ids[i] for i in keep_indices]
        self._metadatas = [self._metadatas[i] for i in keep_indices]
        self._texts = [self._texts[i] for i in keep_indices]

        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(vectors)

    async def count(self) -> int:
        return self._index.ntotal

    async def save(self) -> None:
        faiss.write_index(self._index, str(self._index_path))
        meta = {
            "ids": self._ids,
            "metadatas": self._metadatas,
            "texts": self._texts,
            "dimension": self._dimension,
        }
        self._meta_path.write_text(json.dumps(meta, default=str))
        logger.info("Saved FAISS store (%d vectors) to %s", self._index.ntotal, self._persist_dir)

    async def load(self) -> None:
        if not self._index_path.exists():
            logger.info("No existing FAISS index at %s, starting fresh", self._index_path)
            return

        self._index = faiss.read_index(str(self._index_path))
        meta = json.loads(self._meta_path.read_text())
        self._ids = meta["ids"]
        self._metadatas = meta["metadatas"]
        self._texts = meta["texts"]
        self._dimension = meta.get("dimension", self._dimension)
        logger.info("Loaded FAISS store: %d vectors from %s", self._index.ntotal, self._persist_dir)
