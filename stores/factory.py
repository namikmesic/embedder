from __future__ import annotations

from config.models import StoreConfig
from domain.enums import StoreBackend
from embedding.base import Embedder
from stores.base import VectorStore


async def create_store(config: StoreConfig, embedder: Embedder) -> VectorStore:
    if config.backend == StoreBackend.QDRANT:
        from stores.qdrant import QdrantStore

        return QdrantStore(
            dimension=embedder.dimension,
            persist_dir=config.persist_dir,
            collection_name=config.qdrant_collection,
        )
    elif config.backend == StoreBackend.FAISS:
        from stores.faiss import FAISSStore

        store = FAISSStore(dimension=embedder.dimension, persist_dir=config.persist_dir)
        await store.load()
        return store
    else:
        raise ValueError(f"Store backend '{config.backend.value}' not yet implemented.")
