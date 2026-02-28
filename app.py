from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click

from config.models import PipelineConfig
from embedding.base import Embedder
from embedding.factory import create_embedder
from ingest.chunker import Chunker
from ingest.preprocessor import Preprocessor
from search.reranker import Reranker
from search.retriever import Retriever
from state import StateManager
from stores.base import VectorStore
from stores.factory import create_store


@dataclass
class AppContext:
    config: PipelineConfig
    embedder: Embedder
    store: VectorStore
    preprocessor: Preprocessor
    chunker: Chunker
    retriever: Retriever
    reranker: Reranker
    state: StateManager


pass_app = click.make_pass_decorator(AppContext)


def _run(coro):
    import anyio

    async def _wrapper():
        return await coro

    return anyio.run(_wrapper)


async def _init_app(config: PipelineConfig) -> AppContext:
    embedder = create_embedder(config.embedder)
    store = await create_store(config.store, embedder)

    preprocessor = Preprocessor()
    chunker = Chunker(config.chunk)
    retriever = Retriever(embedder, store)
    reranker = Reranker()
    state_mgr = StateManager(state_path=str(Path(config.store.persist_dir).parent / "state.json"))

    return AppContext(
        config=config,
        embedder=embedder,
        store=store,
        preprocessor=preprocessor,
        chunker=chunker,
        retriever=retriever,
        reranker=reranker,
        state=state_mgr,
    )
