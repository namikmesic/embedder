from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from agents.orchestrator import BootstrapOrchestrator
from config.models import PipelineConfig, RepoConfig
from domain.document import Document
from domain.knowledge import BootstrapMap, KnowledgeDoc
from embedding.base import Embedder
from ingest.chunker import Chunker
from sources import create_source
from state import StateManager
from stores.base import VectorStore

logger = logging.getLogger(__name__)


async def run_bootstrap(
    repo_config: RepoConfig,
    pipeline_config: PipelineConfig,
    embedder: Embedder,
    store: VectorStore,
    chunker: Chunker,
    state_mgr: StateManager,
    force: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    repo_id = repo_config.repo_id
    agent_config = pipeline_config.agents
    data_dir = Path(pipeline_config.store.persist_dir).parent

    map_path = data_dir / "maps" / f"{repo_id}.json"
    if map_path.exists() and not force:
        return {
            "repo_id": repo_id,
            "status": "already_bootstrapped",
            "message": "Use --force to re-bootstrap",
        }

    state_mgr.set_status(repo_id, "ingesting")

    try:
        if progress_callback:
            progress_callback(0.02, "Fetching source content...")
        source = create_source(repo_config, pipeline_config.repos_dir)
        head_sha = await source.fetch()

        orchestrator = BootstrapOrchestrator(
            config=agent_config,
            source=source,
            repo_id=repo_id,
            repo_url=repo_config.url,
            file_globs=repo_config.file_globs,
            exclude_patterns=repo_config.exclude_patterns,
            progress_callback=progress_callback,
        )

        bootstrap_map, knowledge_docs = await orchestrator.run()

        if progress_callback:
            progress_callback(0.91, "Persisting map and docs...")

        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_text(bootstrap_map.model_dump_json(indent=2))

        docs_dir = data_dir / "docs" / repo_id
        docs_dir.mkdir(parents=True, exist_ok=True)
        if force:
            for old_doc in docs_dir.glob("*.md"):
                old_doc.unlink()

        for doc in knowledge_docs:
            safe_title = doc.doc_id + "_" + "".join(c if c.isalnum() or c in "-_" else "_" for c in doc.title)
            doc_path = docs_dir / f"{safe_title}.md"
            doc_path.write_text(doc.content)

        if progress_callback:
            progress_callback(0.93, "Replacing embeddings...")
        await store.delete_by_metadata("repo_id", repo_id)

        documents = _docs_to_documents(knowledge_docs, repo_config)

        if progress_callback:
            progress_callback(0.94, f"Chunking {len(documents)} documents...")
        chunks = chunker.chunk_documents(documents)

        repo_info = source.get_info()
        for chunk in chunks:
            chunk.metadata.update({
                "repo_branch": repo_info.get("branch", ""),
                "repo_head_sha": repo_info.get("head_sha", ""),
            })

        if chunks:
            if progress_callback:
                progress_callback(0.96, f"Embedding {len(chunks)} chunks...")
            texts = [c.text for c in chunks]
            vectors = await embedder.embed_texts(texts)

            ids = [c.chunk_id for c in chunks]
            metadatas = [c.metadata for c in chunks]
            await store.add(ids, vectors, metadatas, texts)
            await store.save()

        total_chunks = await store.count()
        state_mgr.update_ingestion(repo_id, head_sha, total_chunks, "ready")

        if progress_callback:
            progress_callback(1.0, "Bootstrap complete!")

        return {
            "repo_id": repo_id,
            "status": "complete",
            "entities": len(bootstrap_map.entities),
            "connections": len(bootstrap_map.connections),
            "docs_generated": len(knowledge_docs),
            "chunks_created": len(chunks),
            "total_chunks_in_store": total_chunks,
            "tokens_consumed": bootstrap_map.total_tokens_consumed,
        }

    except Exception as e:
        state_mgr.set_status(repo_id, "error", str(e))
        logger.exception("Bootstrap failed for repo %s", repo_id)
        return {"repo_id": repo_id, "status": "error", "error": str(e)}


def _docs_to_documents(docs: list[KnowledgeDoc], repo_config: RepoConfig) -> list[Document]:
    documents = []
    for doc in docs:
        documents.append(Document(
            text=f"# {doc.title}\n\n{doc.content}",
            metadata={
                "repo_id": repo_config.repo_id,
                "repo_url": repo_config.url,
                "source_type": "generated",
                "doc_type": doc.doc_type,
                "doc_id": doc.doc_id,
                "title": doc.title,
                "source_entities": doc.source_entities,
            },
        ))
    return documents


def load_bootstrap_map(data_dir: str, repo_id: str) -> Optional[BootstrapMap]:
    map_path = Path(data_dir) / "maps" / f"{repo_id}.json"
    if not map_path.exists():
        return None
    return BootstrapMap.model_validate_json(map_path.read_text())
