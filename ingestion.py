"""Ingestion pipeline — clones/pulls a repo and indexes its contents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from models import PipelineConfig
from pipeline.chunker import Chunker
from pipeline.embedder import Embedder
from pipeline.git_loader import GitLoader
from pipeline.metadata import enrich_chunk_metadata
from pipeline.preprocessor import Preprocessor
from state import RepoState, StateManager
from stores.base import VectorStore

logger = logging.getLogger(__name__)


async def ingest_repo(
    repo_state: RepoState,
    config: PipelineConfig,
    embedder: Embedder,
    store: VectorStore,
    preprocessor: Preprocessor,
    chunker: Chunker,
    state_mgr: StateManager,
    force_full: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """Run the ingestion pipeline for a single repo."""
    rc = repo_state.config
    state_mgr.set_status(rc.repo_id, "ingesting")

    try:
        # 1. Git clone/pull
        local_path = Path(rc.local_path or (Path(config.repos_dir) / rc.repo_id))
        loader = GitLoader(rc.url, local_path, rc.branch)
        head_sha = loader.clone_or_pull()
        repo_info = loader.get_repo_info()

        # 2. Determine files to process
        if force_full or not repo_state.last_ingested_commit:
            # Full ingestion
            files = loader.discover_files(rc.file_globs, rc.exclude_patterns)
            await store.delete_by_metadata("repo_id", rc.repo_id)
        else:
            # Incremental: only changed files
            changed = loader.get_changed_files(repo_state.last_ingested_commit)
            if not changed:
                state_mgr.update_ingestion(rc.repo_id, head_sha, repo_state.chunk_count, "ready")
                return {"repo_id": rc.repo_id, "status": "up_to_date", "files_processed": 0}

            for f in changed:
                await store.delete_by_metadata("source_path", str(local_path / f))

            deleted = loader.get_deleted_files(repo_state.last_ingested_commit)
            for f in deleted:
                await store.delete_by_metadata("source_path", str(local_path / f))

            import fnmatch

            files = []
            for f in changed:
                fpath = local_path / f
                if fpath.exists() and any(fnmatch.fnmatch(f, g) for g in rc.file_globs):
                    if not any(fnmatch.fnmatch(f, e) for e in rc.exclude_patterns):
                        files.append(fpath)

        if progress_callback:
            progress_callback(0.2, f"Processing {len(files)} files...")

        # 3. Preprocess
        documents = []
        for fpath in files:
            content = loader.get_file_content(fpath)
            doc = preprocessor.process_file(fpath, content, rc.url, rc.repo_id)
            if doc:
                documents.append(doc)

        if progress_callback:
            progress_callback(0.4, f"Chunking {len(documents)} documents...")

        # 4. Chunk
        chunks = chunker.chunk_documents(documents)
        for chunk in chunks:
            enrich_chunk_metadata(chunk, repo_info)

        if not chunks:
            state_mgr.update_ingestion(rc.repo_id, head_sha, await store.count(), "ready")
            return {"repo_id": rc.repo_id, "status": "ready", "files_processed": len(files), "chunks_created": 0}

        if progress_callback:
            progress_callback(0.6, f"Embedding {len(chunks)} chunks...")

        # 5. Embed
        texts = [c.text for c in chunks]
        vectors = await embedder.embed_texts(texts)

        if progress_callback:
            progress_callback(0.8, "Writing to vector store...")

        # 6. Write to store
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]
        await store.add(ids, vectors, metadatas, texts)

        # 7. Persist
        await store.save()

        total_chunks = await store.count()
        state_mgr.update_ingestion(rc.repo_id, head_sha, total_chunks, "ready")

        if progress_callback:
            progress_callback(1.0, "Done!")

        return {
            "repo_id": rc.repo_id,
            "status": "ready",
            "files_processed": len(files),
            "documents_created": len(documents),
            "chunks_created": len(chunks),
            "total_chunks_in_store": total_chunks,
            "head_sha": head_sha,
        }

    except Exception as e:
        state_mgr.set_status(rc.repo_id, "error", str(e))
        logger.exception("Ingestion failed for repo %s", rc.repo_id)
        return {"repo_id": rc.repo_id, "status": "error", "error": str(e)}
