from __future__ import annotations

import fnmatch
import logging
from typing import Any, Callable, Optional

from config.models import PipelineConfig
from embedding.base import Embedder
from ingest.chunker import Chunker
from ingest.preprocessor import Preprocessor
from sources.factory import create_source
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
    rc = repo_state.config
    state_mgr.set_status(rc.repo_id, "ingesting")

    try:
        source = create_source(rc, config.repos_dir)
        head_sha = await source.fetch()
        repo_info = source.get_info()

        if force_full or not repo_state.last_ingested_commit:
            files = source.discover_files(rc.file_globs, rc.exclude_patterns)
            await store.delete_by_metadata("repo_id", rc.repo_id)
        else:
            changed = source.get_changed_files(repo_state.last_ingested_commit)
            if not changed:
                state_mgr.update_ingestion(rc.repo_id, head_sha, repo_state.chunk_count, "ready")
                return {"repo_id": rc.repo_id, "status": "up_to_date", "files_processed": 0}

            for f in changed:
                await store.delete_by_metadata("source_path", str(source.local_path / f))

            deleted = source.get_deleted_files(repo_state.last_ingested_commit)
            for f in deleted:
                await store.delete_by_metadata("source_path", str(source.local_path / f))

            files = []
            for f in changed:
                fpath = source.local_path / f
                if fpath.exists() and any(fnmatch.fnmatch(f, g) for g in rc.file_globs):
                    if not any(fnmatch.fnmatch(f, e) for e in rc.exclude_patterns):
                        files.append(fpath)

        if progress_callback:
            progress_callback(0.2, f"Processing {len(files)} files...")

        documents = []
        for fpath in files:
            content = source.get_file_content(fpath)
            doc = preprocessor.process_file(fpath, content, rc.url, rc.repo_id)
            if doc:
                documents.append(doc)

        if progress_callback:
            progress_callback(0.4, f"Chunking {len(documents)} documents...")

        chunks = chunker.chunk_documents(documents)
        for chunk in chunks:
            chunk.metadata.update({
                "repo_branch": repo_info.get("branch", ""),
                "repo_head_sha": repo_info.get("head_sha", ""),
            })

        if not chunks:
            state_mgr.update_ingestion(rc.repo_id, head_sha, await store.count(), "ready")
            return {"repo_id": rc.repo_id, "status": "ready", "files_processed": len(files), "chunks_created": 0}

        if progress_callback:
            progress_callback(0.6, f"Embedding {len(chunks)} chunks...")

        texts = [c.text for c in chunks]
        vectors = await embedder.embed_texts(texts)

        if progress_callback:
            progress_callback(0.8, "Writing to vector store...")

        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]
        await store.add(ids, vectors, metadatas, texts)

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
