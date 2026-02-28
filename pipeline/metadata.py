"""Metadata enrichment for chunks."""

from __future__ import annotations

from models import Chunk


def enrich_chunk_metadata(chunk: Chunk, repo_info: dict) -> Chunk:
    """Add repo-level metadata to a chunk."""
    chunk.metadata.update({
        "repo_branch": repo_info.get("branch", ""),
        "repo_head_sha": repo_info.get("head_sha", ""),
    })
    return chunk
