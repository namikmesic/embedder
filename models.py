"""Pydantic models for the Git RAG MCP server."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────────


class EmbedderProvider(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"


class StoreBackend(str, Enum):
    FAISS = "faiss"
    QDRANT = "qdrant"
    PINECONE = "pinecone"


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


# ── Pipeline config models ──────────────────────────────────────────────────


class RepoConfig(BaseModel):
    """Configuration for a single git repository knowledge base."""

    model_config = ConfigDict(str_strip_whitespace=True)

    repo_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    url: str = Field(..., description="Git clone URL (HTTPS or SSH)")
    branch: str = Field(default="main", description="Branch to track")
    local_path: Optional[str] = Field(default=None, description="Override local clone path")
    file_globs: list[str] = Field(
        default_factory=lambda: ["**/*.md", "**/*.txt", "**/*.py", "**/*.js", "**/*.ts", "**/*.rs", "**/*.go", "**/*.java", "**/*.rst"],
        description="Glob patterns for files to index",
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["**/node_modules/**", "**/.git/**", "**/venv/**", "**/__pycache__/**", "**/dist/**", "**/build/**"],
        description="Glob patterns to exclude",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not (v.startswith("http") or v.startswith("git@") or v.startswith("/")):
            raise ValueError("URL must be an HTTPS URL, SSH URL, or local path")
        return v


class ChunkConfig(BaseModel):
    chunk_size: int = Field(default=512, description="Target chunk size in tokens", ge=64, le=8192)
    overlap: int = Field(default=64, description="Overlap in tokens between chunks", ge=0)
    strategy: ChunkStrategy = Field(default=ChunkStrategy.FIXED)


class EmbedderConfig(BaseModel):
    provider: EmbedderProvider = Field(default=EmbedderProvider.SENTENCE_TRANSFORMERS)
    model_name: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key (if using OpenAI)")
    batch_size: int = Field(default=64, ge=1, le=2048)


class StoreConfig(BaseModel):
    backend: StoreBackend = Field(default=StoreBackend.QDRANT)
    persist_dir: str = Field(default="./data/qdrant", description="Directory for persisting index")
    # Qdrant
    qdrant_url: Optional[str] = Field(default=None)
    qdrant_api_key: Optional[str] = Field(default=None)
    qdrant_collection: str = Field(default="git_rag")
    # Pinecone
    pinecone_api_key: Optional[str] = Field(default=None)
    pinecone_index: str = Field(default="git-rag")


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    repos_dir: str = Field(default="./data/repos", description="Base dir for cloned repos")


# ── Tool input models ────────────────────────────────────────────────────────


class AddRepoInput(BaseModel):
    """Input for kb_add_repo tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    url: str = Field(..., description="Git clone URL (e.g. 'https://github.com/user/repo.git')", min_length=1)
    branch: str = Field(default="main", description="Branch to track")
    file_globs: Optional[list[str]] = Field(default=None, description="Override default file glob patterns")
    exclude_patterns: Optional[list[str]] = Field(default=None, description="Override default exclude patterns")
    auto_sync: bool = Field(default=True, description="Immediately clone and ingest after adding")


class RemoveRepoInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    repo_id: str = Field(..., description="Repository ID to remove", min_length=1)
    delete_chunks: bool = Field(default=True, description="Also delete indexed chunks from the vector store")


class SyncInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    repo_id: Optional[str] = Field(default=None, description="Specific repo to sync (None = all)")
    force_full: bool = Field(default=False, description="Force full re-ingestion instead of incremental")


class QueryInput(BaseModel):
    """Input for pipeline_query tool."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(..., description="Natural language query", min_length=1, max_length=2000)
    top_k: int = Field(default=10, description="Number of results to return", ge=1, le=100)
    repo_ids: Optional[list[str]] = Field(default=None, description="Filter to specific repos (None = all)")
    file_types: Optional[list[str]] = Field(default=None, description="Filter by file extension (e.g. ['.py', '.md'])")
    rerank: bool = Field(default=False, description="Apply cross-encoder reranking")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class ConfigureInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    chunk_size: Optional[int] = Field(default=None, ge=64, le=8192)
    chunk_overlap: Optional[int] = Field(default=None, ge=0)
    embedder_provider: Optional[EmbedderProvider] = Field(default=None)
    embedder_model: Optional[str] = Field(default=None)
    store_backend: Optional[StoreBackend] = Field(default=None)


class StatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: Optional[str] = Field(default=None, description="Specific repo (None = overview)")


# ── Internal data models ─────────────────────────────────────────────────────


class Document(BaseModel):
    """A preprocessed document ready for chunking."""

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A text chunk with metadata."""

    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A single search result from the vector store."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
