from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from domain.enums import EmbedderProvider, ResponseFormat, StoreBackend


class AddRepoInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    url: str = Field(..., min_length=1)
    branch: str = "main"
    file_globs: Optional[list[str]] = None
    exclude_patterns: Optional[list[str]] = None
    auto_sync: bool = True


class RemoveRepoInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    repo_id: str = Field(..., min_length=1)
    delete_chunks: bool = True


class SyncInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    repo_id: Optional[str] = None
    force_full: bool = False


class QueryInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)
    repo_ids: Optional[list[str]] = None
    file_types: Optional[list[str]] = None
    rerank: bool = False
    response_format: ResponseFormat = ResponseFormat.MARKDOWN


class ConfigureInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    chunk_size: Optional[int] = Field(default=None, ge=64, le=8192)
    chunk_overlap: Optional[int] = Field(default=None, ge=0)
    embedder_provider: Optional[EmbedderProvider] = None
    embedder_model: Optional[str] = None
    store_backend: Optional[StoreBackend] = None


class StatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: Optional[str] = None
