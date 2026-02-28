from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from domain.enums import (
    ChunkStrategy,
    EmbedderProvider,
    LLMProvider,
    SourceType,
    StoreBackend,
)


class RepoConfig(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    repo_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    url: str = Field(...)
    branch: str = "main"
    source_type: SourceType = SourceType.GIT
    local_path: Optional[str] = None
    file_globs: list[str] = Field(
        default_factory=lambda: ["**/*.md", "**/*.txt", "**/*.py", "**/*.js", "**/*.ts", "**/*.rs", "**/*.go", "**/*.java", "**/*.rst"],
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["**/node_modules/**", "**/.git/**", "**/venv/**", "**/__pycache__/**", "**/dist/**", "**/build/**"],
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not (v.startswith("http") or v.startswith("git@") or v.startswith("/")):
            raise ValueError("URL must be an HTTPS URL, SSH URL, or local path")
        return v


class ChunkConfig(BaseModel):
    chunk_size: int = Field(default=512, ge=64, le=8192)
    overlap: int = Field(default=64, ge=0)
    strategy: ChunkStrategy = ChunkStrategy.FIXED


class EmbedderConfig(BaseModel):
    provider: EmbedderProvider = EmbedderProvider.SENTENCE_TRANSFORMERS
    model_name: str = "all-MiniLM-L6-v2"
    openai_api_key: Optional[str] = None
    batch_size: int = Field(default=64, ge=1, le=2048)


class StoreConfig(BaseModel):
    backend: StoreBackend = StoreBackend.QDRANT
    persist_dir: str = "./data/qdrant"
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "git_rag"
    pinecone_api_key: Optional[str] = None
    pinecone_index: str = "git-rag"


class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = Field(default=4096, ge=256, le=128000)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    context_window: int = 128000


class AgentConfig(BaseModel):
    orchestrator: LLMConfig = Field(default_factory=lambda: LLMConfig(model_name="gpt-4o"))
    sub_agent: LLMConfig = Field(default_factory=lambda: LLMConfig(model_name="gpt-4o-mini"))
    context_budget_fraction: float = Field(default=0.25, ge=0.05, le=0.9)
    max_concurrent_agents: int = Field(default=4, ge=1, le=20)
    max_followup_rounds: int = Field(default=2, ge=0, le=10)


class PipelineConfig(BaseModel):
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    repos_dir: str = "./data/repos"
    agents: AgentConfig = Field(default_factory=AgentConfig)
