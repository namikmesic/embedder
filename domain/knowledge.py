from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from domain.enums import BootstrapStatus, ConnectionKind, EntityKind


class Entity(BaseModel):
    entity_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    kind: EntityKind
    file_path: str
    line_range: Optional[list[int]] = None
    description: str = ""
    signature: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class Connection(BaseModel):
    source_entity_id: str
    target_entity_id: str
    kind: ConnectionKind
    description: str = ""
    evidence_file: Optional[str] = None


class Unknown(BaseModel):
    description: str
    file_path: Optional[str] = None
    related_entity_ids: list[str] = Field(default_factory=list)
    priority: int = Field(default=3, ge=1, le=5)


class Finding(BaseModel):
    agent_scope: str
    files_examined: list[str] = Field(default_factory=list)
    tokens_consumed: int = 0
    entities: list[Entity] = Field(default_factory=list)
    connections: list[Connection] = Field(default_factory=list)
    unknowns: list[Unknown] = Field(default_factory=list)
    summary: str = ""


class BootstrapMap(BaseModel):
    repo_id: str
    repo_url: str
    created_at: datetime = Field(default_factory=datetime.now)
    status: BootstrapStatus = BootstrapStatus.PENDING
    entities: list[Entity] = Field(default_factory=list)
    connections: list[Connection] = Field(default_factory=list)
    unresolved_unknowns: list[Unknown] = Field(default_factory=list)
    agent_rounds: int = 0
    total_tokens_consumed: int = 0


class KnowledgeDoc(BaseModel):
    doc_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str
    content: str
    doc_type: str = "overview"
    source_entities: list[str] = Field(default_factory=list)
    repo_id: str = ""
    repo_url: str = ""
