"""State management: repo registry and ingestion tracking."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from models import RepoConfig

logger = logging.getLogger(__name__)


class RepoState:
    """Tracks the state of a single registered repository."""

    def __init__(
        self,
        config: RepoConfig,
        last_ingested_commit: Optional[str] = None,
        last_ingested_at: Optional[str] = None,
        chunk_count: int = 0,
        status: str = "pending",
        error: Optional[str] = None,
    ):
        self.config = config
        self.last_ingested_commit = last_ingested_commit
        self.last_ingested_at = last_ingested_at
        self.chunk_count = chunk_count
        self.status = status  # pending | ingesting | ready | error
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.model_dump(),
            "last_ingested_commit": self.last_ingested_commit,
            "last_ingested_at": self.last_ingested_at,
            "chunk_count": self.chunk_count,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepoState:
        return cls(
            config=RepoConfig(**data["config"]),
            last_ingested_commit=data.get("last_ingested_commit"),
            last_ingested_at=data.get("last_ingested_at"),
            chunk_count=data.get("chunk_count", 0),
            status=data.get("status", "pending"),
            error=data.get("error"),
        )


class StateManager:
    """Manages the persistent state of all registered repos."""

    def __init__(self, state_path: str = "./data/state.json"):
        self._path = Path(state_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._repos: dict[str, RepoState] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                for repo_id, repo_data in data.get("repos", {}).items():
                    self._repos[repo_id] = RepoState.from_dict(repo_data)
                logger.info("Loaded state: %d repos", len(self._repos))
            except Exception as e:
                logger.warning("Failed to load state from %s: %s", self._path, e)

    def save(self) -> None:
        data = {"repos": {rid: rs.to_dict() for rid, rs in self._repos.items()}}
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def add_repo(self, config: RepoConfig) -> RepoState:
        state = RepoState(config=config)
        self._repos[config.repo_id] = state
        self.save()
        return state

    def remove_repo(self, repo_id: str) -> bool:
        if repo_id in self._repos:
            del self._repos[repo_id]
            self.save()
            return True
        return False

    def get_repo(self, repo_id: str) -> Optional[RepoState]:
        return self._repos.get(repo_id)

    def list_repos(self) -> list[RepoState]:
        return list(self._repos.values())

    def update_ingestion(
        self,
        repo_id: str,
        commit_sha: str,
        chunk_count: int,
        status: str = "ready",
        error: Optional[str] = None,
    ) -> None:
        state = self._repos.get(repo_id)
        if state:
            state.last_ingested_commit = commit_sha
            state.last_ingested_at = datetime.now(timezone.utc).isoformat()
            state.chunk_count = chunk_count
            state.status = status
            state.error = error
            self.save()

    def set_status(self, repo_id: str, status: str, error: Optional[str] = None) -> None:
        state = self._repos.get(repo_id)
        if state:
            state.status = status
            state.error = error
            self.save()
