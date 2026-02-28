from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Source(ABC):

    @abstractmethod
    async def fetch(self) -> str:
        """Fetch/sync content. Returns a state identifier (e.g., commit SHA)."""

    @abstractmethod
    def discover_files(self, globs: list[str], excludes: list[str]) -> list[Path]:
        """List indexable files matching globs. Returns absolute paths."""

    @abstractmethod
    def get_file_content(self, path: Path) -> str: ...

    @abstractmethod
    def get_changed_files(self, since: str) -> list[str]:
        """Files changed since checkpoint. Returns relative paths."""

    @abstractmethod
    def get_deleted_files(self, since: str) -> list[str]:
        """Files deleted since checkpoint. Returns relative paths."""

    @abstractmethod
    def get_file_tree(self, globs: list[str], excludes: list[str]) -> str:
        """Human-readable directory tree for the scanner agent."""

    @property
    @abstractmethod
    def local_path(self) -> Path: ...

    @abstractmethod
    def get_info(self) -> dict: ...
