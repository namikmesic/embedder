"""Git operations: clone, pull, diff detection, file discovery."""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import Optional

from git import Repo, InvalidGitRepositoryError, GitCommandError  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class GitLoader:
    """Manages git operations for a single repository knowledge base."""

    def __init__(self, repo_url: str, local_path: Path, branch: str = "main"):
        self.repo_url = repo_url
        self.local_path = Path(local_path)
        self.branch = branch
        self._repo: Optional[Repo] = None

    @property
    def repo(self) -> Repo:
        if self._repo is None:
            raise RuntimeError("Repository not initialized. Call clone_or_pull() first.")
        return self._repo

    def clone_or_pull(self) -> str:
        """Clone the repo if it doesn't exist locally, otherwise pull latest.

        Returns the HEAD commit SHA after the operation.
        """
        if self.local_path.exists() and (self.local_path / ".git").exists():
            try:
                self._repo = Repo(self.local_path)
                origin = self._repo.remotes.origin
                origin.pull(self.branch)
                logger.info("Pulled latest for %s (branch: %s)", self.repo_url, self.branch)
            except (InvalidGitRepositoryError, GitCommandError) as e:
                logger.error("Failed to pull %s: %s", self.repo_url, e)
                raise
        else:
            self.local_path.mkdir(parents=True, exist_ok=True)
            try:
                self._repo = Repo.clone_from(
                    self.repo_url,
                    self.local_path,
                    branch=self.branch,
                    depth=1,  # shallow clone for speed
                )
                logger.info("Cloned %s to %s", self.repo_url, self.local_path)
            except GitCommandError as e:
                logger.error("Failed to clone %s: %s", self.repo_url, e)
                raise

        return self.repo.head.commit.hexsha

    def get_changed_files(self, since_commit: str) -> list[str]:
        """Get list of files changed since a given commit SHA.

        Returns relative file paths that were added, modified, or renamed.
        """
        try:
            old_commit = self.repo.commit(since_commit)
            new_commit = self.repo.head.commit

            if old_commit.hexsha == new_commit.hexsha:
                return []

            diff = old_commit.diff(new_commit)
            changed: list[str] = []
            for d in diff:
                if d.a_path:
                    changed.append(d.a_path)
                if d.b_path and d.b_path != d.a_path:
                    changed.append(d.b_path)
            return list(set(changed))
        except Exception as e:
            logger.warning("Could not compute diff from %s: %s. Will do full re-index.", since_commit, e)
            return []

    def get_deleted_files(self, since_commit: str) -> list[str]:
        """Get files that were deleted since a given commit."""
        try:
            old_commit = self.repo.commit(since_commit)
            new_commit = self.repo.head.commit
            diff = old_commit.diff(new_commit)
            return [d.a_path for d in diff if d.deleted_file and d.a_path]
        except Exception:
            return []

    def discover_files(
        self,
        file_globs: list[str],
        exclude_patterns: list[str],
    ) -> list[Path]:
        """Find all indexable files matching globs, excluding patterns.

        Returns absolute paths.
        """
        matched: list[Path] = []
        for glob_pattern in file_globs:
            for path in self.local_path.glob(glob_pattern):
                if not path.is_file():
                    continue
                rel = str(path.relative_to(self.local_path))
                if any(fnmatch.fnmatch(rel, exc) for exc in exclude_patterns):
                    continue
                matched.append(path)
        return sorted(set(matched))

    def get_file_content(self, file_path: Path) -> str:
        """Read file content as text. Returns empty string on decode errors."""
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Could not read %s: %s", file_path, e)
            return ""

    def get_head_sha(self) -> str:
        return self.repo.head.commit.hexsha

    def get_repo_info(self) -> dict:
        """Return metadata about the repo."""
        return {
            "url": self.repo_url,
            "branch": self.branch,
            "local_path": str(self.local_path),
            "head_sha": self.repo.head.commit.hexsha,
            "head_message": self.repo.head.commit.message.strip(),
            "head_author": str(self.repo.head.commit.author),
            "head_date": self.repo.head.commit.committed_datetime.isoformat(),
        }
