from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import Optional

from git import Repo, InvalidGitRepositoryError, GitCommandError  # type: ignore[import-untyped]

from sources.base import Source

logger = logging.getLogger(__name__)


class GitSource(Source):

    def __init__(self, repo_url: str, local_path: Path, branch: str = "main"):
        self._repo_url = repo_url
        self._local_path = Path(local_path)
        self._branch = branch
        self._repo: Optional[Repo] = None

    @property
    def _git_repo(self) -> Repo:
        if self._repo is None:
            raise RuntimeError("Repository not initialized. Call fetch() first.")
        return self._repo

    async def fetch(self) -> str:
        """Clone or pull. Returns HEAD commit SHA."""
        if self._local_path.exists() and (self._local_path / ".git").exists():
            try:
                self._repo = Repo(self._local_path)
                origin = self._repo.remotes.origin
                origin.pull(self._branch)
                logger.info("Pulled latest for %s (branch: %s)", self._repo_url, self._branch)
            except (InvalidGitRepositoryError, GitCommandError) as e:
                logger.error("Failed to pull %s: %s", self._repo_url, e)
                raise
        else:
            self._local_path.mkdir(parents=True, exist_ok=True)
            try:
                self._repo = Repo.clone_from(
                    self._repo_url,
                    self._local_path,
                    branch=self._branch,
                    depth=1,
                )
                logger.info("Cloned %s to %s", self._repo_url, self._local_path)
            except GitCommandError as e:
                logger.error("Failed to clone %s: %s", self._repo_url, e)
                raise

        return self._git_repo.head.commit.hexsha

    def discover_files(self, globs: list[str], excludes: list[str]) -> list[Path]:
        matched: list[Path] = []
        for glob_pattern in globs:
            for path in self._local_path.glob(glob_pattern):
                if not path.is_file():
                    continue
                rel = str(path.relative_to(self._local_path))
                if any(fnmatch.fnmatch(rel, exc) for exc in excludes):
                    continue
                matched.append(path)
        return sorted(set(matched))

    def get_file_content(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
            return ""

    def get_changed_files(self, since: str) -> list[str]:
        try:
            old_commit = self._git_repo.commit(since)
            new_commit = self._git_repo.head.commit

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
            logger.warning("Could not compute diff from %s: %s. Will do full re-index.", since, e)
            return []

    def get_deleted_files(self, since: str) -> list[str]:
        try:
            old_commit = self._git_repo.commit(since)
            new_commit = self._git_repo.head.commit
            diff = old_commit.diff(new_commit)
            return [d.a_path for d in diff if d.deleted_file and d.a_path]
        except Exception:
            return []

    def get_file_tree(self, globs: list[str], excludes: list[str]) -> str:
        files = self.discover_files(globs, excludes)
        if not files:
            return "(no files found)"

        lines: list[str] = []
        root = self._local_path
        rel_paths = sorted(str(f.relative_to(root)) for f in files)

        prev_parts: list[str] = []
        for rel in rel_paths:
            parts = rel.split("/")
            common = 0
            for i, (a, b) in enumerate(zip(prev_parts, parts)):
                if a == b:
                    common = i + 1
                else:
                    break

            for i in range(common, len(parts) - 1):
                lines.append(f"{'  ' * i}{parts[i]}/")

            lines.append(f"{'  ' * (len(parts) - 1)}{parts[-1]}")
            prev_parts = parts

        return "\n".join(lines)

    @property
    def local_path(self) -> Path:
        return self._local_path

    def get_info(self) -> dict:
        info: dict = {"source_type": "git", "url": self._repo_url, "branch": self._branch}
        try:
            repo = self._git_repo
            info.update({
                "local_path": str(self._local_path),
                "head_sha": repo.head.commit.hexsha,
                "head_message": repo.head.commit.message.strip(),
                "head_author": str(repo.head.commit.author),
                "head_date": repo.head.commit.committed_datetime.isoformat(),
            })
        except RuntimeError:
            pass  # Not yet cloned
        return info
