from __future__ import annotations

from pathlib import Path

from config.models import RepoConfig
from domain.enums import SourceType
from sources.git import GitSource


def create_source(config: RepoConfig, repos_dir: str):
    if config.source_type == SourceType.GIT:
        local_path = Path(config.local_path or (Path(repos_dir) / config.repo_id))
        return GitSource(config.url, local_path, config.branch)
    raise ValueError(f"Unsupported source type: {config.source_type}")
