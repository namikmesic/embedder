from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class FileTokenInfo:
    path: Path
    relative_path: str
    tokens: int


@dataclass
class Partition:
    name: str
    description: str
    files: list[FileTokenInfo] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(f.tokens for f in self.files)


class ContextBudgetManager:
    OVERHEAD_TOKENS = 4000  # system prompt + output schema + response buffer

    def __init__(self, context_window: int, budget_fraction: float = 0.25):
        self._context_window = context_window
        self._budget_fraction = budget_fraction
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    @property
    def source_budget(self) -> int:
        return int((self._context_window - self.OVERHEAD_TOKENS) * self._budget_fraction)

    def compute_file_tokens(self, file_path: Path, content: str, root: Path) -> FileTokenInfo:
        tokens = len(self._tokenizer.encode(content, disallowed_special=()))
        return FileTokenInfo(path=file_path, relative_path=str(file_path.relative_to(root)), tokens=tokens)

    def split_partition_to_budget(self, partition: Partition) -> list[Partition]:
        budget = self.source_budget
        if partition.total_tokens <= budget:
            return [partition]

        result: list[Partition] = []
        current_files: list[FileTokenInfo] = []
        current_tokens = 0
        part_idx = 0

        for finfo in partition.files:
            if finfo.tokens > budget:
                # Flush current batch
                if current_files:
                    result.append(Partition(
                        name=f"{partition.name}_part{part_idx}",
                        description=f"{partition.description} (part {part_idx})",
                        files=current_files,
                    ))
                    current_files, current_tokens = [], 0
                    part_idx += 1
                # Solo partition for oversized file
                result.append(Partition(
                    name=f"{partition.name}_large_{finfo.relative_path.replace('/', '_')}",
                    description=f"{partition.description} — large file: {finfo.relative_path}",
                    files=[finfo],
                ))
                logger.warning("File %s (%d tokens) exceeds budget (%d), assigned solo", finfo.relative_path, finfo.tokens, budget)
                continue

            if current_tokens + finfo.tokens > budget and current_files:
                result.append(Partition(
                    name=f"{partition.name}_part{part_idx}",
                    description=f"{partition.description} (part {part_idx})",
                    files=current_files,
                ))
                current_files, current_tokens = [], 0
                part_idx += 1

            current_files.append(finfo)
            current_tokens += finfo.tokens

        if current_files:
            name = partition.name if not result else f"{partition.name}_part{part_idx}"
            desc = partition.description if not result else f"{partition.description} (part {part_idx})"
            result.append(Partition(name=name, description=desc, files=current_files))

        logger.info("Split '%s' (%d tokens) into %d parts (budget: %d)", partition.name, partition.total_tokens, len(result), budget)
        return result
