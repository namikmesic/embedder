from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from domain.document import Document

logger = logging.getLogger(__name__)

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".sql": "sql",
    ".r": "r",
    ".R": "r",
    ".lua": "lua",
    ".zig": "zig",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
}


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3 :].lstrip("\n")
    return text


def _preprocess_markdown(text: str) -> str:
    text = _strip_frontmatter(text)
    return text.strip()


def _preprocess_code(text: str, language: str) -> str:
    lines = text.split("\n")
    cleaned = [line for line in lines if len(line) < 1000]
    return "\n".join(cleaned).strip()


def _preprocess_html(text: str) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]

        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()


def _preprocess_ipynb(text: str) -> str:
    import json as _json

    try:
        nb = _json.loads(text)
        parts: list[str] = []
        for cell in nb.get("cells", []):
            cell_type = cell.get("cell_type", "")
            source = "".join(cell.get("source", []))
            if cell_type == "markdown":
                parts.append(source)
            elif cell_type == "code":
                parts.append(f"```python\n{source}\n```")
        return "\n\n".join(parts).strip()
    except Exception:
        return text


class Preprocessor:
    def process_file(
        self,
        file_path: Path,
        content: str,
        repo_url: str,
        repo_id: str,
    ) -> Document | None:
        if not content.strip():
            return None

        ext = file_path.suffix.lower()
        language = LANGUAGE_MAP.get(ext, "unknown")

        metadata: dict[str, Any] = {
            "source_file": str(file_path.name),
            "source_path": str(file_path),
            "repo_url": repo_url,
            "repo_id": repo_id,
            "file_extension": ext,
            "language": language,
        }

        if ext in (".md", ".rst"):
            text = _preprocess_markdown(content)
        elif ext == ".html":
            text = _preprocess_html(content)
        elif ext == ".ipynb":
            text = _preprocess_ipynb(content)
        elif language != "unknown":
            text = _preprocess_code(content, language)
        elif ext == ".txt":
            text = content.strip()
        else:
            text = content.strip()

        if not text:
            return None

        metadata["char_count"] = len(text)
        return Document(text=text, metadata=metadata)
