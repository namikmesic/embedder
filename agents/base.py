from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import tiktoken


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class CompletionResult:
    content: str
    tokens_used: int = 0
    model: str = ""
    raw_response: Any = None


class LLMAgent(ABC):

    _tokenizer: Optional[tiktoken.Encoding] = None

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        messages: list[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> CompletionResult: ...

    def estimate_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            LLMAgent._tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(self._tokenizer.encode(text, disallowed_special=()))

    @property
    @abstractmethod
    def context_window(self) -> int: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...


def clean_json(text: str) -> str:
    """Strip markdown code fences from LLM JSON output."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
