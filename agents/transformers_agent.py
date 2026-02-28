from __future__ import annotations

import logging
from typing import Optional

from agents.base import CompletionResult, LLMAgent, Message
from config.models import LLMConfig

logger = logging.getLogger(__name__)


class TransformersAgent(LLMAgent):

    def __init__(self, config: LLMConfig):
        self._config = config
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError("Install with: uv sync --extra local-llm")

        logger.info("Loading local model: %s", self._config.model_name)
        self._pipeline = hf_pipeline(
            "text-generation",
            model=self._config.model_name,
            device_map="auto",
        )

    async def complete(
        self,
        system_prompt: str,
        messages: list[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> CompletionResult:
        self._load_pipeline()

        chat = [{"role": "system", "content": system_prompt}]
        chat.extend({"role": m.role, "content": m.content} for m in messages)

        temp = temperature if temperature is not None else self._config.temperature

        output = self._pipeline(
            chat,
            max_new_tokens=max_tokens or self._config.max_tokens,
            temperature=temp if temp > 0 else None,
            do_sample=temp > 0,
            return_full_text=False,
        )

        generated_text = output[0]["generated_text"]
        if isinstance(generated_text, list):
            generated_text = generated_text[-1].get("content", str(generated_text[-1]))

        return CompletionResult(
            content=generated_text,
            tokens_used=self.estimate_tokens(generated_text),
            model=self._config.model_name,
        )

    @property
    def context_window(self) -> int:
        return self._config.context_window

    @property
    def model_name(self) -> str:
        return self._config.model_name
