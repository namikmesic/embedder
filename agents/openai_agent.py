from __future__ import annotations

import logging
from typing import Optional

from openai import AsyncOpenAI

from agents.base import CompletionResult, LLMAgent, Message
from config.models import LLMConfig

logger = logging.getLogger(__name__)


class OpenAIAgent(LLMAgent):

    def __init__(self, config: LLMConfig):
        self._config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key or "not-set",
            base_url=config.base_url,
        )

    async def complete(
        self,
        system_prompt: str,
        messages: list[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> CompletionResult:
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend({"role": m.role, "content": m.content} for m in messages)

        kwargs: dict = {
            "model": self._config.model_name,
            "messages": api_messages,
            "temperature": temperature if temperature is not None else self._config.temperature,
            "max_tokens": max_tokens or self._config.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        logger.debug("OpenAI request: model=%s, messages=%d", self._config.model_name, len(api_messages))
        response = await self._client.chat.completions.create(**kwargs)

        usage = response.usage
        tokens_used = usage.total_tokens if usage else 0
        logger.debug("OpenAI response: tokens=%d", tokens_used)

        return CompletionResult(
            content=response.choices[0].message.content or "",
            tokens_used=tokens_used,
            model=response.model,
            raw_response=response,
        )

    @property
    def context_window(self) -> int:
        return self._config.context_window

    @property
    def model_name(self) -> str:
        return self._config.model_name
