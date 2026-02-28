from __future__ import annotations

from agents.base import LLMAgent
from agents.openai_agent import OpenAIAgent
from config.models import LLMConfig
from domain.enums import LLMProvider


def create_agent(config: LLMConfig) -> LLMAgent:
    if config.provider == LLMProvider.OPENAI:
        return OpenAIAgent(config)
    elif config.provider == LLMProvider.TRANSFORMERS:
        from agents.transformers_agent import TransformersAgent
        return TransformersAgent(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")
