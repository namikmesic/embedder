from __future__ import annotations

from config.models import EmbedderConfig
from domain.enums import EmbedderProvider
from embedding.base import Embedder


def create_embedder(config: EmbedderConfig) -> Embedder:
    if config.provider == EmbedderProvider.SENTENCE_TRANSFORMERS:
        from embedding.sentence_transformers import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            model_name=config.model_name,
            batch_size=config.batch_size,
        )
    elif config.provider == EmbedderProvider.OPENAI:
        from embedding.openai_embedder import OpenAIEmbedder

        if not config.openai_api_key:
            raise ValueError("OpenAI API key required when using OpenAI embedder. Set OPENAI_API_KEY env var.")
        return OpenAIEmbedder(
            model_name=config.model_name,
            api_key=config.openai_api_key,
            batch_size=config.batch_size,
        )
    else:
        raise ValueError(f"Unknown embedder provider: {config.provider}")
