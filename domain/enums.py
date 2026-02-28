from __future__ import annotations

from enum import Enum


class EmbedderProvider(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"


class StoreBackend(str, Enum):
    FAISS = "faiss"
    QDRANT = "qdrant"
    PINECONE = "pinecone"


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    TRANSFORMERS = "transformers"


class SourceType(str, Enum):
    GIT = "git"


class EntityKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    INTERFACE = "interface"
    CONSTANT = "constant"
    CONFIG = "config"
    DATA_MODEL = "data_model"
    PIPELINE = "pipeline"
    OTHER = "other"


class ConnectionKind(str, Enum):
    IMPORTS = "imports"
    INHERITS = "inherits"
    CALLS = "calls"
    INSTANTIATES = "instantiates"
    CONFIGURES = "configures"
    IMPLEMENTS = "implements"
    DEPENDS_ON = "depends_on"
    CONTAINS = "contains"


class BootstrapStatus(str, Enum):
    PENDING = "pending"
    SCANNING = "scanning"
    EXPLORING = "exploring"
    SYNTHESIZING = "synthesizing"
    GENERATING_DOCS = "generating_docs"
    EMBEDDING = "embedding"
    COMPLETE = "complete"
    ERROR = "error"
