from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None  # type: ignore[assignment]

from config.models import PipelineConfig
from domain.enums import EmbedderProvider, LLMProvider, StoreBackend

CONFIG_FILE_ENV = "GIT_RAG_CONFIG"
DEFAULT_CONFIG_PATH = Path("./pipeline.yaml")
DATA_DIR = Path(os.environ.get("GIT_RAG_DATA_DIR", "./data"))

# Data-driven env-var override table: (env_var, dotted_path, type_converter)
_ENV_MAP: list[tuple[str, str, type]] = [
    ("GIT_RAG_EMBEDDER_PROVIDER", "embedder.provider", EmbedderProvider),
    ("GIT_RAG_EMBEDDER_MODEL", "embedder.model_name", str),
    ("OPENAI_API_KEY", "embedder.openai_api_key", str),
    ("GIT_RAG_STORE_BACKEND", "store.backend", StoreBackend),
    ("GIT_RAG_STORE_PERSIST_DIR", "store.persist_dir", str),
    ("GIT_RAG_REPOS_DIR", "repos_dir", str),
    ("GIT_RAG_ORCHESTRATOR_MODEL", "agents.orchestrator.model_name", str),
    ("GIT_RAG_ORCHESTRATOR_API_KEY", "agents.orchestrator.api_key", str),
    ("GIT_RAG_ORCHESTRATOR_BASE_URL", "agents.orchestrator.base_url", str),
    ("GIT_RAG_ORCHESTRATOR_PROVIDER", "agents.orchestrator.provider", LLMProvider),
    ("GIT_RAG_SUB_AGENT_MODEL", "agents.sub_agent.model_name", str),
    ("GIT_RAG_SUB_AGENT_API_KEY", "agents.sub_agent.api_key", str),
    ("GIT_RAG_SUB_AGENT_BASE_URL", "agents.sub_agent.base_url", str),
    ("GIT_RAG_SUB_AGENT_PROVIDER", "agents.sub_agent.provider", LLMProvider),
    ("GIT_RAG_CONTEXT_BUDGET", "agents.context_budget_fraction", float),
    ("GIT_RAG_MAX_CONCURRENT_AGENTS", "agents.max_concurrent_agents", int),
]


def _apply_env_overrides(config: PipelineConfig) -> None:
    for env_var, dotted_path, converter in _ENV_MAP:
        value = os.environ.get(env_var)
        if not value:
            continue

        parts = dotted_path.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], converter(value))


def load_config(config_path: Optional[Path] = None) -> PipelineConfig:
    """Priority: env vars > YAML file > defaults."""
    config = PipelineConfig()

    path = config_path or (Path(os.environ[CONFIG_FILE_ENV]) if CONFIG_FILE_ENV in os.environ else DEFAULT_CONFIG_PATH)
    if path.exists() and yaml is not None:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        config = PipelineConfig(**raw)
    elif path.exists() and path.suffix == ".json":
        with open(path) as f:
            raw = json.load(f)
        config = PipelineConfig(**raw)

    _apply_env_overrides(config)

    Path(config.repos_dir).mkdir(parents=True, exist_ok=True)
    Path(config.store.persist_dir).mkdir(parents=True, exist_ok=True)

    return config
