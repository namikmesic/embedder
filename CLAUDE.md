# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Git RAG is a Python CLI tool that turns Git repositories into queryable knowledge bases using RAG (Retrieval-Augmented Generation). Users register repos via CLI commands, and the tool handles cloning, chunking, embedding, and semantic search.

**Bootstrap mode** (Phase 1): An LLM agent swarm analyzes source code, generates natural-language documentation, and embeds that documentation instead of raw code — producing far better semantic search results.

## Running

```bash
uv sync                        # Install all dependencies
uv run python cli.py --help    # Show all commands
```

Configure via `pipeline.yaml` (copy from `pipeline.yaml.example`) or environment variables. Env vars override the YAML file; defaults use local Sentence Transformers and Qdrant.

**Key env vars**: `GIT_RAG_CONFIG`, `GIT_RAG_DATA_DIR`, `GIT_RAG_EMBEDDER_PROVIDER`, `GIT_RAG_EMBEDDER_MODEL`, `OPENAI_API_KEY`, `GIT_RAG_STORE_PERSIST_DIR`, `GIT_RAG_REPOS_DIR`, `GIT_RAG_ORCHESTRATOR_API_KEY`, `GIT_RAG_ORCHESTRATOR_MODEL`, `GIT_RAG_SUB_AGENT_API_KEY`, `GIT_RAG_SUB_AGENT_MODEL`, `GIT_RAG_CONTEXT_BUDGET`.

No tests exist yet.

## Architecture

**Entry point**: `cli.py` — Click CLI group with 9 subcommands. Delegates app wiring to `app.py`.

**App wiring**: `app.py` — `AppContext` dataclass, `_init_app()`, `_run()` async bridge, `pass_app` decorator.

**Data flow — Ingestion** (`ingest_repo` in `ingest/pipeline.py`):
`Source.fetch()` → `Preprocessor` → `Chunker` → `_enrich_chunk_metadata` → `Embedder` → `VectorStore`

**Data flow — Bootstrap** (`bootstrap` command → `bootstrap/pipeline.py`):
`Source.fetch()` → Scanner agent → Explorer agents (concurrent) → Follow-up agents → Orchestrator synthesize → `KnowledgeDocs` → `Chunker` → `Embedder` → `VectorStore`

**Data flow — Query** (`query` command):
`Retriever` (embed query → store search with optional filtering) → optional `Reranker` (cross-encoder)

### Package structure

| Package/File | Role |
|---|---|
| `cli.py` | Click CLI commands + output formatting |
| `app.py` | `AppContext`, `_init_app()`, `_run()`, `pass_app` |
| `mcp_tools.py` | MCP tool input models: `AddRepoInput`, `QueryInput`, etc. |
| `state.py` | `RepoState` (Pydantic), `StateManager` — persists repo registry to `state.json` |
| `domain/enums.py` | All enums: `EmbedderProvider`, `StoreBackend`, `EntityKind`, etc. |
| `domain/document.py` | `Document`, `Chunk`, `SearchResult` |
| `domain/knowledge.py` | `Entity`, `Connection`, `Finding`, `BootstrapMap`, `KnowledgeDoc` |
| `config/models.py` | `RepoConfig`, `ChunkConfig`, `EmbedderConfig`, `StoreConfig`, `LLMConfig`, `AgentConfig`, `PipelineConfig` |
| `config/loader.py` | `load_config()` — data-driven env overrides via `_ENV_MAP` table |
| `sources/base.py` | `Source` ABC for content origins |
| `sources/git.py` | `GitSource` — all git operations inline (clone, pull, diff, file discovery) |
| `sources/factory.py` | `create_source()` — factory for Source implementations |
| `embedding/base.py` | `Embedder` ABC |
| `embedding/sentence_transformers.py` | `SentenceTransformerEmbedder` |
| `embedding/openai_embedder.py` | `OpenAIEmbedder` |
| `embedding/factory.py` | `create_embedder()` |
| `stores/base.py` | `VectorStore` ABC |
| `stores/qdrant.py` | Qdrant local embedded mode |
| `stores/faiss.py` | FAISS store with JSON sidecar |
| `stores/factory.py` | `create_store()` — async factory for store backends |
| `ingest/preprocessor.py` | Converts files to plain text Documents |
| `ingest/chunker.py` | Token-based splitting using tiktoken `cl100k_base` |
| `ingest/pipeline.py` | `ingest_repo()` — uses `Source` interface, inlines `_enrich_chunk_metadata()` |
| `search/retriever.py` | Vector similarity search with optional metadata filtering |
| `search/reranker.py` | Optional cross-encoder reranking |
| `agents/base.py` | `LLMAgent` ABC, `Message`, `CompletionResult`, `clean_json()` |
| `agents/openai_agent.py` | `OpenAIAgent` — OpenAI SDK (also works with Anthropic via `base_url`) |
| `agents/transformers_agent.py` | `TransformersAgent` — local HuggingFace models |
| `agents/factory.py` | `create_agent()` — factory for LLM agents |
| `agents/prompts.py` | All prompt templates for scanner, explorer, follow-up, synthesizer |
| `agents/budget.py` | `ContextBudgetManager` — enforces 25% context window rule |
| `agents/tasks.py` | `run_scanner()`, `run_explorer()`, `run_followup()` — sub-agent runners |
| `agents/orchestrator.py` | `BootstrapOrchestrator` — coordinates the full bootstrap pipeline |
| `bootstrap/pipeline.py` | `run_bootstrap()`, `load_bootstrap_map()`, `_docs_to_documents()` |

### Import conventions

Imports follow a strict DAG (no cycles):
- `domain/` — leaf layer, no project imports
- `config/` — imports only from `domain/`
- `sources/`, `embedding/`, `stores/` — import from `domain/` and `config/`
- `ingest/`, `search/` — import from `domain/`, `config/`, `embedding/`, `stores/`, `sources/`
- `agents/` — imports from `domain/`, `config/`, `sources/`
- `bootstrap/` — imports from `agents/`, `domain/`, `config/`, `embedding/`, `stores/`, `ingest/`, `sources/`
- `app.py` — imports from `config/`, `embedding/`, `stores/`, `ingest/`, `search/`, `state`
- `cli.py` — imports from `app`, `config/`, `ingest/`, `bootstrap/`

### Design decisions

- **Click CLI**: Decorator-based framework with `AppContext` dataclass for dependency injection via `@pass_app`.
- **`anyio.run()` per command**: Bridges synchronous Click handlers with async store/embedder/agent APIs. Uses anyio for structured concurrency.
- **Qdrant local embedded mode** (default): `QdrantClient(path=...)` — no external server.
- **Token-based chunking**: Uses tiktoken rather than character counts so chunks respect LLM token budgets.
- **Incremental ingestion**: Tracks last commit SHA per repo; on sync, only re-processes changed/deleted files.
- **Source abstraction**: `Source` ABC decouples both ingestion and bootstrap from git. `GitSource` has all git operations inline; future sources (files, web, email) implement the same interface.
- **Agent swarm**: Single orchestrator (strong model) dispatches concurrent sub-agents (cheap/fast). Context budget rule: each sub-agent uses max 25% of its context window for source material.
- **LLM integration**: OpenAI SDK (works with Anthropic via `base_url`, plus any OpenAI-compatible endpoint).
- **Data-driven config**: Env overrides use a mapping table (`_ENV_MAP` in `config/loader.py`) instead of individual if-statements.
- **RepoState as Pydantic model**: Uses `model_dump()`/`model_validate()` instead of hand-rolled serialization.
- **Store factory**: `stores/factory.py` encapsulates backend selection logic extracted from app initialization.

### CLI Commands

```
git-rag [-v/--verbose] <command>

  add <url>              Add a git repo as a knowledge base
  list                   List all registered repos
  remove <repo_id>       Remove a repo and its chunks
  sync [repo_id]         Pull + re-ingest one or all repos
  status [repo_id]       Show stats and ingestion state
  query <query>          Semantic search across knowledge bases
  config                 View or update pipeline settings
  bootstrap [repo_id]    Run LLM agent swarm to generate docs (--force to redo)
  map <repo_id>          View bootstrap knowledge map (--entities-only, --json)
```

All commands support a `--json` flag for machine-readable output.

### Shared state pattern

All pipeline components live in the `AppContext` dataclass (defined in `app.py`), initialized in the Click group callback and passed to subcommands via `@pass_app`. Store persistence is ensured via `ctx.call_on_close`.
