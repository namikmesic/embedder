# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Git RAG is a Python CLI tool that turns Git repositories into queryable knowledge bases using RAG (Retrieval-Augmented Generation). Users register repos via CLI commands, and the tool handles cloning, chunking, embedding, and semantic search.

## Running

```bash
pip install -r requirements.txt
python cli.py --help
```

Configure via `pipeline.yaml` (copy from `pipeline.yaml.example`) or environment variables (`GIT_RAG_CONFIG`, `GIT_RAG_DATA_DIR`, `GIT_RAG_EMBEDDER_PROVIDER`, `GIT_RAG_EMBEDDER_MODEL`, `OPENAI_API_KEY`, `GIT_RAG_STORE_PERSIST_DIR`, `GIT_RAG_REPOS_DIR`). Env vars override the YAML file; defaults use local Sentence Transformers and Qdrant.

No tests exist yet.

## Architecture

**Entry point**: `cli.py` — Click CLI group with 7 subcommands. Initializes an `AppContext` dataclass holding all pipeline components (embedder, vector store, chunker, retriever, reranker, state manager).

**Data flow — Ingestion** (`ingest_repo` in ingestion.py):
`GitLoader` → `Preprocessor` → `Chunker` → `enrich_chunk_metadata` → `Embedder` → `VectorStore`

**Data flow — Query** (`query` command):
`Retriever` (embed query → store search with optional filtering) → optional `Reranker` (cross-encoder)

### Key modules

| File | Role |
|---|---|
| `cli.py` | Click CLI commands and AppContext initialization |
| `ingestion.py` | `ingest_repo()` — the core ingestion pipeline |
| `models.py` | All Pydantic models: config, tool inputs, internal data types (Document, Chunk, SearchResult) |
| `config.py` | YAML/JSON/env config loading with priority: env > file > defaults |
| `state.py` | `StateManager` — persists repo registry to `state.json`, tracks per-repo status (pending/ingesting/ready/error) and last ingested commit SHA |
| `pipeline/git_loader.py` | Shallow clone, pull, file discovery, incremental change detection via commit diff |
| `pipeline/preprocessor.py` | Converts files to plain text Documents; handles markdown, code (30+ extensions), HTML, Jupyter notebooks |
| `pipeline/chunker.py` | Token-based splitting using tiktoken `cl100k_base`; fixed-size with overlap |
| `pipeline/embedder.py` | Embedding abstraction — `SentenceTransformersEmbedder` (384-dim) or `OpenAIEmbedder` (1536-dim) |
| `pipeline/retriever.py` | Vector similarity search with optional metadata filtering |
| `pipeline/reranker.py` | Optional cross-encoder reranking; lazy-loads model on first use |
| `stores/qdrant_store.py` | Qdrant local embedded mode with `Distance.COSINE`; auto-persists to disk |
| `stores/faiss_store.py` | FAISS `IndexFlatIP` store with JSON sidecar for metadata (used when `backend: faiss`) |
| `stores/base.py` | Abstract `VectorStore` interface |

### Design decisions

- **Click CLI**: Decorator-based framework with `AppContext` dataclass for dependency injection via `@pass_app`.
- **`asyncio.run()` per command**: Bridges synchronous Click handlers with async store/embedder APIs.
- **Qdrant local embedded mode** (default): `QdrantClient(path=...)` — no external server. Deletion without index rebuilds, automatic disk persistence.
- **Token-based chunking**: Uses tiktoken rather than character counts so chunks respect LLM token budgets.
- **Incremental ingestion**: Tracks last commit SHA per repo; on sync, only re-processes changed/deleted files.
- **FAISS still available**: Configure `backend: faiss` to use the FAISS store instead. Uses `IndexFlatIP` with L2-normalized vectors and a JSON sidecar for metadata.

### CLI Commands

```
git-rag [-v/--verbose] <command>

  add <url>           Add a git repo as a knowledge base
  list                List all registered repos
  remove <repo_id>    Remove a repo and its chunks
  sync [repo_id]      Pull + re-ingest one or all repos
  status [repo_id]    Show stats and ingestion state
  query <query>       Semantic search across knowledge bases
  config              View or update pipeline settings
```

All commands support a `--json` flag for machine-readable output.

### Shared state pattern

All pipeline components live in the `AppContext` dataclass, initialized in the Click group callback and passed to subcommands via `@pass_app`. Store persistence is ensured via `ctx.call_on_close`.
