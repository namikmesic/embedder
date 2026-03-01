# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vectordb-mcp is an **MCP server** that provides semantic knowledge storage and search. It exposes text embedding and vector search as MCP tools. The LLM client (Claude Desktop, Cursor, etc.) decides what to store and when to search — this server is pure infrastructure.

## Running

**Prerequisites**: PostgreSQL with the `pgvector` extension installed.

```bash
# Set up the database
psql -c "CREATE DATABASE vectordb;"
psql -d vectordb -c "CREATE EXTENSION IF NOT EXISTS vector;"

uv sync                        # Install all dependencies
uv run python server.py        # Run MCP server (stdio mode)
uv run mcp dev server.py       # Run with MCP inspector
```

Configure via `pipeline.yaml` (copy from `pipeline.yaml.example`) or environment variables.

**Key env vars**: `VECTORDB_CONFIG`, `VECTORDB_EMBEDDER_PROVIDER`, `VECTORDB_EMBEDDER_MODEL`, `OPENAI_API_KEY`, `VECTORDB_PG_DSN` (or `DATABASE_URL`), `VECTORDB_TABLE_NAME`, `VECTORDB_EF_SEARCH`, `VECTORDB_ITERATIVE_SCAN`, `VECTORDB_HYBRID_ALPHA`, `VECTORDB_POOL_MIN_SIZE`, `VECTORDB_POOL_MAX_SIZE`.

No tests exist yet.

## Architecture

**Entry point**: `server.py` — FastMCP server with 4 tools. Uses `VectorDBService` for orchestration and lazy initialization on first tool call. Clean shutdown via FastMCP lifespan.

### File structure

| File | Role |
|---|---|
| `server.py` | FastMCP server — 4 MCP tool wrappers, `VectorDBService` orchestration, lifespan shutdown |
| `config.py` | Enums, Pydantic config models, `load_config()` with data-driven env overrides |
| `embedding.py` | `Embedder` ABC, `SentenceTransformerEmbedder`, `OpenAIEmbedder`, `create_embedder()` |
| `store.py` | `SearchResult` model, `VectorStore` ABC, `PgVectorStore`, `create_store()` factory |

### Import DAG

`config` → `embedding` → `store` → `server`

### Design decisions

- **MCP server via FastMCP**: Tools are thin async wrappers that delegate to `VectorDBService`. Each returns a JSON string.
- **VectorDBService**: Encapsulates all business logic (token validation, embedding, store calls). Unit-testable without MCP. Single global instance behind an `asyncio.Lock`.
- **VectorStore ABC**: Abstract base class with methods `initialize`, `add`, `search`, `delete`, `delete_by_metadata`, `count`, `close`. `PgVectorStore` is the sole concrete implementation.
- **Lifespan shutdown**: FastMCP's `lifespan` context manager closes the asyncpg pool on server exit, preventing connection leaks.
- **PostgreSQL + pgvector**: HNSW index for cosine similarity. Schema auto-created on first connection. Configurable `ef_search`, iterative scan, and pool sizes via `SET LOCAL` in transactions.
- **Hybrid search**: Optional BM25 + vector blending via `hybrid_alpha` (0.0–1.0). Default `1.0` = pure vector.
- **Embedding versioning**: Chunks tagged with `model_version` (= `embedder.model_id`). Search filters by current model; switching models excludes old chunks until re-embedded.
- **Model naming convention**: `model` (config field — which model to load), `model_id` (Embedder ABC property — runtime identifier), `model_version` (DB column — version discriminator). Three layers, three names, no ambiguity.
- **Data-driven config**: Env overrides use a mapping table (`_ENV_MAP` in `config.py`).

### MCP Tools

| Tool | Description |
|------|-------------|
| `store_chunk(text, created_at, title?)` | Embed and store (rejects chunks over model token limit) |
| `search(query, top_k?, after?, before?)` | Semantic search; optional BM25 blending via `hybrid_alpha` |
| `delete_chunks(title)` | Delete all chunks matching a title |
| `count_chunks()` | Return total number of stored chunks |

### Chunking

The MCP client splits documents before calling `store_chunk`. Aim for 256–1024 tokens per chunk with ~10% overlap. Always pass a `title` so chunks can be grouped/deleted together.

### Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vectordb": {
      "command": "uv",
      "args": ["--directory", "/path/to/vectordb-mcp", "run", "server.py"],
      "env": {
        "VECTORDB_PG_DSN": "postgresql://localhost:5432/vectordb"
      }
    }
  }
}
```
