# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vectordb-mcp is an **MCP server** that provides multi-tenant semantic knowledge storage and search. It exposes text embedding and vector search as MCP tools. The LLM client (Claude Desktop, Cursor, etc.) decides what to store and when to search — this server is pure infrastructure.

**Architecture**: PostgreSQL is a pure search index (halfvec/vector embeddings + tsvector), MinIO stores content blobs. Multi-tenancy enforced via RLS. UUIDv7 provides temporal ordering. PG18 with pgvector. Each **knowledgebase** gets its own PG table with tailored embedding model, vector precision, distance metric, and HNSW params.

## Running

**Prerequisites**: Docker (for PostgreSQL 18 + pgvector, MinIO).

```bash
docker compose up -d             # Start PG, MinIO
uv sync                          # Install all dependencies
uv run python server.py          # Run MCP server (stdio mode)
uv run mcp dev server.py         # Run with MCP inspector
```

Configure via environment variables.

**Key env vars**:
- **Embedder**: `VECTORDB_EMBEDDER_PROVIDER`, `VECTORDB_EMBEDDER_MODEL`, `OPENAI_API_KEY`
- **PostgreSQL**: `VECTORDB_PG_DSN` (or `DATABASE_URL`), `VECTORDB_POOL_MIN_SIZE`, `VECTORDB_POOL_MAX_SIZE`
- **KB defaults** (applied to new KBs): `VECTORDB_EF_SEARCH`, `VECTORDB_ITERATIVE_SCAN`, `VECTORDB_HYBRID_ALPHA`, `VECTORDB_CANDIDATE_MULTIPLIER`, `VECTORDB_HNSW_M`, `VECTORDB_HNSW_EF_CONSTRUCTION`, `VECTORDB_TSV_LANGUAGE`
- **MinIO**: `VECTORDB_MINIO_ENDPOINT`, `VECTORDB_MINIO_ACCESS_KEY`, `VECTORDB_MINIO_SECRET_KEY`, `VECTORDB_MINIO_BUCKET`, `VECTORDB_MINIO_SECURE`

**Testing** (requires Docker services running):

```bash
uv sync --extra test              # Install pytest + pytest-asyncio
uv run pytest -v                  # Run integration tests
```

Tests use real PG + MinIO — no mocks. Each test gets a unique `tenant_id`; RLS provides isolation.

## Architecture

**Entry point**: `server.py` — FastMCP server with 6 tools. Uses `VectorDBService` for orchestration via `KnowledgebaseManager`. Lazy initialization on first tool call. Clean shutdown via FastMCP lifespan.

### File structure

| File | Role |
|---|---|
| `server.py` | FastMCP server — 6 MCP tool wrappers, `VectorDBService` orchestration via `KnowledgebaseManager`, lifespan shutdown |
| `config.py` | Enums (`EmbedderProvider`, `DistanceMetric`, `VectorPrecision`), Pydantic config models (`EmbedderConfig`, `StoreConfig`, `ObjectStoreConfig`, `KnowledgebaseDefaults`), `load_config()` |
| `embedding.py` | `Embedder` ABC, `SentenceTransformerEmbedder`, `OpenAIEmbedder`, `create_embedder()`, `EmbedderCache` |
| `store.py` | `SearchResult` model, `PgVectorStore` — parameterized per-KB (table, vec type, distance op, tsv language), RLS, UUIDv7, upsert with versioning |
| `object_store.py` | `ChunkContent` dataclass, `ObjectStore` — async MinIO wrapper with `kb_id`-aware key paths |
| `knowledgebase.py` | `KnowledgebaseRecord`, `KnowledgebaseManager` (registry, DDL builder, KB cache, embedder cache), operator lookups |
| `init.sql` | Schema DDL — vector extension, chunks table, indexes, RLS policy, knowledgebases registry table + default seed |
| `docker-compose.yml` | PG (pgvector/pg18), MinIO services |

### Import DAG

`config` → `{embedding, object_store, store}` → `knowledgebase` → `server`

### Design decisions

- **Knowledgebase concept**: A named collection with its own PG table, embedding model, vector precision, distance metric, HNSW params, and tsvector language. Table-per-collection is the dominant pattern for pgvector multi-model deployments.
- **KnowledgebaseManager**: Central manager holding shared asyncpg pool, ObjectStore, and EmbedderCache. Builds per-KB `PgVectorStore` instances on demand and caches `(record, store, embedder)` triples by KB name.
- **EmbedderCache**: Deduplicates embedder instances by `(provider, model)` key. Two KBs using the same model share one embedder.
- **MCP server via FastMCP**: Tools are thin async wrappers that delegate to `VectorDBService`. Each returns a JSON string.
- **VectorDBService**: Encapsulates all business logic (token validation, embedding, content hashing). Takes a `KnowledgebaseManager`; `create()` classmethod builds from env config. Single global instance behind an `asyncio.Lock`.
- **Two-store separation**: PG = search index (embeddings + tsvector), MinIO = content storage (JSON blobs). PG stores no text content.
- **PgVectorStore**: Parameterized concrete class — takes a pre-built pool and per-KB settings (table, vec_type, distance_op, tsv_language, ef_search, etc.). No lifecycle management (pool owned by manager).
- **Multi-tenancy**: Row-Level Security (RLS) on each KB table. Every query sets `app.tenant_id` via `SET LOCAL` in a transaction. Belt-and-suspenders: explicit `tenant_id = $N` filter in all queries.
- **UUIDv7**: Content date embedded in the UUID via `uuid_utils.uuid7(timestamp=ms)`. Time filtering converts datetimes to UUIDv7 boundaries — no `created_at` column needed.
- **Content versioning**: `SELECT ... FOR UPDATE` + content_hash comparison. If hash matches, no-op. If different, old row gets `latest=false`, new row inserted with `latest=true`. Partial unique index (`WHERE latest`) prevents double-latest race conditions.
- **Old MinIO objects kept**: On version update, old content remains in MinIO for audit. Only PG row gets `latest=false`.
- **Lifespan shutdown**: FastMCP's `lifespan` context manager closes embedder cache, MinIO client, and asyncpg pool on server exit.
- **Hybrid search**: Optional BM25 + vector blending via `hybrid_alpha` (0.0–1.0, per-KB). Default `1.0` = pure vector. CTE approach with per-KB ops class.
- **Data-driven config**: Env vars are the sole config source (no YAML/JSON files). Uses a mapping table (`_ENV_MAP` in `config.py`) to apply env overrides to Pydantic defaults.
- **DDL bootstrapping**: `build_table_ddl()` in `knowledgebase.py` generates complete DDL for a new KB table: CREATE TABLE, HNSW + GIN + B-tree indexes, RLS policy, autovacuum tuning. Executed in a transaction with the registry INSERT.

### MCP Tools

| Tool | Description |
|------|-------------|
| `store_chunk(text, source_id, tenant_id, content_date, title?, knowledgebase?)` | Embed and store; UUIDv7 chunk ID, content hashing, idempotent upsert |
| `search(query, tenant_id, top_k?, after?, before?, knowledgebase?)` | Semantic search; reads content from MinIO; optional BM25 blending |
| `delete_chunks(source_id, tenant_id, knowledgebase?)` | Delete all chunks matching a source_id; cleans PG and MinIO |
| `count_chunks(tenant_id, knowledgebase?)` | Return total number of stored chunks for a tenant |
| `create_knowledgebase(name, embedder_model?, ...)` | Create a new KB with its own table and embedding model |
| `list_knowledgebases()` | List all KBs and their configuration |

### Chunking

The MCP client splits documents before calling `store_chunk`. Aim for 256–1024 tokens per chunk with ~10% overlap. Always pass a `source_id` to identify the document and a `title` for display.

### Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vectordb": {
      "command": "uv",
      "args": ["--directory", "/path/to/vectordb-mcp", "run", "server.py"],
      "env": {
        "VECTORDB_PG_DSN": "postgresql://localhost:5432/vectordb",
        "VECTORDB_MINIO_ENDPOINT": "localhost:9000"
      }
    }
  }
}
```
