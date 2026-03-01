-- vectordb-mcp schema initialization
-- Requires: pgvector extension

CREATE EXTENSION IF NOT EXISTS vector;

-- Main chunks table (regular, non-partitioned)
CREATE TABLE IF NOT EXISTS chunks (
    id            UUID        NOT NULL PRIMARY KEY,
    tenant_id     UUID        NOT NULL,
    source_id     TEXT        NOT NULL,
    embedding     halfvec(768) NOT NULL,
    content_hash  BYTEA       NOT NULL,
    tsv           tsvector    NOT NULL,
    latest        BOOLEAN     NOT NULL DEFAULT true
);

-- HNSW vector index (set maintenance_work_mem for build performance)
SET maintenance_work_mem = '512MB';
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw
    ON chunks USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE latest;

CREATE INDEX IF NOT EXISTS idx_chunks_tenant
    ON chunks (tenant_id) WHERE latest;

CREATE INDEX IF NOT EXISTS idx_chunks_tsv
    ON chunks USING gin (tsv) WHERE latest;

CREATE INDEX IF NOT EXISTS idx_chunks_tenant_source
    ON chunks (tenant_id, source_id) WHERE latest;

CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_tenant_source_uniq
    ON chunks (tenant_id, source_id) WHERE latest;

CREATE INDEX IF NOT EXISTS idx_chunks_hash
    ON chunks (content_hash);

-- Row-Level Security — enforces tenant isolation at the database level
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE tablename = 'chunks' AND policyname = 'tenant_isolation'
    ) THEN
        CREATE POLICY tenant_isolation ON chunks
            USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
            WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);
    END IF;
END
$$;

-- Autovacuum tuning for high-write vector tables
ALTER TABLE chunks SET (autovacuum_vacuum_scale_factor = 0.01);

-- Knowledgebase registry — each row describes a KB with its own PG table
CREATE TABLE IF NOT EXISTS knowledgebases (
    id                   UUID PRIMARY KEY,
    name                 TEXT NOT NULL UNIQUE,
    table_name           TEXT NOT NULL UNIQUE,
    embedder_provider    TEXT NOT NULL,
    embedder_model       TEXT NOT NULL,
    dimension            INT  NOT NULL,
    precision            TEXT NOT NULL DEFAULT 'halfvec',
    distance_metric      TEXT NOT NULL DEFAULT 'cosine',
    hnsw_m               INT  NOT NULL DEFAULT 16,
    hnsw_ef_construction INT  NOT NULL DEFAULT 64,
    tsv_language         TEXT NOT NULL DEFAULT 'english',
    ef_search            INT  NOT NULL DEFAULT 40,
    iterative_scan       BOOLEAN NOT NULL DEFAULT false,
    hybrid_alpha         REAL NOT NULL DEFAULT 1.0,
    candidate_multiplier INT  NOT NULL DEFAULT 5,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Seed the default KB pointing at the existing chunks table
INSERT INTO knowledgebases (id, name, table_name, embedder_provider, embedder_model, dimension)
VALUES ('00000000-0000-7000-8000-000000000000', 'default', 'chunks',
        'sentence_transformers', 'BAAI/bge-base-en-v1.5', 768)
ON CONFLICT (name) DO NOTHING;
