-- =============================================================
-- setup_supabase.sql
-- Run this once in the Supabase SQL Editor before ingestion.
-- =============================================================

-- Step 1: Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 2: Create the main chunks table
-- voyage-3-lite produces 512-dimensional embeddings
CREATE TABLE IF NOT EXISTS docs_chunks (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    content     TEXT        NOT NULL,
    embedding   VECTOR(1024) NOT NULL,
    source_url  TEXT        NOT NULL,
    page_title  TEXT,
    chunk_index INTEGER,
    metadata    JSONB       DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Step 3: Create HNSW index for fast approximate nearest-neighbor search
-- HNSW is better than IVFFlat for < 1M rows and low-latency queries
CREATE INDEX IF NOT EXISTS docs_chunks_embedding_idx
ON docs_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Step 4: Unique constraint to prevent duplicate chunks on re-ingestion
ALTER TABLE docs_chunks
    DROP CONSTRAINT IF EXISTS docs_chunks_source_url_chunk_index_key;
ALTER TABLE docs_chunks
    ADD CONSTRAINT docs_chunks_source_url_chunk_index_key
    UNIQUE (source_url, chunk_index);

CREATE INDEX IF NOT EXISTS docs_chunks_source_url_idx
ON docs_chunks (source_url);

-- Step 5: Create the RPC function the agent will call
-- This is called from Python via: supabase.rpc("match_docs", {...})
CREATE OR REPLACE FUNCTION match_docs(
    query_embedding  VECTOR(1024),
    match_count      INT     DEFAULT 5,
    match_threshold  FLOAT   DEFAULT 0.70
)
RETURNS TABLE (
    id          UUID,
    content     TEXT,
    source_url  TEXT,
    page_title  TEXT,
    similarity  FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        docs_chunks.id,
        docs_chunks.content,
        docs_chunks.source_url,
        docs_chunks.page_title,
        -- cosine similarity = 1 - cosine distance
        1 - (docs_chunks.embedding <=> query_embedding) AS similarity
    FROM docs_chunks
    WHERE
        1 - (docs_chunks.embedding <=> query_embedding) > match_threshold
    ORDER BY
        docs_chunks.embedding <=> query_embedding  -- ascending distance = descending similarity
    LIMIT match_count;
END;
$$;

-- Step 6: Optional — helper to check row count after ingestion
-- SELECT COUNT(*) FROM docs_chunks;

-- Step 7: Optional — helper to wipe and re-ingest during development
-- TRUNCATE TABLE docs_chunks;

-- =============================================================
-- Expected output after ingestion:
--   SELECT COUNT(*) FROM docs_chunks;
--   Should return ~150-400 rows depending on crawl depth.
-- =============================================================
