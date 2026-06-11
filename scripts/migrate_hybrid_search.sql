-- =============================================================
-- migrate_hybrid_search.sql
-- Run this once in the Supabase SQL Editor AFTER setup_supabase.sql.
-- Adds: full-text search column + index, hybrid RRF search RPC,
--       content hashing for idempotent ingestion, chat persistence
--       tables, and fixes match_docs to return chunk_index.
-- After running, re-run ingestion: uv run python scripts/ingest.py --force-crawl
-- =============================================================

-- -------------------------------------------------------------
-- Step 1: Content hash column for idempotent ingestion.
-- Lets the pipeline skip re-embedding unchanged chunks.
-- -------------------------------------------------------------
ALTER TABLE docs_chunks
    ADD COLUMN IF NOT EXISTS content_hash TEXT;

-- -------------------------------------------------------------
-- Step 2: Generated tsvector column + GIN index for full-text search.
-- -------------------------------------------------------------
ALTER TABLE docs_chunks
    ADD COLUMN IF NOT EXISTS fts TSVECTOR
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS docs_chunks_fts_idx
ON docs_chunks
USING gin (fts);

-- -------------------------------------------------------------
-- Step 3: Ensure match_docs returns chunk_index (the Python dedup
-- key needs it). Identical to the definition in setup_supabase.sql;
-- kept here so older installs created before that fix are upgraded.
-- -------------------------------------------------------------
DROP FUNCTION IF EXISTS match_docs(VECTOR(1024), INT, FLOAT);

CREATE OR REPLACE FUNCTION match_docs(
    query_embedding  VECTOR(1024),
    match_count      INT     DEFAULT 6,
    match_threshold  FLOAT   DEFAULT 0.30
)
RETURNS TABLE (
    id          UUID,
    content     TEXT,
    source_url  TEXT,
    page_title  TEXT,
    chunk_index INTEGER,
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
        docs_chunks.chunk_index,
        1 - (docs_chunks.embedding <=> query_embedding) AS similarity
    FROM docs_chunks
    WHERE 1 - (docs_chunks.embedding <=> query_embedding) > match_threshold
    ORDER BY docs_chunks.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- -------------------------------------------------------------
-- Step 4: Hybrid search RPC — vector + full-text fused with
-- Reciprocal Rank Fusion (RRF). rrf_score = sum(1 / (rrf_k + rank)).
-- "similarity" (cosine) is still returned for observability/grading.
-- -------------------------------------------------------------
DROP FUNCTION IF EXISTS hybrid_match_docs(VECTOR(1024), TEXT, INT, INT);

CREATE OR REPLACE FUNCTION hybrid_match_docs(
    query_embedding  VECTOR(1024),
    query_text       TEXT,
    match_count      INT DEFAULT 6,
    rrf_k            INT DEFAULT 60
)
RETURNS TABLE (
    id          UUID,
    content     TEXT,
    source_url  TEXT,
    page_title  TEXT,
    chunk_index INTEGER,
    similarity  FLOAT,
    rrf_score   FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT
            dc.id AS doc_id,
            ROW_NUMBER() OVER (ORDER BY dc.embedding <=> query_embedding) AS rank
        FROM docs_chunks dc
        ORDER BY dc.embedding <=> query_embedding
        LIMIT match_count * 3
    ),
    text_results AS (
        SELECT
            dc.id AS doc_id,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(dc.fts, websearch_to_tsquery('english', query_text)) DESC
            ) AS rank
        FROM docs_chunks dc
        WHERE dc.fts @@ websearch_to_tsquery('english', query_text)
        LIMIT match_count * 3
    ),
    fused AS (
        SELECT
            COALESCE(v.doc_id, t.doc_id) AS doc_id,
            COALESCE(1.0 / (rrf_k + v.rank), 0.0)
              + COALESCE(1.0 / (rrf_k + t.rank), 0.0) AS score
        FROM vector_results v
        FULL OUTER JOIN text_results t ON v.doc_id = t.doc_id
    )
    SELECT
        dc.id,
        dc.content,
        dc.source_url,
        dc.page_title,
        dc.chunk_index,
        1 - (dc.embedding <=> query_embedding) AS similarity,
        fused.score AS rrf_score
    FROM fused
    JOIN docs_chunks dc ON dc.id = fused.doc_id
    ORDER BY fused.score DESC
    LIMIT match_count;
END;
$$;

-- -------------------------------------------------------------
-- Step 5: Chat persistence tables (long-term memory).
-- chat_sessions holds the rolling conversation summary per session;
-- chat_messages holds the full turn-by-turn history.
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  TEXT        PRIMARY KEY,
    summary     TEXT        NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGSERIAL   PRIMARY KEY,
    session_id  TEXT        NOT NULL,
    role        TEXT        NOT NULL CHECK (role IN ('human', 'ai')),
    content     TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chat_messages_session_idx
ON chat_messages (session_id, id);

-- -------------------------------------------------------------
-- Verification helpers (run manually):
--   SELECT COUNT(*) FROM docs_chunks WHERE fts IS NOT NULL;
--   SELECT * FROM hybrid_match_docs(
--       (SELECT embedding FROM docs_chunks LIMIT 1), 'xml tags', 5);
-- =============================================================
