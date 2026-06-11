"""Supabase vector store access: upserts, similarity search, hybrid search.

Clients are created lazily so the module can be imported (and unit-tested)
without Supabase credentials in the environment.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from supabase import Client, create_client

from core.config import get_settings

logger = logging.getLogger(__name__)

_client: Optional[Client] = None

_UPSERT_BATCH_SIZE = 50


def get_client() -> Client:
    """Return the lazily-initialized Supabase client.

    Returns:
        The shared Supabase client.

    Raises:
        RuntimeError: If Supabase credentials are not configured.
    """
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set to use the vector store")

    _client = create_client(settings.supabase_url, settings.supabase_service_key)
    return _client


def upsert_chunks(chunks: List[Dict[str, Any]]) -> int:
    """Upsert chunk rows into docs_chunks in batches.

    Args:
        chunks: Rows with content, embedding, source_url, page_title, chunk_index, content_hash.

    Returns:
        Number of rows upserted.
    """
    if not chunks:
        return 0

    client = get_client()
    count = 0
    for i in range(0, len(chunks), _UPSERT_BATCH_SIZE):
        batch = chunks[i : i + _UPSERT_BATCH_SIZE]
        client.table("docs_chunks").upsert(batch, on_conflict="source_url,chunk_index").execute()
        count += len(batch)
    logger.info(f"Upserted {count} chunks to Supabase")
    return count


def fetch_existing_hashes() -> Dict[tuple, str]:
    """Fetch (source_url, chunk_index) -> content_hash for all stored chunks.

    Used by the ingestion pipeline to skip re-embedding unchanged content.

    Returns:
        Mapping of (source_url, chunk_index) to content_hash. Empty on failure
        (e.g. before the migration adds the content_hash column).
    """
    try:
        client = get_client()
        rows: List[Dict[str, Any]] = []
        page_size = 1000
        offset = 0
        while True:
            result = (
                client.table("docs_chunks")
                .select("source_url, chunk_index, content_hash")
                .range(offset, offset + page_size - 1)
                .execute()
            )
            rows.extend(result.data or [])
            if len(result.data or []) < page_size:
                break
            offset += page_size
        return {
            (r["source_url"], r["chunk_index"]): r.get("content_hash") or ""
            for r in rows
        }
    except Exception:
        logger.warning("Could not fetch existing content hashes; will re-embed everything", exc_info=True)
        return {}


def delete_stale_chunks(source_url: str, valid_indexes: Set[int]) -> int:
    """Delete chunks of a page whose chunk_index is no longer produced by chunking.

    Prevents orphaned chunks when a re-crawled page shrinks.

    Args:
        source_url: Page URL whose chunks should be pruned.
        valid_indexes: chunk_index values that are still valid.

    Returns:
        Number of stale rows deleted (best effort; 0 on failure).
    """
    try:
        client = get_client()
        result = (
            client.table("docs_chunks")
            .select("id, chunk_index")
            .eq("source_url", source_url)
            .execute()
        )
        stale_ids = [r["id"] for r in (result.data or []) if r["chunk_index"] not in valid_indexes]
        if not stale_ids:
            return 0
        client.table("docs_chunks").delete().in_("id", stale_ids).execute()
        logger.info(f"Deleted {len(stale_ids)} stale chunks for {source_url}")
        return len(stale_ids)
    except Exception:
        logger.warning(f"Failed to delete stale chunks for {source_url}", exc_info=True)
        return 0


def similarity_search(
    query_embedding: List[float],
    k: int = 6,
    threshold: float = 0.30,
) -> List[Dict[str, Any]]:
    """Pure vector search via the match_docs RPC.

    Args:
        query_embedding: 1024-dim query embedding.
        k: Max rows to return.
        threshold: Minimum cosine similarity.

    Returns:
        Chunk dicts with id, content, source_url, page_title, chunk_index, similarity.
    """
    client = get_client()
    result = client.rpc(
        "match_docs",
        {
            "query_embedding": query_embedding,
            "match_count": k,
            "match_threshold": threshold,
        },
    ).execute()
    logger.info(f"similarity_search returned {len(result.data)} results")
    return result.data


def hybrid_search(
    query_embedding: List[float],
    query_text: str,
    k: int = 6,
    threshold: float | None = None,
) -> List[Dict[str, Any]]:
    """Hybrid vector + full-text search fused with Reciprocal Rank Fusion.

    Falls back to pure vector search if the hybrid RPC is unavailable
    (i.e. migrate_hybrid_search.sql has not been applied yet).

    Args:
        query_embedding: 1024-dim query embedding.
        query_text: Raw query text for the full-text leg.
        k: Max rows to return.
        threshold: Cosine recall floor for the vector fallback (RRF itself
            has no score floor); defaults to Settings.recall_threshold.

    Returns:
        Chunk dicts with id, content, source_url, page_title, chunk_index,
        similarity, rrf_score.
    """
    settings = get_settings()
    client = get_client()
    try:
        result = client.rpc(
            "hybrid_match_docs",
            {
                "query_embedding": query_embedding,
                "query_text": query_text,
                "match_count": k,
                "rrf_k": settings.rrf_k,
            },
        ).execute()
        logger.info(f"hybrid_search returned {len(result.data)} results")
        return result.data
    except Exception:
        logger.warning("hybrid_match_docs RPC failed (migration applied?); falling back to vector search")
        if threshold is None:
            threshold = settings.recall_threshold
        return similarity_search(query_embedding, k=k, threshold=threshold)


def health_check() -> bool:
    """Check that the docs_chunks table is reachable.

    Returns:
        True if a trivial select succeeds, False otherwise.
    """
    try:
        get_client().table("docs_chunks").select("id").limit(1).execute()
        return True
    except Exception:
        logger.exception("Supabase health check failed")
        return False
