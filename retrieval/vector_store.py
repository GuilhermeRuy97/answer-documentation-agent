import os
import logging
from typing import Any, Dict, List

from supabase import create_client, Client

logger = logging.getLogger(__name__)

_supabase_url = os.getenv("SUPABASE_URL")
_supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

if not _supabase_url:
    raise ValueError("SUPABASE_URL environment variable is not set")
if not _supabase_key:
    raise ValueError("SUPABASE_SERVICE_KEY environment variable is not set")

supabase_client: Client = create_client(_supabase_url, _supabase_key)

_UPSERT_BATCH_SIZE = 50


def upsert_chunks(chunks: List[Dict[str, Any]]) -> int:
    count = 0
    for i in range(0, len(chunks), _UPSERT_BATCH_SIZE):
        batch = chunks[i : i + _UPSERT_BATCH_SIZE]
        supabase_client.table("docs_chunks").upsert(
            batch, on_conflict="source_url,chunk_index"
        ).execute()
        count += len(batch)
    logger.info(f"Upserted {count} chunks to Supabase")
    return count


def similarity_search(
    query_embedding: List[float],
    k: int = 6,
    threshold: float = 0.65,
) -> List[Dict[str, Any]]:
    result = supabase_client.rpc(
        "match_docs",
        {
            "query_embedding": query_embedding,
            "match_count": k,
            "match_threshold": threshold,
        },
    ).execute()
    logger.info(f"similarity_search returned {len(result.data)} results")
    return result.data


def health_check() -> bool:
    try:
        supabase_client.table("docs_chunks").select("id").limit(1).execute()
        return True
    except Exception:
        logger.exception("Supabase health check failed")
        return False
