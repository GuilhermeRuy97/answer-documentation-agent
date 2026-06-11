"""High-level retriever used by the agent's search node and the search_docs tool.

Embeds the query and runs hybrid (vector + full-text RRF) search by default,
falling back to pure vector search when hybrid is disabled or unavailable.
"""

import logging
from typing import Any, Dict, List

from core.config import get_settings
from ingestion.embedder import embed_query
from retrieval.vector_store import hybrid_search, similarity_search

logger = logging.getLogger(__name__)


def retrieve(query: str, k: int | None = None, threshold: float | None = None) -> List[Dict[str, Any]]:
    """Retrieve ranked chunks for a query string.

    Args:
        query: Natural-language query (or HyDE paragraph).
        k: Max chunks to return; defaults to Settings.retrieval_top_k.
        threshold: Cosine recall floor for the pure-vector path and the
            hybrid fallback; defaults to Settings.recall_threshold. The
            hybrid RRF ranking itself has no score floor and widens via k.

    Returns:
        Ranked chunk dicts; empty list when nothing matches.
    """
    settings = get_settings()
    k = k if k is not None else settings.retrieval_top_k
    threshold = threshold if threshold is not None else settings.recall_threshold

    embedding = embed_query(query)

    if settings.use_hybrid_search:
        results = hybrid_search(embedding, query, k=k, threshold=threshold)
    else:
        results = similarity_search(embedding, k=k, threshold=threshold)

    if not results:
        logger.warning(f"No chunks retrieved for query: {query[:60]}...")
        return []

    logger.info(f"Retrieved {len(results)} chunks for query: {query[:60]}...")
    return results
