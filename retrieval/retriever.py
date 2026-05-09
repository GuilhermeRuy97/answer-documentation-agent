import os
import logging
from typing import Any, Dict, List

from ingestion.embedder import embed_query
from retrieval.vector_store import similarity_search

logger = logging.getLogger(__name__)

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "6"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.65"))


def retrieve(query: str, k: int = None, threshold: float = None) -> List[Dict[str, Any]]:
    k = k if k is not None else RETRIEVAL_TOP_K
    threshold = threshold if threshold is not None else RELEVANCE_THRESHOLD

    embedding = embed_query(query)
    results = similarity_search(embedding, k, threshold)

    if not results:
        logger.warning(f"No chunks retrieved for query: {query[:60]}...")
        return []

    logger.info(f"Retrieved {len(results)} chunks for query: {query[:60]}...")
    return results
