"""Voyage rerank wrapper used as the precision stage after hybrid retrieval."""

import logging
from typing import Any, Dict, List

from core.config import get_settings
from ingestion.embedder import get_voyage_client

logger = logging.getLogger(__name__)


def rerank(query: str, chunks: List[Dict[str, Any]], top_k: int | None = None) -> List[Dict[str, Any]]:
    """Re-rank candidate chunks against the query using Voyage's rerank model.

    Args:
        query: The original user question.
        chunks: Candidate chunk dicts (must have a "content" key).
        top_k: Number of chunks to keep; defaults to len(chunks).

    Returns:
        Chunks sorted by relevance, each annotated with "rerank_score".
        Falls back to the original order on API failure.
    """
    if not chunks:
        return chunks

    settings = get_settings()
    documents = [c["content"] for c in chunks]
    top_k = top_k or len(chunks)

    try:
        result = get_voyage_client().rerank(
            query=query,
            documents=documents,
            model=settings.rerank_model,
            top_k=top_k,
        )
    except Exception:
        logger.exception("Rerank failed; returning original chunks")
        return chunks[:top_k]

    reranked: List[Dict[str, Any]] = []
    for r in result.results:
        chunk = dict(chunks[r.index])
        chunk["rerank_score"] = r.relevance_score
        reranked.append(chunk)

    logger.info(
        f"Reranked {len(chunks)} chunks -> top {len(reranked)} (top score: {reranked[0]['rerank_score']:.3f})"
    )
    return reranked
