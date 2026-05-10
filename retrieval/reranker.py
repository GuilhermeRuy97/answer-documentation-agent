import logging
from typing import Any, Dict, List

from ingestion.embedder import client as voyage_client

logger = logging.getLogger(__name__)

_RERANK_MODEL = "rerank-2"


def rerank(query: str, chunks: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
    """Re-rank candidate chunks against the query using Voyage's rerank-2 model."""
    if not chunks:
        return chunks

    documents = [c["content"] for c in chunks]
    top_k = top_k or len(chunks)

    try:
        result = voyage_client.rerank(
            query=query,
            documents=documents,
            model=_RERANK_MODEL,
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

    logger.info(f"Reranked {len(chunks)} chunks → top {len(reranked)} (top score: {reranked[0]['rerank_score']:.3f})")
    return reranked
