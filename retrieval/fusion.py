"""Reciprocal Rank Fusion for merging ranked result lists from multiple queries.

The SQL hybrid RPC fuses vector + full-text per query; this module fuses the
results ACROSS query variants (original question + HyDE paragraphs + keyword
variant) so chunks ranked highly by several variants float to the top.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _chunk_key(chunk: Dict[str, Any]) -> tuple:
    """Build a stable identity key for a chunk.

    Args:
        chunk: Retrieved chunk dict.

    Returns:
        Tuple key preferring the row id, falling back to (source_url, chunk_index).
    """
    if chunk.get("id"):
        return ("id", chunk["id"])
    return ("loc", chunk.get("source_url", ""), chunk.get("chunk_index"))


def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse multiple ranked result lists with Reciprocal Rank Fusion.

    score(chunk) = sum over lists of 1 / (rrf_k + rank_in_list), 1-based rank.
    Chunks appearing in several lists accumulate score and rank higher.

    Args:
        result_lists: One ranked chunk list per query variant.
        rrf_k: RRF dampening constant (larger = flatter contribution of top ranks).

    Returns:
        Deduplicated chunks sorted by fused score (descending), each annotated
        with a "fusion_score" field.
    """
    if not result_lists:
        return []

    scores: Dict[tuple, float] = {}
    chunk_by_key: Dict[tuple, Dict[str, Any]] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            key = _chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            if key not in chunk_by_key:
                chunk_by_key[key] = chunk

    fused = []
    for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        chunk = dict(chunk_by_key[key])
        chunk["fusion_score"] = score
        fused.append(chunk)

    logger.debug(f"Fused {sum(len(r) for r in result_lists)} results into {len(fused)} unique chunks")
    return fused
