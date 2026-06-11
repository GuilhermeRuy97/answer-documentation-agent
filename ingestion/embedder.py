"""Voyage AI embedding client wrapper.

The client is created lazily so the module can be imported without
VOYAGE_API_KEY set (required for unit testing and tooling).
The input_type parameter is mandatory: it meaningfully affects retrieval quality.
"""

import logging
from typing import List, Optional

import voyageai

from core.config import get_settings

logger = logging.getLogger(__name__)

_client: Optional[voyageai.Client] = None

_BATCH_SIZE = 128


def get_voyage_client() -> voyageai.Client:
    """Return the lazily-initialized Voyage AI client.

    Returns:
        The shared Voyage client.

    Raises:
        RuntimeError: If VOYAGE_API_KEY is not configured.
    """
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    if not settings.voyage_api_key:
        raise RuntimeError("VOYAGE_API_KEY must be set to use embeddings")

    _client = voyageai.Client(api_key=settings.voyage_api_key)
    return _client


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed document texts in batches with input_type='document'.

    Args:
        texts: Document chunk texts.

    Returns:
        One embedding_dim-dimensional embedding per input text.
    """
    if not texts:
        return []

    settings = get_settings()
    client = get_voyage_client()
    n_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE
    logger.info(f"Embedding {len(texts)} documents in {n_batches} batches")

    embeddings: List[List[float]] = []
    for i in range(n_batches):
        batch = texts[i * _BATCH_SIZE : (i + 1) * _BATCH_SIZE]
        result = client.embed(
            batch,
            model=settings.embedding_model,
            input_type="document",
            output_dimension=settings.embedding_dim,
        )
        embeddings.extend(result.embeddings)

    return embeddings


def embed_query(text: str) -> List[float]:
    """Embed a search query with input_type='query'.

    Args:
        text: Query string.

    Returns:
        embedding_dim-dimensional query embedding.
    """
    settings = get_settings()
    logger.debug(f"Embedding query: {text[:60]}...")
    result = get_voyage_client().embed(
        [text],
        model=settings.embedding_model,
        input_type="query",
        output_dimension=settings.embedding_dim,
    )
    return result.embeddings[0]
