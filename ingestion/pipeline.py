"""Ingestion orchestrator: crawl -> chunk -> diff against stored hashes -> embed -> upsert.

Idempotent: chunks whose content hash is unchanged are skipped (no Voyage
credits burned), and chunks that disappeared from a re-crawled page are deleted.
"""

import logging
from typing import Any, Dict, List

from ingestion.chunker import chunk_pages
from ingestion.crawler import crawl_docs
from ingestion.embedder import embed_documents
from retrieval.vector_store import delete_stale_chunks, fetch_existing_hashes, upsert_chunks

logger = logging.getLogger(__name__)


def _diff_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out chunks whose stored content hash is unchanged.

    Args:
        chunks: Freshly chunked documents with content_hash.

    Returns:
        Only the chunks that are new or whose content changed.
    """
    existing = fetch_existing_hashes()
    if not existing:
        return chunks

    changed = [
        c for c in chunks
        if existing.get((c["source_url"], c["chunk_index"])) != c["content_hash"]
    ]
    skipped = len(chunks) - len(changed)
    if skipped:
        logger.info(f"Skipping {skipped} unchanged chunks (content hash match)")
    return changed


def _prune_stale(chunks: List[Dict[str, Any]]) -> int:
    """Delete stored chunks no longer produced by the current crawl.

    Args:
        chunks: Full set of freshly produced chunks.

    Returns:
        Number of stale rows deleted.
    """
    valid_by_url: Dict[str, set] = {}
    for c in chunks:
        valid_by_url.setdefault(c["source_url"], set()).add(c["chunk_index"])

    deleted = 0
    for source_url, indexes in valid_by_url.items():
        deleted += delete_stale_chunks(source_url, indexes)
    return deleted


def run_ingestion(force_crawl: bool = False) -> dict:
    """Run the full ingestion pipeline.

    Args:
        force_crawl: Ignore the local crawl cache and re-crawl.

    Returns:
        Summary dict: pages, chunks, embedded, stored, pruned.

    Raises:
        RuntimeError: If any pipeline step fails.
    """
    try:
        pages = crawl_docs(force=force_crawl)
        logger.info(f"Step 1 complete: {len(pages)} pages crawled")
    except Exception as e:
        raise RuntimeError(f"Ingestion failed at step 1 (crawl): {e}") from e

    try:
        chunks = chunk_pages(pages)
        logger.info(f"Step 2 complete: {len(chunks)} chunks created")
    except Exception as e:
        raise RuntimeError(f"Ingestion failed at step 2 (chunk): {e}") from e

    try:
        to_embed = _diff_chunks(chunks)
        texts = [chunk["content"] for chunk in to_embed]
        embeddings = embed_documents(texts)
        for i, chunk in enumerate(to_embed):
            chunk["embedding"] = embeddings[i]
        logger.info(f"Step 3 complete: {len(embeddings)} embeddings generated ({len(chunks) - len(to_embed)} skipped)")
    except Exception as e:
        raise RuntimeError(f"Ingestion failed at step 3 (embed): {e}") from e

    try:
        count = upsert_chunks(to_embed)
        pruned = _prune_stale(chunks)
        logger.info(f"Step 4 complete: {count} chunks stored, {pruned} stale chunks pruned")
    except Exception as e:
        raise RuntimeError(f"Ingestion failed at step 4 (store): {e}") from e

    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "embedded": len(to_embed),
        "stored": count,
        "pruned": pruned,
    }
