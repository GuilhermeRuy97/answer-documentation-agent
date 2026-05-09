import logging

from ingestion.crawler import crawl_docs
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_documents
from retrieval.vector_store import upsert_chunks

logger = logging.getLogger(__name__)


def run_ingestion(force_crawl: bool = False) -> dict:
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
        texts = [chunk["content"] for chunk in chunks]
        embeddings = embed_documents(texts)
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]
        logger.info(f"Step 3 complete: {len(embeddings)} embeddings generated")
    except Exception as e:
        raise RuntimeError(f"Ingestion failed at step 3 (embed): {e}") from e

    try:
        count = upsert_chunks(chunks)
        logger.info(f"Step 4 complete: {count} chunks stored in Supabase")
    except Exception as e:
        raise RuntimeError(f"Ingestion failed at step 4 (store): {e}") from e

    return {"pages": len(pages), "chunks": len(chunks), "stored": count}
