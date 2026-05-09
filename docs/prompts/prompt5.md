# ───────────────────────────────────────────────
# FILE 5: ingestion/pipeline.py
# ───────────────────────────────────────────────
"""
Create `ingestion/pipeline.py`.

This is the main ingestion orchestrator. It wires crawl → chunk → embed → store.

Requirements:
- Import from: ingestion.crawler, ingestion.chunker, ingestion.embedder, retrieval.vector_store
- Import: logging, os

- Define one function:

  def def run_ingestion(force_crawl: bool = False) -> dict:
    - Step 1: pages = crawl_docs(force=force_crawl)
    - Log: f"Step 1 complete: {len(pages)} pages crawled"

    - Step 2: Call chunk_pages(pages) → chunks
    - Log: f"Step 2 complete: {len(chunks)} chunks created"

    - Step 3: Extract chunk["content"] for each chunk, call embed_documents(texts) → embeddings
    - Attach embedding to each chunk: chunk["embedding"] = embeddings[i]
    - Log: f"Step 3 complete: {len(embeddings)} embeddings generated"

    - Step 4: Call upsert_chunks(chunks) → count
    - Log: f"Step 4 complete: {count} chunks stored in Supabase"

    - Return: {"pages": len(pages), "chunks": len(chunks), "stored": count}

    - Wrap each step in try/except and re-raise with a descriptive message.
"""