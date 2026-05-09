# ───────────────────────────────────────────────
# FILE 1: ingestion/embedder.py
# ───────────────────────────────────────────────
"""
Create `ingestion/embedder.py`.

This module wraps the Voyage AI client for embedding documents and queries.

Requirements:
- Import: voyageai, os, logging, List from typing
- Load VOYAGE_API_KEY from environment (raise ValueError if missing)
- Create a module-level voyageai.Client instance
- Define two functions:

  def embed_documents(texts: List[str]) -> List[List[float]]:
    - Calls client.embed(texts, model="voyage-4", input_type="document")
    - Handles batching: Voyage AI max batch is 128. If len(texts) > 128, split into batches and concatenate results.
    - Returns list of embedding vectors (each is a list of 1024 floats)
    - Logs: f"Embedding {len(texts)} documents in {n_batches} batches"

  def embed_query(text: str) -> List[float]:
    - Calls client.embed([text], model="voyage-4", input_type="query")
    - Returns single embedding vector (list of 1024 floats)
    - Logs: f"Embedding query: {text[:60]}..."

- Add a simple __main__ block that tests both functions and prints the first 5 dims.
"""