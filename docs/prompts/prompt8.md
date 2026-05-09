# ───────────────────────────────────────────────
# FILE 8: retrieval/retriever.py
# ───────────────────────────────────────────────
"""
Create `retrieval/retriever.py`.

High-level retriever used by the agent tool. Takes a query string, returns chunks.

Requirements:
- Import: retrieval.vector_store (similarity_search), ingestion.embedder (embed_query)
- Import: os, logging, List, Dict, Any

- Load RETRIEVAL_TOP_K (default 5) and RELEVANCE_THRESHOLD (default 0.70) from env

- Define one function:

  def retrieve(query: str, k: int = None, threshold: float = None) -> List[Dict[str, Any]]:
    - k defaults to RETRIEVAL_TOP_K, threshold defaults to RELEVANCE_THRESHOLD
    - Calls embed_query(query) to get query embedding
    - Calls similarity_search(embedding, k, threshold)
    - Returns list of chunk dicts: [{content, source_url, page_title, similarity}]
    - If no results found, logs a warning and returns []
    - Logs: f"Retrieved {len(results)} chunks for query: {query[:60]}..."
"""