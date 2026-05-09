# ───────────────────────────────────────────────
# FILE 4: retrieval/vector_store.py
# ───────────────────────────────────────────────
"""
Create `retrieval/vector_store.py`.

This module manages the Supabase connection and all vector store operations.

Requirements:
- Import: supabase (create_client, Client), os, logging, List, Dict, Any
- Load SUPABASE_URL and SUPABASE_SERVICE_KEY from environment (raise ValueError if missing)
- Create module-level client: supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

- Define three functions:

  def upsert_chunks(chunks: List[Dict[str, Any]]) -> int:
    - Each chunk dict has: content, embedding (List[float]), source_url, page_title, chunk_index
    - Calls supabase_client.table("docs_chunks").upsert(rows).execute()
    - Returns count of inserted rows
    - Logs: f"Upserted {count} chunks to Supabase"
    - Use batches of 50 rows to avoid payload limits

  def similarity_search(
      query_embedding: List[float],
      k: int = 6,
      threshold: float = 0.65
  ) -> List[Dict[str, Any]]:
    - Calls supabase_client.rpc("match_docs", {
        "query_embedding": query_embedding,
        "match_count": k,
        "match_threshold": threshold
      }).execute()
    - Returns result.data (list of dicts with: id, content, source_url, page_title, similarity)
    - Logs the number of results returned

  def health_check() -> bool:
    - Tries to query docs_chunks with limit 1
    - Returns True if successful, False otherwise
"""