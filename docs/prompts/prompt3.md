# ───────────────────────────────────────────────
# FILE 3: ingestion/chunker.py
# ───────────────────────────────────────────────
"""
Create `ingestion/chunker.py`.

This module splits crawled pages into overlapping chunks, preserving source metadata.

Requirements:
- Import: langchain_text_splitters.RecursiveCharacterTextSplitter, os, logging, List, Dict, Any
- Load CHUNK_SIZE (default 1200) and CHUNK_OVERLAP (default 200) from environment as ints

- Define one function:

  def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    - Creates RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    - For each page in pages:
        - Splits page["markdown"] into text chunks
        - For each chunk, creates a dict:
          {
            "content": chunk_text,
            "source_url": page["url"],
            "page_title": page["title"],
            "chunk_index": i  # index within this page
          }
        - Skips chunks shorter than 50 characters
    - Returns flat list of all chunk dicts across all pages
    - Logs: f"Split {len(pages)} pages into {len(all_chunks)} chunks"
"""