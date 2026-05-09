import os
import logging
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    all_chunks: List[Dict[str, Any]] = []
    for page in pages:
        text_chunks = splitter.split_text(page["markdown"])
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text) < 50:
                continue
            all_chunks.append({
                "content": chunk_text,
                "source_url": page["url"],
                "page_title": page["title"],
                "chunk_index": i,
            })

    logger.info(f"Split {len(pages)} pages into {len(all_chunks)} chunks")
    return all_chunks
