import os
import re
import logging
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


_BOILERPLATE_LINES = (
    "Was this page helpful?",
    "Ask Docs",
    "Copy page",
)


def _clean_markdown(text: str) -> str:
    """Strip docs-site boilerplate, image markdown, and excess whitespace."""
    # Remove image markdown: ![alt](url) and linked images [![...](...)](...)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    # Remove standalone boilerplate lines
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(stripped == bp or stripped.startswith(bp) for bp in _BOILERPLATE_LINES):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    all_chunks: List[Dict[str, Any]] = []
    for page in pages:
        clean = _clean_markdown(page["markdown"])
        text_chunks = splitter.split_text(clean)
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
