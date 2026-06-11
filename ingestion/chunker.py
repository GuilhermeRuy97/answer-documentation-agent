"""Markdown cleaning and chunking with contextual headers.

Each chunk is prefixed with its page title and nearest section heading before
embedding ("contextual chunking"), which measurably improves retrieval
precision because chunks carry their own topical context.
"""

import hashlib
import logging
import re
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import get_settings

logger = logging.getLogger(__name__)

_BOILERPLATE_LINES = (
    "Was this page helpful?",
    "Ask Docs",
    "Copy page",
)

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


def _clean_markdown(text: str) -> str:
    """Strip docs-site boilerplate, image markdown, and excess whitespace.

    Args:
        text: Raw page markdown.

    Returns:
        Cleaned markdown.
    """
    # Remove image markdown: ![alt](url) and linked images [![...](...)](...)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(stripped == bp or stripped.startswith(bp) for bp in _BOILERPLATE_LINES):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    """Compute a stable sha256 hash of chunk content.

    Used by the ingestion pipeline to skip re-embedding unchanged chunks.

    Args:
        text: Chunk content.

    Returns:
        Hex sha256 digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _heading_context(full_text: str, chunk_text: str) -> str:
    """Find the nearest markdown heading at or before the chunk's position.

    Args:
        full_text: The full cleaned page markdown.
        chunk_text: The chunk content (a substring of full_text in most cases).

    Returns:
        The nearest preceding heading text, or "" when not found.
    """
    pos = full_text.find(chunk_text[:200])
    if pos < 0:
        return ""
    last_heading = ""
    for match in _HEADING_RE.finditer(full_text):
        if match.start() > pos:
            break
        last_heading = match.group(2).strip()
    return last_heading


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split crawled pages into contextualized, hash-stamped chunks.

    Args:
        pages: Page dicts with url, title, markdown.

    Returns:
        Chunk dicts: content (with contextual header), source_url, page_title,
        chunk_index, content_hash.
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    all_chunks: List[Dict[str, Any]] = []
    for page in pages:
        clean = _clean_markdown(page["markdown"])
        text_chunks = splitter.split_text(clean)
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text) < settings.min_chunk_chars:
                continue
            heading = _heading_context(clean, chunk_text)
            # Contextual header: page title (+ section) prepended so the chunk
            # is self-describing for both embedding and full-text search.
            header = page["title"] if not heading or heading == page["title"] else f"{page['title']} - {heading}"
            contextualized = f"{header}\n\n{chunk_text}"
            all_chunks.append({
                "content": contextualized,
                "source_url": page["url"],
                "page_title": page["title"],
                "chunk_index": i,
                "content_hash": content_hash(contextualized),
            })

    logger.info(f"Split {len(pages)} pages into {len(all_chunks)} chunks")
    return all_chunks
