import logging
from typing import Dict, List

from langchain_core.tools import tool

from retrieval.retriever import retrieve

logger = logging.getLogger(__name__)


@tool
def search_docs(query: str, k: int = 6) -> List[Dict]:
    """Search the Anthropic prompt engineering docs for relevant content.
    Use this when you need to find information about prompt engineering techniques.
    Args:
      query: The search query string
      k: Number of results to return (default 6)
    """
    logger.info(f"search_docs called with: {query}")
    return retrieve(query, k=k)


@tool
def format_citations(answer: str, chunks: List[Dict]) -> Dict:
    """Format an answer with numbered citations from retrieved chunks.
    Args:
      answer: The raw answer text
      chunks: List of retrieved chunk dicts with source_url, page_title, content
    """
    seen_urls: set = set()
    citations: List[Dict] = []
    for chunk in chunks:
        url = chunk.get("source_url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        snippet = chunk.get("content", "")[:120].strip() + "..."
        citations.append({
            "title": chunk.get("page_title", "Untitled"),
            "url": url,
            "snippet": snippet,
        })

    source_lines = ", ".join(f"[{i + 1}] {c['title']}" for i, c in enumerate(citations))
    formatted_answer = answer + (f"\n\nSources: {source_lines}" if citations else "")

    return {"formatted_answer": formatted_answer, "citations": citations}


tools = [search_docs, format_citations]
