"""LangChain tool wrappers around the shared retrieval and citation logic.

Thin adapters: the actual implementations live in retrieval/retriever.py and
agent/citations.py and are the same code the graph nodes use.
"""

import logging
from typing import Dict, List

from langchain_core.tools import tool

from agent.citations import build_citations
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
      answer: The raw answer text containing [N] markers
      chunks: List of retrieved chunk dicts with source_url, page_title, content
    """
    result = build_citations(answer, chunks)
    return {"formatted": result["final_response"], "citations": result["citations"]}


tools = [search_docs, format_citations]
