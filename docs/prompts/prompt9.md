# ───────────────────────────────────────────────
# FILE 9: agent/tools.py
# ───────────────────────────────────────────────
"""
Create `agent/tools.py`.

Defines two LangChain tools used by the LangGraph agent.

Requirements:
- Import: langchain_core.tools (tool decorator), retrieval.retriever (retrieve), os, logging, List, Dict

TOOL 1: search_docs
  @tool
  def search_docs(query: str, k: int = 5) -> List[Dict]:
    \"\"\"Search the Anthropic prompt engineering docs for relevant content.
    Use this when you need to find information about prompt engineering techniques.
    Args:
      query: The search query string
      k: Number of results to return (default 5)
    \"\"\"
    - Calls retrieve(query, k=k)
    - Returns the list of chunk dicts directly
    - Logs: f"search_docs called with: {query}"

TOOL 2: format_citations
  @tool
  def format_citations(answer: str, chunks: List[Dict]) -> Dict:
    \"\"\"Format an answer with numbered citations from retrieved chunks.
    Args:
      answer: The raw answer text
      chunks: List of retrieved chunk dicts with source_url, page_title, content
    \"\"\"
    - Deduplicate chunks by source_url (keep first occurrence per URL)
    - For each unique source, extract snippet: chunk["content"][:120].strip() + "..."
    - Build citations list: [{"title": page_title, "url": source_url, "snippet": snippet}]
    - Add citation markers to answer: if answer mentions a concept from source, add [1], [2], etc.
      Simple approach: append "\\n\\nSources: [1] title, [2] title..." at end of answer
    - Return: {"formatted_answer": str, "citations": List[Dict]}

Export: tools = [search_docs, format_citations]
"""