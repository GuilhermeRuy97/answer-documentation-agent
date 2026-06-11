"""Citation building shared by the graph node and the format_citations tool."""

import re
from typing import Any, Dict, List

_SNIPPET_MAX_CHARS = 120


def _snippet(content: str) -> str:
    """Build a short citation preview, with an ellipsis only when truncated.

    Args:
        content: Full chunk content.

    Returns:
        Preview string of at most _SNIPPET_MAX_CHARS characters (plus "...").
    """
    snippet = content[:_SNIPPET_MAX_CHARS].strip()
    if len(content) > _SNIPPET_MAX_CHARS:
        snippet += "..."
    return snippet


def build_citations(answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the citation list from [N] markers in the answer, deduped by URL.

    Rewrites the answer so visible markers map 1:1 to the citation list. If two
    chunks share a source URL, their markers collapse to one citation number.
    Invalid markers (numbers Claude hallucinated) are dropped. When the answer
    has no markers but chunks exist, the top 3 sources are surfaced as fallback.

    Args:
        answer: Raw generated answer containing [N] markers.
        chunks: Retrieved chunks in the order they were numbered for Claude.

    Returns:
        {"final_response": str, "citations": [{title, url, snippet}]}
    """
    used_indexes = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})

    citations: List[Dict[str, Any]] = []
    url_to_seq: Dict[str, int] = {}
    n_to_seq: Dict[int, int] = {}

    for n in used_indexes:
        if not (1 <= n <= len(chunks)):
            continue
        chunk = chunks[n - 1]
        url = chunk.get("source_url", "")
        if url in url_to_seq:
            n_to_seq[n] = url_to_seq[url]
            continue
        seq = len(citations) + 1
        url_to_seq[url] = seq
        n_to_seq[n] = seq
        citations.append({
            "title": chunk.get("page_title", "Untitled"),
            "url": url,
            "snippet": _snippet(chunk.get("content", "")),
        })

    def _remap(match: "re.Match") -> str:
        original = int(match.group(1))
        if original in n_to_seq:
            return f"[{n_to_seq[original]}]"
        return ""  # drop invalid markers

    final_answer = re.sub(r"\[(\d+)\]", _remap, answer)

    if not citations and chunks:
        for chunk in chunks[:3]:
            url = chunk.get("source_url", "")
            if url in url_to_seq:
                continue
            url_to_seq[url] = len(citations) + 1
            citations.append({
                "title": chunk.get("page_title", "Untitled"),
                "url": url,
                "snippet": _snippet(chunk.get("content", "")),
            })

    return {"final_response": final_answer, "citations": citations}
