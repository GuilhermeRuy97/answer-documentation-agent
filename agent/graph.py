import os
import re

from langgraph.graph import END, StateGraph

from agent.nodes import generate_answer, grade_relevance, rewrite_query
from agent.state import AgentState
from retrieval.reranker import rerank
from retrieval.retriever import retrieve

RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.45"))
MAX_RETRY_COUNT = int(os.getenv("MAX_RETRY_COUNT", "2"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "6"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "6"))
RECALL_THRESHOLD = float(os.getenv("RECALL_THRESHOLD", "0.30"))


def search_docs_node(state: AgentState) -> dict:
    """Search across ALL query variants in parallel, union, then rerank against original question."""
    original_query = state["query"]
    variants = state.get("rewritten_queries", [original_query])
    retry_count = state.get("retry_count", 0)

    # Always include the original question; deduplicate while preserving order
    all_queries = list(dict.fromkeys([original_query] + variants))

    # On retry, lower the recall floor and pull wider per variant; rerank narrows down later
    recall_threshold = max(0.15, RECALL_THRESHOLD - 0.05 * retry_count)
    per_variant_k = RETRIEVAL_TOP_K * 2 * (retry_count + 1)

    seen_keys: set = set()
    candidates: list = []
    for q in all_queries:
        for chunk in retrieve(q, k=per_variant_k, threshold=recall_threshold):
            key = (chunk.get("source_url", ""), chunk.get("chunk_index"), chunk.get("id"))
            if key not in seen_keys:
                seen_keys.add(key)
                candidates.append(chunk)

    # Rerank against the original user question (not the hypothetical answers)
    if candidates:
        candidates = rerank(original_query, candidates, top_k=RERANK_TOP_K)

    return {"retrieved_chunks": candidates, "retry_count": retry_count}


def should_retry(state: AgentState) -> str:
    if state["relevance_score"] < RELEVANCE_THRESHOLD and state["retry_count"] < MAX_RETRY_COUNT:
        return "search_docs_node"
    return "generate_answer"


def format_citations_node(state: AgentState) -> dict:
    """Build the citation list from [N] markers Claude used, deduping by URL.

    Rewrites the answer so visible markers map 1:1 to the citation list. If two chunks
    share a source URL, both [3] and [5] (say) collapse to the same citation number.
    """
    answer = state.get("answer", "")
    chunks = state.get("retrieved_chunks", [])

    used_indexes = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})

    citations: list = []
    url_to_seq: dict = {}
    n_to_seq: dict = {}

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
        snippet = chunk.get("content", "")[:120].strip() + "..."
        citations.append({
            "title": chunk.get("page_title", "Untitled"),
            "url": url,
            "snippet": snippet,
        })

    def _remap(match: "re.Match") -> str:
        original = int(match.group(1))
        if original in n_to_seq:
            return f"[{n_to_seq[original]}]"
        return ""  # drop invalid markers Claude may have hallucinated

    final_answer = re.sub(r"\[(\d+)\]", _remap, answer)

    # Fallback: if Claude didn't use any markers but we have chunks, surface top sources
    if not citations and chunks:
        for chunk in chunks[:3]:
            url = chunk.get("source_url", "")
            if url in url_to_seq:
                continue
            url_to_seq[url] = len(citations) + 1
            snippet = chunk.get("content", "")[:120].strip() + "..."
            citations.append({
                "title": chunk.get("page_title", "Untitled"),
                "url": url,
                "snippet": snippet,
            })

    return {"citations": citations, "final_response": final_answer}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("search_docs_node", search_docs_node)
    graph.add_node("grade_relevance", grade_relevance)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("format_citations_node", format_citations_node)

    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "search_docs_node")
    graph.add_edge("search_docs_node", "grade_relevance")
    graph.add_conditional_edges("grade_relevance", should_retry)
    graph.add_edge("generate_answer", "format_citations_node")
    graph.add_edge("format_citations_node", END)

    return graph.compile()


compiled_graph = build_graph()
