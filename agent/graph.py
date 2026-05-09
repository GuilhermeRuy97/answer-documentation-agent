import os

from langgraph.graph import END, StateGraph

from agent.nodes import generate_answer, grade_relevance, rewrite_query
from agent.state import AgentState
from retrieval.retriever import retrieve

RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.65"))
MAX_RETRY_COUNT = int(os.getenv("MAX_RETRY_COUNT", "2"))


def search_docs_node(state: AgentState) -> dict:
    queries = state.get("rewritten_queries", [state["query"]])
    retry_count = state.get("retry_count", 0)

    if retry_count == 0:
        selected = queries[:1]
    else:
        idx = min(retry_count, len(queries) - 1)
        selected = queries[idx : idx + 1]

    seen_urls: set = set()
    merged: list = []
    for q in selected:
        for chunk in retrieve(q):
            url = chunk.get("source_url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                merged.append(chunk)

    return {"retrieved_chunks": merged, "retry_count": retry_count}


def should_retry(state: AgentState) -> str:
    if state["relevance_score"] < RELEVANCE_THRESHOLD and state["retry_count"] < MAX_RETRY_COUNT:
        return "search_docs_node"
    return "generate_answer"


def format_citations_node(state: AgentState) -> dict:
    answer = state.get("answer", "")
    chunks = state.get("retrieved_chunks", [])

    seen_urls: set = set()
    citations: list = []
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

    return {"citations": citations, "final_response": formatted_answer}


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
