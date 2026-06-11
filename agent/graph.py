"""LangGraph StateGraph assembly.

Flow:
    load_memory -> rewrite_query -> search_docs -> grade_relevance
        -> (retry: search_docs | ok: generate_answer)
        -> format_citations -> save_memory -> END
"""

import logging

from langgraph.graph import END, StateGraph

from agent.citations import build_citations
from agent.nodes import (
    generate_answer,
    grade_relevance,
    load_memory,
    rewrite_query,
    save_memory,
)
from agent.state import AgentState
from core.config import get_settings
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import rerank
from retrieval.retriever import retrieve

logger = logging.getLogger(__name__)


def search_docs_node(state: AgentState) -> dict:
    """Search all query variants, fuse with RRF, then rerank against the question.

    On retries the per-variant breadth (k) widens and the cosine recall floor
    drops, so the reranker has more candidates to work with. The recall floor
    applies to the pure-vector path and the hybrid fallback; the hybrid RRF
    ranking itself widens through k only.

    Args:
        state: Current agent state.

    Returns:
        Partial update with retrieved_chunks.
    """
    settings = get_settings()
    original_query = state["query"]
    variants = state.get("rewritten_queries", [original_query])
    retry_count = state.get("retry_count", 0)

    all_queries = list(dict.fromkeys([original_query] + variants))

    recall_threshold = max(0.15, settings.recall_threshold - 0.05 * retry_count)
    per_variant_k = settings.retrieval_top_k * 2 * (retry_count + 1)

    result_lists = []
    for q in all_queries:
        results = retrieve(q, k=per_variant_k, threshold=recall_threshold)
        if results:
            result_lists.append(results)

    # Fuse rankings across variants: chunks ranked highly by several variants win.
    candidates = reciprocal_rank_fusion(result_lists, rrf_k=settings.rrf_k)

    if candidates:
        candidates = rerank(original_query, candidates, top_k=settings.rerank_top_k)

    return {"retrieved_chunks": candidates}


def should_retry(state: AgentState) -> str:
    """Decide whether to re-search or proceed to answer generation.

    retry_count counts completed search->grade cycles, so cycles beyond the
    first are retries: with MAX_RETRY_COUNT=2 the agent searches at most
    3 times (1 initial + 2 retries).

    Args:
        state: Current agent state.

    Returns:
        Next node name: "search_docs" or "generate_answer".
    """
    settings = get_settings()
    retries_used = state["retry_count"] - 1
    if state["relevance_score"] < settings.relevance_threshold and retries_used < settings.max_retry_count:
        logger.info(f"Relevance {state['relevance_score']:.3f} below threshold; retry {retries_used + 1}")
        return "search_docs"
    return "generate_answer"


def format_citations_node(state: AgentState) -> dict:
    """Format inline [N] markers into a deduplicated citation list.

    Args:
        state: Current agent state.

    Returns:
        Partial update with citations and final_response.
    """
    result = build_citations(state.get("answer", ""), state.get("retrieved_chunks", []))
    return {"citations": result["citations"], "final_response": result["final_response"]}


def build_graph():
    """Assemble and compile the agent StateGraph.

    Returns:
        The compiled LangGraph runnable.
    """
    graph = StateGraph(AgentState)

    graph.add_node("load_memory", load_memory)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("search_docs", search_docs_node)
    graph.add_node("grade_relevance", grade_relevance)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("format_citations", format_citations_node)
    graph.add_node("save_memory", save_memory)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "rewrite_query")
    graph.add_edge("rewrite_query", "search_docs")
    graph.add_edge("search_docs", "grade_relevance")
    graph.add_conditional_edges("grade_relevance", should_retry)
    graph.add_edge("generate_answer", "format_citations")
    graph.add_edge("format_citations", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()


compiled_graph = build_graph()
