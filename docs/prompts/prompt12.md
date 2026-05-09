# ───────────────────────────────────────────────
# FILE 12: agent/graph.py
# ───────────────────────────────────────────────
"""
Create `agent/graph.py`.

Assembles the LangGraph StateGraph. This is the heart of the agent.

Requirements:
- Import: StateGraph, END from langgraph.graph
- Import AgentState from agent.state
- Import rewrite_query, grade_relevance, generate_answer from agent.nodes
- Import search_docs from agent.tools (the raw function)
- Import retrieve from retrieval.retriever
- Import os

- Load RELEVANCE_THRESHOLD (default 0.70) and MAX_RETRY_COUNT (default 2) from env

SEARCH NODE (wrap retriever as a node, not a tool):
  def search_docs_node(state: AgentState) -> dict:
    - Takes the FIRST query in state["rewritten_queries"] for first attempt
    - On retries (retry_count > 0), uses the NEXT variant if available
    - Calls retrieve(query) for each variant and merges results (deduplicate by source_url)
    - Returns: {"retrieved_chunks": merged_chunks, "retry_count": state["retry_count"]}

CONDITIONAL EDGE FUNCTION:
  def should_retry(state: AgentState) -> str:
    - If state["relevance_score"] < RELEVANCE_THRESHOLD AND state["retry_count"] < MAX_RETRY_COUNT:
        return "search_docs_node"
    - Else: return "generate_answer"

FORMAT NODE (wrap format_citations as a node):
  def format_citations_node(state: AgentState) -> dict:
    - Calls the format_citations tool logic directly (not via tool decorator)
    - Returns: {"citations": citations, "final_response": formatted_answer}

GRAPH ASSEMBLY:
  def build_graph() -> CompiledGraph:
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

  # Module-level compiled graph (singleton)
  compiled_graph = build_graph()
"""