# ───────────────────────────────────────────────
# FILE 10: agent/nodes.py
# ───────────────────────────────────────────────
"""
Create `agent/nodes.py`.

Defines the three LangGraph node functions. Each takes AgentState, returns partial state dict.

Requirements:
- Import: anthropic, os, logging, json
- Import AgentState from agent.state
- Import search_docs from agent.tools (use the underlying function, not as a tool)
- Import retrieve from retrieval.retriever

- Load: ANTHROPIC_API_KEY, MAX_RETRY_COUNT (default 2) from env
- Create module-level: client = anthropic.Anthropic()

NODE 1: rewrite_query(state: AgentState) -> dict
  - Takes state["query"] and state["messages"] (history context)
  - Calls Claude claude-sonnet-4-6 with a prompt like:
    "Given this question: {query}
     Generate 2 alternative phrasings that might retrieve better results
     from a vector database of prompt engineering documentation.
     Return ONLY a JSON array of 2 strings. No explanation."
  - Parses the JSON array response
  - Returns: {"rewritten_queries": [original_query] + [variant1, variant2]}
  - On parse error, returns: {"rewritten_queries": [state["query"]]}
  - Logs: f"Generated {len(variants)} query variants"

NODE 2: grade_relevance(state: AgentState) -> dict
  - Reads state["retrieved_chunks"]
  - If empty, returns {"relevance_score": 0.0, "retry_count": state["retry_count"] + 1}
  - Calculates mean of chunk["similarity"] for all chunks
  - Returns: {"relevance_score": mean_score}
  - Logs: f"Relevance score: {mean_score:.3f} from {len(chunks)} chunks"

NODE 3: generate_answer(state: AgentState) -> dict
  - Builds context string from state["retrieved_chunks"]:
    Each chunk: "SOURCE: {source_url}\\nCONTENT: {content}\\n---"
  - Calls Claude claude-sonnet-4-6 with system + user message:
    System: "You are an expert on Anthropic prompt engineering. Answer questions
             ONLY using the provided documentation context. Be specific and cite sources."
    User: f"Context:\\n{context}\\n\\nQuestion: {state['query']}"
  - Returns: {"answer": response_text}
  - Also appends HumanMessage(query) + AIMessage(answer) to messages
  - Logs: f"Generated answer ({len(answer)} chars)"

NOTE: All nodes are pure functions. Return ONLY the keys you update.
"""