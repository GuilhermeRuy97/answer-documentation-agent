# ───────────────────────────────────────────────
# FILE 14: api/routes.py
# ───────────────────────────────────────────────
"""
Create `api/routes.py`.

FastAPI route handlers. Import the compiled graph and session manager.

Requirements:
- Import: APIRouter, HTTPException from fastapi
- Import: AskRequest, AskResponse, HealthResponse, HistoryResponse from api.schemas
- Import: compiled_graph from agent.graph
- Import: get_or_create_session, get_history, save_history from agent.session
- Import: health_check from retrieval.vector_store
- Import: HumanMessage from langchain_core.messages
- Import: logging

router = APIRouter()

ROUTE 1: POST /ask
  async def ask(request: AskRequest) -> AskResponse:
    - session_id = get_or_create_session(request.session_id)
    - history = get_history(session_id)
    - Build initial state:
      {
        "session_id": session_id,
        "messages": history,
        "query": request.question,
        "rewritten_queries": [],
        "retrieved_chunks": [],
        "relevance_score": 0.0,
        "answer": "",
        "citations": [],
        "final_response": "",
        "retry_count": 0
      }
    - result = compiled_graph.invoke(initial_state)
    - save_history(session_id, result["messages"])
    - Return AskResponse(
        answer=result["final_response"],
        citations=result["citations"],
        session_id=session_id,
        rewritten_queries=result["rewritten_queries"]
      )
    - Wrap in try/except HTTPException(500)

ROUTE 2: GET /health
  async def health() -> HealthResponse:
    - vs_status = "connected" if health_check() else "error"
    - Return HealthResponse(status="ok", vector_store=vs_status)

ROUTE 3: GET /history/{session_id}
  async def get_session_history(session_id: str) -> HistoryResponse:
    - history = get_history(session_id)
    - If empty, raise HTTPException(404, "Session not found")
    - Convert BaseMessage list to HistoryMessage list
    - Return HistoryResponse(...)
"""