import logging

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import compiled_graph
from agent.session import get_history, get_or_create_session, save_history
from api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    HistoryMessage,
    HistoryResponse,
)
from retrieval.vector_store import health_check

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    try:
        session_id = get_or_create_session(request.session_id)
        history = get_history(session_id)

        initial_state = {
            "session_id": session_id,
            "messages": history,
            "query": request.question,
            "rewritten_queries": [],
            "retrieved_chunks": [],
            "relevance_score": 0.0,
            "answer": "",
            "citations": [],
            "final_response": "",
            "retry_count": 0,
        }

        result = compiled_graph.invoke(initial_state)
        save_history(session_id, result["messages"])

        return AskResponse(
            answer=result["final_response"],
            citations=result["citations"],
            session_id=session_id,
            rewritten_queries=result["rewritten_queries"],
        )
    except Exception as e:
        logger.exception("Error processing /ask request")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    vs_status = "connected" if health_check() else "error"
    return HealthResponse(status="ok", vector_store=vs_status)


@router.get("/history/{session_id}", response_model=HistoryResponse)
async def get_session_history(session_id: str) -> HistoryResponse:
    history = get_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            messages.append(HistoryMessage(role="human", content=msg.content))
        elif isinstance(msg, AIMessage):
            messages.append(HistoryMessage(role="ai", content=msg.content))

    return HistoryResponse(
        session_id=session_id,
        message_count=len(messages),
        messages=messages,
    )
