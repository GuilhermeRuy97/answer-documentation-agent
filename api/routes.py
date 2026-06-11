"""API route handlers."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import compiled_graph
from agent.session import clear_history, get_history, get_or_create_session
from api.schemas import (
    AskRequest,
    AskResponse,
    DeleteHistoryResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    HistoryMessage,
    HistoryResponse,
)
from api.security import limiter, require_api_key
from core.config import get_settings
from retrieval.vector_store import health_check

logger = logging.getLogger(__name__)

# Authenticated routes (auth disabled when API_KEYS is empty).
router = APIRouter(dependencies=[Depends(require_api_key)])
# Public routes: /health stays open for load balancers and container healthchecks.
public_router = APIRouter()


@router.post("/ask", response_model=AskResponse)
@limiter.limit(get_settings().rate_limit_ask)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    """Answer a question with the RAG agent.

    Args:
        request: Raw request (required by the rate limiter).
        body: Validated ask payload.

    Returns:
        Answer with citations, session id, query variants, and trace id.
    """
    settings = get_settings()
    # Session resolution can hit Supabase on a cache miss; keep it off the event loop.
    session_id = await run_in_threadpool(get_or_create_session, body.session_id)
    run_id = uuid.uuid4()

    initial_state = {
        "session_id": session_id,
        "messages": [],
        "query": body.question,
        "summary": "",
        "rewritten_queries": [],
        "retrieved_chunks": [],
        "relevance_score": 0.0,
        "answer": "",
        "citations": [],
        "final_response": "",
        "retry_count": 0,
        "error": "",
    }
    config = {
        "run_id": run_id,
        "tags": ["api", "ask"],
        "metadata": {
            "session_id": session_id,
            "model": settings.generation_model,
            "retrieval_top_k": settings.retrieval_top_k,
            "relevance_threshold": settings.relevance_threshold,
            "hybrid_search": settings.use_hybrid_search,
        },
    }

    try:
        # The graph is synchronous; run it off the event loop.
        result = await run_in_threadpool(compiled_graph.invoke, initial_state, config)
    except Exception:
        logger.exception("Error processing /ask request")
        # Sanitized: never leak internal exception details to clients.
        raise HTTPException(status_code=500, detail="Internal server error")

    return AskResponse(
        answer=result["final_response"],
        citations=result["citations"],
        session_id=session_id,
        rewritten_queries=result["rewritten_queries"],
        trace_id=str(run_id),
    )


@public_router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report API and vector store health.

    Returns:
        Health status payload.
    """
    vs_status = "connected" if await run_in_threadpool(health_check) else "error"
    return HealthResponse(status="ok", vector_store=vs_status)


@router.get("/history/{session_id}", response_model=HistoryResponse)
async def get_session_history(session_id: str) -> HistoryResponse:
    """Return the message history for a session.

    Args:
        session_id: Session identifier.

    Returns:
        Session transcript.

    Raises:
        HTTPException: 404 when the session has no messages.
    """
    history = await run_in_threadpool(get_history, session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            messages.append(HistoryMessage(role="human", content=str(msg.content)))
        elif isinstance(msg, AIMessage):
            messages.append(HistoryMessage(role="ai", content=str(msg.content)))

    return HistoryResponse(
        session_id=session_id,
        message_count=len(messages),
        messages=messages,
    )


@router.delete("/history/{session_id}", response_model=DeleteHistoryResponse)
async def delete_session_history(session_id: str) -> DeleteHistoryResponse:
    """Delete a session's history (memory cache and persisted rows).

    Args:
        session_id: Session identifier.

    Returns:
        Deletion result.
    """
    deleted = await run_in_threadpool(clear_history, session_id)
    return DeleteHistoryResponse(session_id=session_id, deleted=deleted)


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    """Record user feedback (thumbs up/down) against a LangSmith trace.

    Falls back to local logging when LangSmith is not configured.

    Args:
        body: Feedback payload with the trace_id returned by /ask.

    Returns:
        Status payload.
    """
    if not get_settings().langchain_api_key:
        logger.info(f"Feedback (local only): trace={body.trace_id} score={body.score} comment={body.comment}")
        return FeedbackResponse(status="logged_locally")

    try:
        from langsmith import Client

        def _send() -> None:
            Client().create_feedback(
                run_id=body.trace_id,
                key="user_rating",
                score=body.score,
                comment=body.comment,
            )

        await run_in_threadpool(_send)
        return FeedbackResponse(status="recorded")
    except Exception:
        logger.exception("Failed to record feedback in LangSmith")
        raise HTTPException(status_code=502, detail="Failed to record feedback")
