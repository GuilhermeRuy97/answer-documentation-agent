"""LangGraph node functions.

All nodes are pure functions (state -> partial state update) with node-level
error handling: a failed Claude call degrades gracefully instead of crashing
the request. The Anthropic client is lazy so the module imports without keys.
"""

import json
import logging
import time
from typing import List, Optional

import anthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent import session as session_store
from agent.prompts import (
    ANSWER_SYSTEM,
    REWRITE_PREFILL,
    REWRITE_SYSTEM,
    SUMMARY_SYSTEM,
    build_answer_prompt,
    build_rewrite_prompt,
    build_summary_prompt,
)
from agent.state import AgentState
from core.config import get_settings

logger = logging.getLogger(__name__)

_client: Optional[anthropic.Anthropic] = None


def get_anthropic_client() -> anthropic.Anthropic:
    """Return the lazily-initialized Anthropic client.

    Returns:
        The shared Anthropic client.

    Raises:
        RuntimeError: If ANTHROPIC_API_KEY is not configured.
    """
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY must be set to call Claude")

    _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _call_claude(
    system: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    prefill: str | None = None,
) -> str:
    """Call Claude and return the (prefill-prepended) text, logging usage/latency.

    Args:
        system: System prompt.
        user_prompt: User message content.
        max_tokens: Generation cap.
        temperature: Sampling temperature.
        prefill: Optional assistant-turn prefill for structured output.

    Returns:
        Generated text, including the prefill prefix when provided.
    """
    settings = get_settings()
    messages = [{"role": "user", "content": user_prompt}]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    started = time.perf_counter()
    response = get_anthropic_client().messages.create(
        model=settings.generation_model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=messages,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    usage = response.usage
    logger.info(
        f"Claude call: {elapsed_ms:.0f}ms, in={usage.input_tokens} out={usage.output_tokens} tokens"
    )
    text = response.content[0].text
    return (prefill or "") + text


def _format_turns(messages: List[BaseMessage], limit: int, max_chars: int = 300) -> str:
    """Format the last N conversation turns as plain text.

    Args:
        messages: Conversation messages.
        limit: Max number of trailing messages to include.
        max_chars: Per-message truncation.

    Returns:
        "User: ...\nAssistant: ..." formatted text.
    """
    lines = []
    for msg in messages[-limit:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {str(msg.content)[:max_chars]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def load_memory(state: AgentState) -> dict:
    """Load conversation history and rolling summary into the graph state.

    Args:
        state: Current agent state (needs session_id).

    Returns:
        Partial update with messages and summary.
    """
    session_id = state["session_id"]
    history = session_store.get_history(session_id)
    summary = session_store.get_summary(session_id)
    logger.info(f"Loaded memory for session {session_id}: {len(history)} messages, summary={'yes' if summary else 'no'}")
    return {"messages": history, "summary": summary}


def rewrite_query(state: AgentState) -> dict:
    """Generate retrieval query variants: HyDE paragraphs + a keyword query.

    Hypothetical answers embed closer to real documentation chunks than raw
    questions do; the keyword variant feeds the full-text leg of hybrid search.

    Args:
        state: Current agent state.

    Returns:
        Partial update with rewritten_queries (original question always first).
    """
    query = state["query"]
    history_text = _format_turns(state.get("messages", []), limit=4)
    summary = state.get("summary", "")

    prompt = build_rewrite_prompt(query, history_text, summary)
    try:
        raw = _call_claude(
            system=REWRITE_SYSTEM,
            user_prompt=prompt,
            max_tokens=512,
            temperature=0.7,
            prefill=REWRITE_PREFILL,
        )
        parsed = json.loads(raw)
        hyde = [v for v in parsed.get("hyde", []) if isinstance(v, str) and v.strip()]
        keywords = parsed.get("keywords", "")
        variants = hyde + ([keywords] if keywords else [])
        logger.info(f"Generated {len(hyde)} HyDE paragraphs + {1 if keywords else 0} keyword variant")
        return {"rewritten_queries": [query] + variants}
    except Exception:
        logger.exception("Query rewrite failed; falling back to original query only")
        return {"rewritten_queries": [query]}


def grade_relevance(state: AgentState) -> dict:
    """Score retrieval quality as the mean rerank (or cosine) score of the chunks.

    Also increments retry_count, which counts completed search->grade cycles.

    Args:
        state: Current agent state.

    Returns:
        Partial update with relevance_score and retry_count.
    """
    chunks = state.get("retrieved_chunks", [])
    cycles = state.get("retry_count", 0) + 1
    if not chunks:
        return {"relevance_score": 0.0, "retry_count": cycles}

    if "rerank_score" in chunks[0]:
        mean_score = sum(c["rerank_score"] for c in chunks) / len(chunks)
        logger.info(f"Mean rerank score: {mean_score:.3f} from {len(chunks)} chunks (cycle {cycles})")
    else:
        mean_score = sum(c.get("similarity", 0.0) for c in chunks) / len(chunks)
        logger.info(f"Mean similarity: {mean_score:.3f} from {len(chunks)} chunks (cycle {cycles})")

    return {"relevance_score": mean_score, "retry_count": cycles}


def generate_answer(state: AgentState) -> dict:
    """Generate the cited answer from the retrieved chunks.

    Degrades gracefully: on API failure, returns an apologetic answer and sets
    state["error"] instead of raising.

    Args:
        state: Current agent state.

    Returns:
        Partial update with answer, messages, and error.
    """
    chunks = state.get("retrieved_chunks", [])
    query = state["query"]
    prompt = build_answer_prompt(query, chunks, state.get("summary", ""))

    try:
        answer = _call_claude(
            system=ANSWER_SYSTEM,
            user_prompt=prompt,
            max_tokens=2048,
            temperature=0.2,
        )
        logger.info(f"Generated answer ({len(answer)} chars)")
        error = ""
    except Exception as e:
        logger.exception("Answer generation failed")
        answer = (
            "I ran into a problem while generating the answer. Please try again in a moment."
        )
        error = f"generate_answer: {type(e).__name__}"

    return {
        "answer": answer,
        "error": error,
        "messages": [HumanMessage(content=query), AIMessage(content=answer)],
    }


def save_memory(state: AgentState) -> dict:
    """Persist the new turn and maintain the rolling summary (long-term memory).

    When the in-context history grows past max_history_messages, the older half
    is folded into the rolling summary via Claude and trimmed from context.
    The full transcript always remains in the DB.

    Args:
        state: Current agent state.

    Returns:
        Partial update with summary (possibly refreshed).
    """
    settings = get_settings()
    session_id = state["session_id"]
    query = state["query"]
    answer = state.get("answer", "")
    summary = state.get("summary", "")

    # Skip persisting failed turns so a transient error doesn't pollute memory.
    if not state.get("error"):
        session_store.append_messages(
            session_id, [HumanMessage(content=query), AIMessage(content=answer)]
        )

    messages = session_store.get_history(session_id)
    if len(messages) <= settings.max_history_messages:
        return {"summary": summary}

    keep = settings.max_history_messages // 2
    older = messages[:-keep] if keep else messages
    try:
        turns_text = _format_turns(older, limit=len(older), max_chars=500)
        new_summary = _call_claude(
            system=SUMMARY_SYSTEM,
            user_prompt=build_summary_prompt(summary, turns_text),
            max_tokens=300,
            temperature=0.0,
        ).strip()
        session_store.save_summary(session_id, new_summary)
        session_store.trim_history(session_id, keep=keep)
        logger.info(f"Summarized {len(older)} older messages for session {session_id}")
        return {"summary": new_summary}
    except Exception:
        logger.exception("Summarization failed; keeping full history in context")
        return {"summary": summary}
