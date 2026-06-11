"""Session memory.

Short-term: in-process cache with TTL eviction (fast path for active chats).
Long-term: chat_messages / chat_sessions tables in Supabase, so conversations
and rolling summaries survive restarts. The DB is best-effort: when Supabase is
not configured (tests, offline dev) everything still works in-memory.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class _SessionEntry:
    messages: List[BaseMessage] = field(default_factory=list)
    summary: str = ""
    last_access: float = field(default_factory=time.time)
    loaded_from_db: bool = False


_cache: Dict[str, _SessionEntry] = {}


def _db_enabled() -> bool:
    """Check whether Supabase-backed persistence is configured and enabled.

    Returns:
        True if sessions should be persisted to Supabase.
    """
    settings = get_settings()
    return settings.persist_sessions and bool(settings.supabase_url) and bool(settings.supabase_service_key)


def _evict_expired() -> None:
    """Drop cache entries idle longer than the session TTL (DB copy remains)."""
    ttl = get_settings().session_ttl_seconds
    now = time.time()
    expired = [sid for sid, entry in _cache.items() if now - entry.last_access > ttl]
    for sid in expired:
        del _cache[sid]
    if expired:
        logger.info(f"Evicted {len(expired)} idle sessions from memory cache")


def _load_from_db(session_id: str) -> Optional[_SessionEntry]:
    """Load a session's messages and summary from Supabase.

    Args:
        session_id: Session to load.

    Returns:
        Populated entry, or None when persistence is disabled or loading fails.
    """
    if not _db_enabled():
        return None
    try:
        from retrieval.vector_store import get_client

        client = get_client()
        msg_result = (
            client.table("chat_messages")
            .select("role, content")
            .eq("session_id", session_id)
            .order("id")
            .execute()
        )
        messages: List[BaseMessage] = []
        for row in msg_result.data or []:
            if row["role"] == "human":
                messages.append(HumanMessage(content=row["content"]))
            else:
                messages.append(AIMessage(content=row["content"]))

        summary = ""
        sess_result = (
            client.table("chat_sessions").select("summary").eq("session_id", session_id).execute()
        )
        if sess_result.data:
            summary = sess_result.data[0].get("summary") or ""

        if not messages and not summary:
            return None
        logger.info(f"Loaded session {session_id} from DB ({len(messages)} messages)")
        return _SessionEntry(messages=messages, summary=summary, loaded_from_db=True)
    except Exception:
        logger.warning(f"Could not load session {session_id} from DB", exc_info=True)
        return None


def _ensure(session_id: str) -> _SessionEntry:
    """Get or create the cache entry for a session, loading from DB on miss.

    Args:
        session_id: Session identifier.

    Returns:
        The (possibly new) cache entry.
    """
    _evict_expired()
    entry = _cache.get(session_id)
    if entry is None:
        entry = _load_from_db(session_id) or _SessionEntry()
        _cache[session_id] = entry
    entry.last_access = time.time()
    return entry


def get_or_create_session(session_id: str | None = None) -> str:
    """Resolve a session id, generating a new UUID when none is provided.

    Args:
        session_id: Optional client-provided session id.

    Returns:
        The resolved session id.
    """
    session_id = session_id or str(uuid.uuid4())
    _ensure(session_id)
    return session_id


def get_history(session_id: str) -> List[BaseMessage]:
    """Return the in-context message history for a session.

    Args:
        session_id: Session identifier.

    Returns:
        Copy of the session's message list (may be empty).
    """
    return list(_ensure(session_id).messages)


def get_summary(session_id: str) -> str:
    """Return the rolling conversation summary for a session.

    Args:
        session_id: Session identifier.

    Returns:
        Summary text; empty string when none exists.
    """
    return _ensure(session_id).summary


def append_messages(session_id: str, messages: List[BaseMessage]) -> None:
    """Append new turns to the session (cache + best-effort DB insert).

    Args:
        session_id: Session identifier.
        messages: New messages to append (typically [HumanMessage, AIMessage]).
    """
    if not messages:
        return
    entry = _ensure(session_id)
    entry.messages.extend(messages)

    if not _db_enabled():
        return
    try:
        from retrieval.vector_store import get_client

        client = get_client()
        rows = [
            {
                "session_id": session_id,
                "role": "human" if isinstance(m, HumanMessage) else "ai",
                "content": str(m.content),
            }
            for m in messages
        ]
        client.table("chat_messages").insert(rows).execute()
        client.table("chat_sessions").upsert(
            {"session_id": session_id, "summary": entry.summary},
            on_conflict="session_id",
        ).execute()
    except Exception:
        logger.warning(f"Could not persist messages for session {session_id}", exc_info=True)


def save_summary(session_id: str, summary: str) -> None:
    """Store the rolling summary (cache + best-effort DB upsert).

    Args:
        session_id: Session identifier.
        summary: Updated rolling summary text.
    """
    entry = _ensure(session_id)
    entry.summary = summary

    if not _db_enabled():
        return
    try:
        from retrieval.vector_store import get_client

        get_client().table("chat_sessions").upsert(
            {"session_id": session_id, "summary": summary},
            on_conflict="session_id",
        ).execute()
    except Exception:
        logger.warning(f"Could not persist summary for session {session_id}", exc_info=True)


def trim_history(session_id: str, keep: int) -> None:
    """Trim the in-context history to the most recent messages.

    Only the cache is trimmed; the DB keeps the full transcript (long-term log).

    Args:
        session_id: Session identifier.
        keep: Number of most recent messages to keep in context.
    """
    entry = _ensure(session_id)
    if len(entry.messages) > keep:
        entry.messages = entry.messages[-keep:]


def clear_history(session_id: str) -> bool:
    """Delete a session everywhere (cache and DB).

    Args:
        session_id: Session identifier.

    Returns:
        True if a session existed in the cache (DB delete is best effort).
    """
    existed = session_id in _cache
    _cache.pop(session_id, None)

    if _db_enabled():
        try:
            from retrieval.vector_store import get_client

            client = get_client()
            client.table("chat_messages").delete().eq("session_id", session_id).execute()
            client.table("chat_sessions").delete().eq("session_id", session_id).execute()
            existed = True
        except Exception:
            logger.warning(f"Could not delete session {session_id} from DB", exc_info=True)

    return existed
