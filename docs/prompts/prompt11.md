# ───────────────────────────────────────────────
# FILE 11: agent/session.py
# ───────────────────────────────────────────────
"""
Create `agent/session.py`.

Simple in-memory session store for multi-turn conversations.

Requirements:
- Import: uuid, logging, Dict, List, Any from typing
- Import: BaseMessage from langchain_core.messages

- Use a module-level dict: _sessions: Dict[str, List[BaseMessage]] = {}

- Define three functions:

  def get_or_create_session(session_id: str = None) -> str:
    - If session_id is None or not in _sessions, create new UUID and init empty list
    - Return the session_id

  def get_history(session_id: str) -> List[BaseMessage]:
    - Return _sessions.get(session_id, [])

  def save_history(session_id: str, messages: List[BaseMessage]) -> None:
    - _sessions[session_id] = messages

  def get_all_sessions() -> Dict[str, int]:
    - Returns {session_id: len(messages)} for all sessions
    - Useful for debugging

NOTE: This is in-memory only. Add a note: "Production: replace with Redis using
langchain_community.chat_message_histories.RedisChatMessageHistory"
"""
