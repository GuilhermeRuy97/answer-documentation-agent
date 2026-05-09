import logging
import uuid
from typing import Dict, List

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# Production: replace with Redis using langchain_community.chat_message_histories.RedisChatMessageHistory
_sessions: Dict[str, List[BaseMessage]] = {}


def get_or_create_session(session_id: str = None) -> str:
    if session_id is None or session_id not in _sessions:
        session_id = session_id or str(uuid.uuid4())
        _sessions[session_id] = []
    return session_id


def get_history(session_id: str) -> List[BaseMessage]:
    return _sessions.get(session_id, [])


def save_history(session_id: str, messages: List[BaseMessage]) -> None:
    _sessions[session_id] = messages


def get_all_sessions() -> Dict[str, int]:
    return {sid: len(msgs) for sid, msgs in _sessions.items()}
