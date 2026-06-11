"""Pydantic v2 request/response models."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from core.config import get_settings


class Citation(BaseModel):
    title: str
    url: str
    snippet: str


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

    @field_validator("question")
    @classmethod
    def question_must_be_valid(cls, v: str) -> str:
        """Validate and normalize the question.

        Args:
            v: Raw question string.

        Returns:
            Stripped question.

        Raises:
            ValueError: When empty or over the configured length limit.
        """
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        max_chars = get_settings().max_question_chars
        if len(v) > max_chars:
            raise ValueError(f"question must be {max_chars} characters or fewer")
        return v

    @field_validator("session_id")
    @classmethod
    def session_id_must_be_sane(cls, v: Optional[str]) -> Optional[str]:
        """Reject absurd session ids (defense in depth for the DB layer).

        Args:
            v: Raw session id or None.

        Returns:
            The validated session id.

        Raises:
            ValueError: When longer than 128 chars.
        """
        if v is not None and len(v) > 128:
            raise ValueError("session_id must be 128 characters or fewer")
        return v


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    session_id: str
    rewritten_queries: List[str]
    # LangSmith root run id for this request; pass back via POST /feedback.
    trace_id: str


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    model: str = Field(default_factory=lambda: get_settings().generation_model)


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    message_count: int
    messages: List[HistoryMessage]


class DeleteHistoryResponse(BaseModel):
    session_id: str
    deleted: bool


class FeedbackRequest(BaseModel):
    trace_id: str
    # 1.0 = thumbs up, 0.0 = thumbs down.
    score: float = Field(ge=0.0, le=1.0)
    comment: Optional[str] = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    status: str
