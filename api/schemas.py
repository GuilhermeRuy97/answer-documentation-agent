from typing import List

from pydantic import BaseModel, field_validator


class Citation(BaseModel):
    title: str
    url: str
    snippet: str


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None

    @field_validator("question")
    @classmethod
    def question_must_be_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        if len(v) > 1000:
            raise ValueError("question must be 1000 characters or fewer")
        return v


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    session_id: str
    rewritten_queries: List[str]


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    model: str = "claude-sonnet-4-6"


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    message_count: int
    messages: List[HistoryMessage]
