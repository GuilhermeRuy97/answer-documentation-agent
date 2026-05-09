# ───────────────────────────────────────────────
# FILE 13: api/schemas.py
# ───────────────────────────────────────────────
"""
Create `api/schemas.py`.

Pydantic v2 models for FastAPI request and response validation.

Requirements:
- Use Pydantic v2 syntax (no class Config, use model_config)

- class Citation(BaseModel):
    title: str
    url: str
    snippet: str

- class AskRequest(BaseModel):
    question: str
    session_id: str | None = None
    # Validation: question must be non-empty, max 1000 chars

- class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    session_id: str
    rewritten_queries: List[str]

- class HealthResponse(BaseModel):
    status: str
    vector_store: str
    model: str = "claude-sonnet-4-5"

- class HistoryMessage(BaseModel):
    role: str  # "human" or "ai"
    content: str

- class HistoryResponse(BaseModel):
    session_id: str
    message_count: int
    messages: List[HistoryMessage]
"""