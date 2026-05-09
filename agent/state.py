import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    session_id: str
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    rewritten_queries: List[str]
    retrieved_chunks: List[dict]
    relevance_score: float
    answer: str
    citations: List[dict]
    final_response: str
    retry_count: int
