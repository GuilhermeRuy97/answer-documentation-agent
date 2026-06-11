"""LangGraph agent state definition."""

import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    session_id: str
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    # Rolling summary of older conversation turns (long-term context).
    summary: str
    rewritten_queries: List[str]
    retrieved_chunks: List[dict]
    relevance_score: float
    answer: str
    citations: List[dict]
    final_response: str
    # Number of search->grade cycles completed (1 = initial search, >1 = retries).
    retry_count: int
    # Non-empty when a node failed and the answer is a graceful fallback.
    error: str
