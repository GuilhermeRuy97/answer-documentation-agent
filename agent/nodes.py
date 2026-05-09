import json
import logging
import os

import anthropic
from langchain_core.messages import AIMessage, HumanMessage

from agent.state import AgentState
from retrieval.retriever import retrieve

logger = logging.getLogger(__name__)

MAX_RETRY_COUNT = int(os.getenv("MAX_RETRY_COUNT", "2"))

client = anthropic.Anthropic()


def rewrite_query(state: AgentState) -> dict:
    query = state["query"]
    prompt = (
        f"Given this question: {query}\n"
        "Generate 2 alternative phrasings that might retrieve better results "
        "from a vector database of prompt engineering documentation.\n"
        "Return ONLY a JSON array of 2 strings. No explanation."
    )
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        variants = json.loads(response.content[0].text)
        logger.info(f"Generated {len(variants)} query variants")
        return {"rewritten_queries": [query] + variants}
    except Exception:
        logger.exception("Failed to parse query variants")
        return {"rewritten_queries": [query]}


def grade_relevance(state: AgentState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"relevance_score": 0.0, "retry_count": state["retry_count"] + 1}

    mean_score = sum(c["similarity"] for c in chunks) / len(chunks)
    logger.info(f"Relevance score: {mean_score:.3f} from {len(chunks)} chunks")
    return {"relevance_score": mean_score}


def generate_answer(state: AgentState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    context = "\n".join(
        f"SOURCE: {c['source_url']}\nCONTENT: {c['content']}\n---" for c in chunks
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=(
            "You are an expert on Anthropic prompt engineering. Answer questions "
            "ONLY using the provided documentation context. Be specific and cite sources."
        ),
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['query']}"}
        ],
    )
    answer = response.content[0].text
    logger.info(f"Generated answer ({len(answer)} chars)")

    return {
        "answer": answer,
        "messages": [HumanMessage(content=state["query"]), AIMessage(content=answer)],
    }
