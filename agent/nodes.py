import json
import logging
import os

import anthropic
from langchain_core.messages import AIMessage, HumanMessage

from agent.state import AgentState

logger = logging.getLogger(__name__)

MAX_RETRY_COUNT = int(os.getenv("MAX_RETRY_COUNT", "2"))

client = anthropic.Anthropic()


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


def rewrite_query(state: AgentState) -> dict:
    """HyDE: generate hypothetical answer paragraphs to use as embedding queries.

    Hypothetical answers are more similar to real document chunks than questions are,
    so they tend to retrieve better matches.
    """
    query = state["query"]

    history = state.get("messages", [])
    history_text = ""
    if history:
        lines = []
        for msg in history[-4:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            lines.append(f"{role}: {msg.content[:300]}")
        history_text = "\nConversation so far:\n" + "\n".join(lines) + "\n"

    prompt = (
        f"{history_text}"
        f"User question: {query}\n\n"
        "Write 2 short hypothetical paragraphs (3-5 sentences each) that would directly "
        "answer this question, written in the style of Anthropic's prompt engineering "
        "documentation. These paragraphs will be used to find similar real documentation, "
        "so phrase them as concrete factual statements about prompt engineering. "
        "If the question is a follow-up (e.g. 'give me an example'), use the conversation "
        "context to make each paragraph self-contained.\n\n"
        "Return ONLY a JSON array of 2 strings (each string a paragraph). No explanation."
    )
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        variants = json.loads(_strip_json_fence(response.content[0].text))
        logger.info(f"Generated {len(variants)} hypothetical answers (HyDE)")
        # Keep original query first so reranker can also retrieve via the actual question
        return {"rewritten_queries": [query] + variants}
    except Exception:
        logger.exception("Failed to parse HyDE variants")
        return {"rewritten_queries": [query]}


def grade_relevance(state: AgentState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"relevance_score": 0.0, "retry_count": state["retry_count"] + 1}

    # Prefer rerank_score when available (more accurate than cosine similarity)
    if "rerank_score" in chunks[0]:
        mean_score = sum(c["rerank_score"] for c in chunks) / len(chunks)
        logger.info(f"Mean rerank score: {mean_score:.3f} from {len(chunks)} chunks")
    else:
        mean_score = sum(c["similarity"] for c in chunks) / len(chunks)
        logger.info(f"Mean similarity: {mean_score:.3f} from {len(chunks)} chunks")

    return {"relevance_score": mean_score, "retry_count": state["retry_count"] + 1}


def generate_answer(state: AgentState) -> dict:
    chunks = state.get("retrieved_chunks", [])

    # Number chunks so the model can cite them with [N] markers inline
    context_parts = []
    for i, c in enumerate(chunks, start=1):
        context_parts.append(
            f"[{i}] SOURCE: {c['source_url']}\nTITLE: {c.get('page_title', 'Untitled')}\nCONTENT: {c['content']}"
        )
    context = "\n\n".join(context_parts) if context_parts else "(no documentation retrieved)"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=(
            "You are an expert on Anthropic prompt engineering. Answer questions ONLY using "
            "the provided documentation context. Be specific.\n\n"
            "CITATION RULES:\n"
            "- Each context chunk is numbered [1], [2], etc.\n"
            "- Insert the relevant marker [N] inline immediately after each fact you draw from chunk N.\n"
            "- Use only the numbers actually present in the context. Never invent citations.\n"
            "- If multiple chunks support a claim, cite all of them, e.g. 'XML tags improve clarity [1][3].'\n"
            "- If the documentation does not contain the answer, say so directly without citations."
        ),
        messages=[
            {"role": "user", "content": f"Documentation:\n{context}\n\nQuestion: {state['query']}"}
        ],
    )
    answer = response.content[0].text
    logger.info(f"Generated answer ({len(answer)} chars)")

    return {
        "answer": answer,
        "messages": [HumanMessage(content=state["query"]), AIMessage(content=answer)],
    }
