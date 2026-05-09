import json
import logging
import os

import anthropic

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic()
_MODEL = "claude-opus-4-7"


def _judge(prompt: str) -> dict:
    response = _client.messages.create(
        model=_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.content[0].text)


def relevance_evaluator(question: str, context: str) -> int:
    prompt = (
        "Rate 1-5: Are these retrieved chunks relevant to the question?\n"
        "1=Not relevant, 5=Perfectly relevant.\n"
        f"Question: {question}\n"
        f"Context: {context}\n"
        'Return ONLY a JSON: {"score": N, "reason": "..."}'
    )
    try:
        result = _judge(prompt)
        return int(result["score"])
    except Exception:
        logger.exception("relevance_evaluator failed")
        return 0


def faithfulness_evaluator(answer: str, context: str) -> int:
    prompt = (
        "Rate 1-5: Does this answer ONLY use facts from the context?\n"
        "1=Many hallucinations, 5=Fully grounded.\n"
        f"Answer: {answer}\n"
        f"Context: {context}\n"
        'Return ONLY a JSON: {"score": N, "reason": "..."}'
    )
    try:
        result = _judge(prompt)
        return int(result["score"])
    except Exception:
        logger.exception("faithfulness_evaluator failed")
        return 0


def citation_quality_evaluator(response: str, citations: list) -> int:
    prompt = (
        "Rate 1-5: Are citations accurate, present, and linked to relevant sources?\n"
        "1=No citations or wrong, 5=Perfect citations.\n"
        f"Response: {response}\n"
        f"Citations: {json.dumps(citations)}\n"
        'Return ONLY a JSON: {"score": N, "reason": "..."}'
    )
    try:
        result = _judge(prompt)
        return int(result["score"])
    except Exception:
        logger.exception("citation_quality_evaluator failed")
        return 0


def run_single_evaluation(question: str, agent_output: dict) -> dict:
    context = "\n---\n".join(
        c.get("content", "") for c in agent_output.get("retrieved_chunks", [])
    )
    answer = agent_output.get("answer", "")
    final_response = agent_output.get("final_response", answer)
    citations = agent_output.get("citations", [])

    return {
        "relevance": relevance_evaluator(question, context),
        "faithfulness": faithfulness_evaluator(answer, context),
        "citation_quality": citation_quality_evaluator(final_response, citations),
    }
