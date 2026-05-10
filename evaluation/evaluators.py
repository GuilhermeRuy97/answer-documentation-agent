import json
import logging
import os
import re

import anthropic

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic()
_MODEL = "claude-opus-4-7"


def _parse_score(raw: str) -> int:
    """Extract an integer score 1-5 from any of the response shapes Claude might return:
    - {"score": 4, "reason": "..."}  (the requested format)
    - 4                              (bare int)
    - "Score: 4"                     (prose with embedded number)
    - ```json\n{"score": 4}\n```     (markdown-fenced JSON)
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "score" in parsed:
            return int(parsed["score"])
        if isinstance(parsed, (int, float)):
            return int(parsed)
    except json.JSONDecodeError:
        pass

    # Last resort: regex for the first 1-5 in the text
    match = re.search(r"\b([1-5])\b", text)
    if match:
        return int(match.group(1))

    raise ValueError(f"Could not extract score from: {raw[:100]}")


def _judge(prompt: str) -> int:
    response = _client.messages.create(
        model=_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_score(response.content[0].text)


def relevance_evaluator(question: str, context: str) -> int:
    prompt = (
        "Rate 1-5: Are these retrieved chunks relevant to the question?\n"
        "1=Not relevant, 5=Perfectly relevant.\n"
        f"Question: {question}\n"
        f"Context: {context}\n"
        'Return ONLY a JSON: {"score": N, "reason": "..."}'
    )
    try:
        return _judge(prompt)
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
        return _judge(prompt)
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
        return _judge(prompt)
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
