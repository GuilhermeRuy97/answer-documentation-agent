"""Evaluators for the RAG agent.

LLM-as-judge (Claude Opus): relevance, faithfulness, citation_quality,
answer_relevance. Deterministic: retrieval_hit_rate (did the expected source
URL appear in the retrieved chunks?).
Each judge metric is scored 1-5; hit rate is 0/1.
"""

import json
import logging
import re
from typing import Optional
from urllib.parse import urlparse

import anthropic

from core.config import get_settings

logger = logging.getLogger(__name__)

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    """Return the lazily-initialized judge client.

    Returns:
        Anthropic client for evaluation calls.

    Raises:
        RuntimeError: If ANTHROPIC_API_KEY is not configured.
    """
    global _client
    if _client is not None:
        return _client
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY must be set to run evaluators")
    _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _parse_score(raw: str) -> int:
    """Extract an integer score 1-5 from any response shape Claude may return.

    Handles: {"score": 4, ...}, bare ints, "Score: 4" prose, and fenced JSON.

    Args:
        raw: Raw judge response text.

    Returns:
        Integer score 1-5.

    Raises:
        ValueError: When no score can be extracted.
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

    match = re.search(r"\b([1-5])\b", text)
    if match:
        return int(match.group(1))

    raise ValueError(f"Could not extract score from: {raw[:100]}")


def _judge(prompt: str) -> int:
    """Run one judge call with the JSON-object prefill and parse the score.

    Args:
        prompt: Judge prompt.

    Returns:
        Integer score 1-5.
    """
    response = _get_client().messages.create(
        model=get_settings().judge_model,
        max_tokens=256,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "{"},
        ],
    )
    return _parse_score("{" + response.content[0].text)


def relevance_evaluator(question: str, context: str) -> int:
    """Judge whether retrieved chunks are relevant to the question.

    Args:
        question: User question.
        context: Concatenated retrieved chunk contents.

    Returns:
        Score 1-5 (0 on judge failure).
    """
    prompt = (
        "Rate 1-5: Are these retrieved chunks relevant to the question?\n"
        "1=Not relevant, 5=Perfectly relevant.\n"
        f"<question>{question}</question>\n"
        f"<context>{context}</context>\n"
        'Return ONLY a JSON: {"score": N, "reason": "..."}'
    )
    try:
        return _judge(prompt)
    except Exception:
        logger.exception("relevance_evaluator failed")
        return 0


def faithfulness_evaluator(answer: str, context: str) -> int:
    """Judge whether the answer only uses facts from the context.

    Args:
        answer: Generated answer.
        context: Concatenated retrieved chunk contents.

    Returns:
        Score 1-5 (0 on judge failure).
    """
    prompt = (
        "Rate 1-5: Does this answer ONLY use facts from the context?\n"
        "1=Many hallucinations, 5=Fully grounded.\n"
        f"<answer>{answer}</answer>\n"
        f"<context>{context}</context>\n"
        'Return ONLY a JSON: {"score": N, "reason": "..."}'
    )
    try:
        return _judge(prompt)
    except Exception:
        logger.exception("faithfulness_evaluator failed")
        return 0


def citation_quality_evaluator(response: str, citations: list) -> int:
    """Judge citation accuracy and linkage.

    Args:
        response: Final formatted answer.
        citations: Citation dicts.

    Returns:
        Score 1-5 (0 on judge failure).
    """
    prompt = (
        "Rate 1-5: Are citations accurate, present, and linked to relevant sources?\n"
        "1=No citations or wrong, 5=Perfect citations.\n"
        f"<response>{response}</response>\n"
        f"<citations>{json.dumps(citations)}</citations>\n"
        'Return ONLY a JSON: {"score": N, "reason": "..."}'
    )
    try:
        return _judge(prompt)
    except Exception:
        logger.exception("citation_quality_evaluator failed")
        return 0


def answer_relevance_evaluator(question: str, answer: str) -> int:
    """Judge whether the answer actually addresses the question asked.

    Complements faithfulness: an answer can be perfectly grounded yet
    off-topic; this metric catches that failure mode.

    Args:
        question: User question.
        answer: Generated answer.

    Returns:
        Score 1-5 (0 on judge failure).
    """
    prompt = (
        "Rate 1-5: Does this answer directly and completely address the question?\n"
        "1=Off-topic or evasive, 5=Directly and completely answers it.\n"
        f"<question>{question}</question>\n"
        f"<answer>{answer}</answer>\n"
        'Return ONLY a JSON: {"score": N, "reason": "..."}'
    )
    try:
        return _judge(prompt)
    except Exception:
        logger.exception("answer_relevance_evaluator failed")
        return 0


def _normalize_url(url: str) -> str:
    """Normalize a URL for hit-rate comparison (host + path, no trailing slash).

    Args:
        url: Raw URL.

    Returns:
        Normalized "host/path" string.
    """
    try:
        parsed = urlparse(url)
        return (parsed.netloc + parsed.path).rstrip("/").lower()
    except ValueError:
        return url.rstrip("/").lower()


def retrieval_hit_rate_evaluator(expected_url: str, chunks: list) -> int:
    """Deterministic check: did retrieval surface the expected source page?

    Compares by path tail so old docs.anthropic.com URLs still match their
    platform.claude.com equivalents.

    Args:
        expected_url: Ground-truth source URL from the dataset.
        chunks: Retrieved chunk dicts.

    Returns:
        1 when the expected page was retrieved, 0 otherwise.
    """
    if not expected_url:
        return 0
    expected_tail = _normalize_url(expected_url).split("/")[-1]
    if not expected_tail:
        return 0
    for chunk in chunks:
        retrieved = _normalize_url(chunk.get("source_url", ""))
        if retrieved.endswith(expected_tail):
            return 1
    return 0


def run_single_evaluation(question: str, agent_output: dict, expected_url: str = "") -> dict:
    """Score one agent run across all metrics.

    Args:
        question: User question.
        agent_output: Final graph state for the question.
        expected_url: Ground-truth source URL (enables hit rate).

    Returns:
        Dict of metric name -> score.
    """
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
        "answer_relevance": answer_relevance_evaluator(question, answer),
        "retrieval_hit_rate": retrieval_hit_rate_evaluator(
            expected_url, agent_output.get("retrieved_chunks", [])
        ),
    }


# ---------------------------------------------------------------------------
# LangSmith-compatible evaluators: (inputs, outputs[, reference_outputs]) ->
# {"key": str, "score": number}
# ---------------------------------------------------------------------------

def ls_relevance(inputs: dict, outputs: dict) -> dict:
    question = inputs.get("question", "")
    chunks = outputs.get("retrieved_chunks", [])
    context = "\n---\n".join(c.get("content", "") for c in chunks)
    return {"key": "relevance", "score": relevance_evaluator(question, context)}


def ls_faithfulness(inputs: dict, outputs: dict) -> dict:
    chunks = outputs.get("retrieved_chunks", [])
    context = "\n---\n".join(c.get("content", "") for c in chunks)
    answer = outputs.get("answer", "")
    return {"key": "faithfulness", "score": faithfulness_evaluator(answer, context)}


def ls_citation_quality(inputs: dict, outputs: dict) -> dict:
    final_response = outputs.get("final_response", outputs.get("answer", ""))
    citations = outputs.get("citations", [])
    return {"key": "citation_quality", "score": citation_quality_evaluator(final_response, citations)}


def ls_answer_relevance(inputs: dict, outputs: dict) -> dict:
    question = inputs.get("question", "")
    answer = outputs.get("answer", "")
    return {"key": "answer_relevance", "score": answer_relevance_evaluator(question, answer)}


def ls_retrieval_hit_rate(inputs: dict, outputs: dict, reference_outputs: dict | None = None) -> dict:
    expected_url = (reference_outputs or {}).get("source_url", "")
    chunks = outputs.get("retrieved_chunks", [])
    return {"key": "retrieval_hit_rate", "score": retrieval_hit_rate_evaluator(expected_url, chunks)}
