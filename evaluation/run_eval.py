"""Evaluation runner.

With LANGCHAIN_API_KEY set, runs a LangSmith experiment (dataset synced
automatically) with five evaluators. Without it, runs a local loop.
Both paths print a summary table with per-metric means and latency percentiles.
"""

import json
import logging
import os
import statistics
import time
from typing import List

from dotenv import load_dotenv

load_dotenv()

from agent.graph import compiled_graph
from core.config import get_settings
from core.logging import setup_logging
from evaluation.evaluators import (
    ls_answer_relevance,
    ls_citation_quality,
    ls_faithfulness,
    ls_relevance,
    ls_retrieval_hit_rate,
    run_single_evaluation,
)

setup_logging()
logger = logging.getLogger(__name__)

DATASET_NAME = "anthropic-prompt-engineering-eval-v2"

JUDGE_METRICS = ("relevance", "faithfulness", "citation_quality", "answer_relevance")


def load_dataset(path: str = "evaluation/dataset.json") -> List[dict]:
    """Load the local Q&A dataset, skipping unfilled entries.

    Args:
        path: Path to dataset.json.

    Returns:
        Valid dataset entries.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    entries = data["questions"] if isinstance(data, dict) else data
    valid = []
    for entry in entries:
        if entry.get("answer") == "FILL_IN_AFTER_INGESTION":
            logger.warning(f"Skipping unfilled entry: {entry.get('question', '')[:60]}")
            continue
        valid.append(entry)
    return valid


def run_agent(question: str) -> dict:
    """Run the compiled graph for one evaluation question.

    Args:
        question: Dataset question.

    Returns:
        Final graph state, plus "latency_s".
    """
    initial_state = {
        "session_id": f"eval-{int(time.time() * 1000)}",
        "messages": [],
        "query": question,
        "summary": "",
        "rewritten_queries": [],
        "retrieved_chunks": [],
        "relevance_score": 0.0,
        "answer": "",
        "citations": [],
        "final_response": "",
        "retry_count": 0,
        "error": "",
    }
    started = time.perf_counter()
    result = compiled_graph.invoke(initial_state, {"tags": ["evaluation"]})
    result["latency_s"] = time.perf_counter() - started
    return result


def _target(inputs: dict) -> dict:
    """Adapter for langsmith.evaluate(): example inputs -> agent outputs.

    Args:
        inputs: LangSmith example inputs ({"question": ...}).

    Returns:
        Agent outputs.
    """
    return run_agent(inputs["question"])


def _sync_dataset(client, entries: List[dict]) -> str:
    """Create the LangSmith dataset if missing.

    Args:
        client: LangSmith client.
        entries: Local dataset entries.

    Returns:
        Dataset name.
    """
    try:
        client.read_dataset(dataset_name=DATASET_NAME)
        logger.info(f"LangSmith dataset already exists: '{DATASET_NAME}'")
    except Exception:
        ls_dataset = client.create_dataset(
            DATASET_NAME,
            description="Anthropic prompt engineering RAG evaluation - 15 Q&A pairs",
        )
        for entry in entries:
            client.create_example(
                inputs={"question": entry["question"]},
                outputs={
                    "answer": entry.get("answer", ""),
                    "source_url": entry.get("source_url", ""),
                    "citations": entry.get("citations", []),
                },
                dataset_id=ls_dataset.id,
            )
        logger.info(f"Created LangSmith dataset '{DATASET_NAME}' with {len(entries)} examples")
    return DATASET_NAME


def _percentile(values: List[float], pct: float) -> float:
    """Compute a simple nearest-rank percentile.

    Args:
        values: Sample values.
        pct: Percentile in [0, 100].

    Returns:
        Percentile value, or 0.0 for empty input.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def _print_summary(results: List[dict], latencies: List[float] | None = None) -> None:
    """Print the metric summary table.

    Args:
        results: Per-question metric dicts.
        latencies: Optional per-question latencies in seconds.
    """
    rows = []
    judge_means = []
    for metric in JUDGE_METRICS:
        scores = [r[metric] for r in results if metric in r]
        mean = statistics.mean(scores) if scores else 0.0
        low = min(scores) if scores else 0
        judge_means.append(mean)
        rows.append((metric, f"{mean:.1f}/5", f"min {low}"))

    hits = [r["retrieval_hit_rate"] for r in results if "retrieval_hit_rate" in r]
    if hits:
        rows.append(("retrieval_hit_rate", f"{statistics.mean(hits) * 100:.0f}%", f"{sum(hits)}/{len(hits)}"))

    overall = statistics.mean(judge_means) if judge_means else 0.0
    rows.append(("overall (judges)", f"{overall:.1f}/5", ""))

    if latencies:
        rows.append(("latency p50", f"{_percentile(latencies, 50):.1f}s", ""))
        rows.append(("latency p95", f"{_percentile(latencies, 95):.1f}s", ""))

    width = max(len(r[0]) for r in rows) + 2
    print("\n" + "=" * (width + 22))
    print(f"{'Metric':<{width}}{'Score':<12}{'Detail'}")
    print("-" * (width + 22))
    for name, score, detail in rows:
        print(f"{name:<{width}}{score:<12}{detail}")
    print("=" * (width + 22))


def main():
    """Entry point: LangSmith experiment when configured, local loop otherwise."""
    dataset = load_dataset()

    if os.getenv("LANGCHAIN_API_KEY"):
        _run_with_langsmith(dataset)
    else:
        logger.warning("LANGCHAIN_API_KEY not set - running local evaluation only")
        _run_local(dataset)


def _run_with_langsmith(dataset: List[dict]) -> None:
    """Run the evaluation as a LangSmith experiment.

    Args:
        dataset: Local dataset entries.
    """
    from langsmith import Client
    from langsmith.evaluation import evaluate

    settings = get_settings()
    client = Client()
    dataset_name = _sync_dataset(client, dataset)

    logger.info(f"Starting LangSmith evaluation against dataset '{dataset_name}'")
    results = evaluate(
        _target,
        data=dataset_name,
        evaluators=[
            ls_relevance,
            ls_faithfulness,
            ls_citation_quality,
            ls_answer_relevance,
            ls_retrieval_hit_rate,
        ],
        experiment_prefix="anthropic-rag",
        metadata={
            "model": settings.generation_model,
            "judge_model": settings.judge_model,
            "embedding_model": settings.embedding_model,
            "rerank_model": settings.rerank_model,
            "retrieval_top_k": settings.retrieval_top_k,
            "rerank_top_k": settings.rerank_top_k,
            "relevance_threshold": settings.relevance_threshold,
            "recall_threshold": settings.recall_threshold,
            "max_retry_count": settings.max_retry_count,
            "hybrid_search": settings.use_hybrid_search,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
        max_concurrency=1,
    )

    logger.info("Evaluation complete. View results in the LangSmith UI:")
    logger.info(f"  Datasets & Testing -> '{dataset_name}' -> Experiments tab")

    local_results = []
    latencies = []
    for r in results:
        eval_results = r.get("evaluation_results")
        if eval_results is None:
            continue
        result_list = getattr(eval_results, "results", []) or []
        scores = {ev.key: ev.score for ev in result_list if ev.score is not None}
        if scores:
            local_results.append(scores)
        run_output = r.get("run")
        outputs = getattr(run_output, "outputs", None) or {}
        if "latency_s" in outputs:
            latencies.append(outputs["latency_s"])

    if local_results:
        _print_summary(local_results, latencies or None)


def _run_local(dataset: List[dict]) -> None:
    """Run the evaluation loop locally, without LangSmith.

    Args:
        dataset: Local dataset entries.
    """
    total = len(dataset)
    results = []
    latencies = []

    for i, entry in enumerate(dataset):
        question = entry["question"]
        agent_output = run_agent(question)
        latencies.append(agent_output.get("latency_s", 0.0))
        scores = run_single_evaluation(question, agent_output, expected_url=entry.get("source_url", ""))
        scores["question"] = question
        results.append(scores)
        print(f"[{i + 1}/{total}] {question[:50]}... ({agent_output.get('latency_s', 0):.1f}s)")

    _print_summary(results, latencies)


if __name__ == "__main__":
    main()
