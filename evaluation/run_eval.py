import json
import logging
import os
import statistics
from typing import List

from dotenv import load_dotenv

load_dotenv()

from agent.graph import compiled_graph
from evaluation.evaluators import (
    ls_citation_quality,
    ls_faithfulness,
    ls_relevance,
    run_single_evaluation,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "anthropic-prompt-engineering-eval-v1"


def load_dataset(path: str = "evaluation/dataset.json") -> List[dict]:
    with open(path) as f:
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
    initial_state = {
        "session_id": "eval",
        "messages": [],
        "query": question,
        "rewritten_queries": [],
        "retrieved_chunks": [],
        "relevance_score": 0.0,
        "answer": "",
        "citations": [],
        "final_response": "",
        "retry_count": 0,
    }
    return compiled_graph.invoke(initial_state)


def _target(inputs: dict) -> dict:
    """Wrapper for langsmith.evaluate() — receives example inputs, returns agent outputs."""
    return run_agent(inputs["question"])


def _sync_dataset(client, entries: List[dict]) -> str:
    """Create the LangSmith dataset if it doesn't exist, then return its name."""
    try:
        client.read_dataset(dataset_name=DATASET_NAME)
        logger.info(f"LangSmith dataset already exists: '{DATASET_NAME}'")
    except Exception:
        ls_dataset = client.create_dataset(
            DATASET_NAME,
            description="Anthropic prompt engineering RAG evaluation — 15 Q&A pairs",
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


def _print_summary(results: List[dict]) -> None:
    relevance_scores = [r["relevance"] for r in results]
    faithfulness_scores = [r["faithfulness"] for r in results]
    citation_scores = [r["citation_quality"] for r in results]

    mean_relevance = statistics.mean(relevance_scores) if relevance_scores else 0.0
    mean_faithfulness = statistics.mean(faithfulness_scores) if faithfulness_scores else 0.0
    mean_citation = statistics.mean(citation_scores) if citation_scores else 0.0
    overall = statistics.mean([mean_relevance, mean_faithfulness, mean_citation])

    print("\n┌─────────────────────┬───────┐")
    print("│ Metric              │ Score │")
    print("├─────────────────────┼───────┤")
    print(f"│ Relevance           │ {mean_relevance:.1f}/5 │")
    print(f"│ Faithfulness        │ {mean_faithfulness:.1f}/5 │")
    print(f"│ Citation Quality    │ {mean_citation:.1f}/5 │")
    print(f"│ Overall             │ {overall:.1f}/5 │")
    print("└─────────────────────┴───────┘")


def main():
    dataset = load_dataset()

    if os.getenv("LANGCHAIN_API_KEY"):
        _run_with_langsmith(dataset)
    else:
        logger.warning("LANGCHAIN_API_KEY not set — running local evaluation only")
        _run_local(dataset)


def _run_with_langsmith(dataset: List[dict]) -> None:
    from langsmith import Client
    from langsmith.evaluation import evaluate

    client = Client()
    dataset_name = _sync_dataset(client, dataset)

    logger.info(f"Starting LangSmith evaluation against dataset '{dataset_name}'")
    results = evaluate(
        _target,
        data=dataset_name,
        evaluators=[ls_relevance, ls_faithfulness, ls_citation_quality],
        experiment_prefix="anthropic-rag",
        metadata={
            "model": "claude-sonnet-4-6",
            "retrieval_top_k": os.getenv("RETRIEVAL_TOP_K", "5"),
            "relevance_threshold": os.getenv("RELEVANCE_THRESHOLD", "0.70"),
        },
        max_concurrency=1,
    )

    logger.info("Evaluation complete. View results in the LangSmith UI:")
    logger.info(f"  Datasets & Testing → '{dataset_name}' → Experiments tab")

    # Print local summary from the experiment results
    local_results = []
    for r in results:
        eval_results = r.get("evaluation_results")
        if eval_results is None:
            continue
        result_list = getattr(eval_results, "results", []) or []
        scores = {ev.key: ev.score for ev in result_list if ev.score is not None}
        if scores:
            local_results.append(scores)

    if local_results:
        _print_summary(local_results)


def _run_local(dataset: List[dict]) -> None:
    total = len(dataset)
    results = []

    for i, entry in enumerate(dataset):
        question = entry["question"]
        agent_output = run_agent(question)
        scores = run_single_evaluation(question, agent_output)
        scores["question"] = question
        results.append(scores)
        print(f"[{i + 1}/{total}] {question[:50]}...")

    _print_summary(results)


if __name__ == "__main__":
    main()
