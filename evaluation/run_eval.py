import json
import logging
import os
import statistics
from typing import List

from dotenv import load_dotenv

load_dotenv()

from agent.graph import compiled_graph
from evaluation.evaluators import run_single_evaluation

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(path: str = "evaluation/dataset.json") -> List[dict]:
    with open(path) as f:
        entries = json.load(f)

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


def main():
    dataset = load_dataset()
    total = len(dataset)
    results = []

    for i, entry in enumerate(dataset):
        question = entry["question"]
        agent_output = run_agent(question)
        scores = run_single_evaluation(question, agent_output)
        scores["question"] = question
        results.append(scores)
        print(f"[{i + 1}/{total}] {question[:50]}... ✅")

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

    if os.getenv("LANGCHAIN_API_KEY"):
        try:
            from langsmith import Client
            ls_client = Client()
            dataset_name = "anthropic-rag-eval"
            try:
                ls_dataset = ls_client.read_dataset(dataset_name=dataset_name)
            except Exception:
                ls_dataset = ls_client.create_dataset(dataset_name)

            for r in results:
                ls_client.create_example(
                    inputs={"question": r["question"]},
                    outputs={
                        "relevance": r["relevance"],
                        "faithfulness": r["faithfulness"],
                        "citation_quality": r["citation_quality"],
                    },
                    dataset_id=ls_dataset.id,
                )
            logger.info(f"Logged {len(results)} results to LangSmith dataset '{dataset_name}'")
        except Exception:
            logger.exception("Failed to log results to LangSmith")


if __name__ == "__main__":
    main()
