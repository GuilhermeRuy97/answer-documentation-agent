# ───────────────────────────────────────────────
# FILE 17: evaluation/run_eval.py
# ───────────────────────────────────────────────
"""
Create `evaluation/run_eval.py`.

Main evaluation script. Loads the dataset, runs the agent, scores, prints summary.

Requirements:
- Import: json, os, logging, time, statistics
- Import: load_dotenv from dotenv (call at top)
- Import: compiled_graph from agent.graph
- Import: run_single_evaluation from evaluation.evaluators
- Import: langsmith Client from langsmith

- def load_dataset(path: str = "evaluation/dataset.json") -> List[dict]:
    - Reads the JSON, returns list of question dicts
    - Skips entries where answer == "FILL_IN_AFTER_INGESTION"
    - Logs warning for each skipped entry

- def run_agent(question: str) -> dict:
    - Builds initial state (same as routes.py /ask)
    - Calls compiled_graph.invoke(state)
    - Returns result

- def main():
    - Loads dataset
    - For each question:
        - Runs agent
        - Scores with run_single_evaluation
        - Stores result
        - Prints progress: f"[{i+1}/{total}] {question[:50]}... ✅"
    - Calculates mean scores across all questions
    - Prints summary table:
        ┌─────────────────────┬───────┐
        │ Metric              │ Score │
        ├─────────────────────┼───────┤
        │ Relevance           │ 4.2/5 │
        │ Faithfulness        │ 4.5/5 │
        │ Citation Quality    │ 3.8/5 │
        │ Overall             │ 4.2/5 │
        └─────────────────────┴───────┘
    - Also logs results to LangSmith dataset if LANGCHAIN_API_KEY is set

if __name__ == "__main__":
    main()
"""