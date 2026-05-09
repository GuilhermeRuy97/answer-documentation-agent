# ───────────────────────────────────────────────
# FILE 16: evaluation/evaluators.py
# ───────────────────────────────────────────────
"""
Create `evaluation/evaluators.py`.

LangSmith LLM-as-judge evaluators for the RAG pipeline.

Requirements:
- Import: langsmith.evaluation (LangChainStringEvaluator, evaluate), anthropic, os, logging

- Create three evaluator prompt functions using Claude as judge:

  EVALUATOR 1: relevance_evaluator
    - Input: question, retrieved context (from run output)
    - Prompt to Claude: "Rate 1-5: Are these retrieved chunks relevant to the question?
      1=Not relevant, 5=Perfectly relevant.
      Question: {question}
      Context: {context}
      Return ONLY a JSON: {\"score\": N, \"reason\": \"...\"}"
    - Returns score 1-5

  EVALUATOR 2: faithfulness_evaluator
    - Input: answer, retrieved context
    - Prompt: "Rate 1-5: Does this answer ONLY use facts from the context?
      1=Many hallucinations, 5=Fully grounded.
      Answer: {answer}
      Context: {context}
      Return ONLY a JSON: {\"score\": N, \"reason\": \"...\"}"
    - Returns score 1-5

  EVALUATOR 3: citation_quality_evaluator
    - Input: answer with citations, citations list
    - Prompt: "Rate 1-5: Are citations accurate, present, and linked to relevant sources?
      1=No citations or wrong, 5=Perfect citations.
      Response: {response}
      Citations: {citations}
      Return ONLY a JSON: {\"score\": N, \"reason\": \"...\"}"
    - Returns score 1-5

- def run_single_evaluation(question: str, agent_output: dict) -> dict:
    - Runs all three evaluators on the output
    - Returns {"relevance": score, "faithfulness": score, "citation_quality": score}
"""