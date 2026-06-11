"""Unit tests for graph control flow: should_retry, grade_relevance, search node."""

import agent.graph as graph_module
from agent.graph import search_docs_node, should_retry
from agent.nodes import grade_relevance


def _state(**overrides) -> dict:
    state = {
        "session_id": "test",
        "messages": [],
        "query": "What are XML tags?",
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
    state.update(overrides)
    return state


class TestShouldRetry:
    def test_good_score_proceeds(self, safe_settings):
        safe_settings.relevance_threshold = 0.45
        state = _state(relevance_score=0.9, retry_count=1)
        assert should_retry(state) == "generate_answer"

    def test_low_score_retries(self, safe_settings):
        safe_settings.relevance_threshold = 0.45
        safe_settings.max_retry_count = 2
        state = _state(relevance_score=0.1, retry_count=1)
        assert should_retry(state) == "search_docs"

    def test_allows_exactly_max_retry_count_retries(self, safe_settings):
        """With MAX_RETRY_COUNT=2: cycles 1 and 2 retry, cycle 3 proceeds (off-by-one fix)."""
        safe_settings.relevance_threshold = 0.45
        safe_settings.max_retry_count = 2
        assert should_retry(_state(relevance_score=0.1, retry_count=1)) == "search_docs"
        assert should_retry(_state(relevance_score=0.1, retry_count=2)) == "search_docs"
        assert should_retry(_state(relevance_score=0.1, retry_count=3)) == "generate_answer"

    def test_zero_max_retries_never_retries(self, safe_settings):
        safe_settings.max_retry_count = 0
        assert should_retry(_state(relevance_score=0.0, retry_count=1)) == "generate_answer"


class TestGradeRelevance:
    def test_empty_chunks_scores_zero_and_counts_cycle(self):
        result = grade_relevance(_state(retrieved_chunks=[], retry_count=0))
        assert result["relevance_score"] == 0.0
        assert result["retry_count"] == 1

    def test_prefers_rerank_score(self):
        chunks = [
            {"rerank_score": 0.8, "similarity": 0.1},
            {"rerank_score": 0.6, "similarity": 0.1},
        ]
        result = grade_relevance(_state(retrieved_chunks=chunks, retry_count=1))
        assert abs(result["relevance_score"] - 0.7) < 1e-9
        assert result["retry_count"] == 2

    def test_falls_back_to_similarity(self):
        chunks = [{"similarity": 0.5}, {"similarity": 0.7}]
        result = grade_relevance(_state(retrieved_chunks=chunks, retry_count=0))
        assert abs(result["relevance_score"] - 0.6) < 1e-9


class TestSearchDocsNode:
    def test_fuses_variants_and_reranks(self, safe_settings, monkeypatch):
        safe_settings.retrieval_top_k = 2
        safe_settings.rerank_top_k = 2

        def fake_retrieve(query, k=None, threshold=None):
            if query == "What are XML tags?":
                return [{"id": "shared", "content": "s"}, {"id": "a", "content": "a"}]
            return [{"id": "shared", "content": "s"}, {"id": "b", "content": "b"}]

        def fake_rerank(query, chunks, top_k=None):
            return [dict(c, rerank_score=0.9) for c in chunks[:top_k]]

        monkeypatch.setattr(graph_module, "retrieve", fake_retrieve)
        monkeypatch.setattr(graph_module, "rerank", fake_rerank)

        state = _state(rewritten_queries=["What are XML tags?", "hypothetical paragraph"])
        result = search_docs_node(state)

        chunks = result["retrieved_chunks"]
        assert len(chunks) == 2
        # "shared" appears in both variant result lists -> fused to the top.
        assert chunks[0]["id"] == "shared"
        assert all("rerank_score" in c for c in chunks)

    def test_no_results_returns_empty(self, monkeypatch):
        monkeypatch.setattr(graph_module, "retrieve", lambda *a, **kw: [])
        result = search_docs_node(_state(rewritten_queries=["q"]))
        assert result["retrieved_chunks"] == []

    def test_retry_widens_search(self, safe_settings, monkeypatch):
        safe_settings.retrieval_top_k = 3
        seen_k = []

        def fake_retrieve(query, k=None, threshold=None):
            seen_k.append(k)
            return []

        monkeypatch.setattr(graph_module, "retrieve", fake_retrieve)

        search_docs_node(_state(retry_count=0))
        first_k = seen_k[-1]
        search_docs_node(_state(retry_count=1))
        retry_k = seen_k[-1]
        assert retry_k > first_k
