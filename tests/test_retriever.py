"""Unit tests for threshold propagation through retrieve() and the hybrid fallback."""

import retrieval.retriever as retriever_module
import retrieval.vector_store as vector_store_module
from retrieval.retriever import retrieve


class TestRetrieveThresholdPropagation:
    def test_hybrid_path_receives_threshold(self, safe_settings, monkeypatch):
        safe_settings.use_hybrid_search = True
        seen = {}

        def fake_hybrid(embedding, query, k=None, threshold=None):
            seen["k"] = k
            seen["threshold"] = threshold
            return [{"id": "x"}]

        monkeypatch.setattr(retriever_module, "embed_query", lambda q: [0.0])
        monkeypatch.setattr(retriever_module, "hybrid_search", fake_hybrid)

        retrieve("q", k=4, threshold=0.2)
        assert seen["k"] == 4
        assert seen["threshold"] == 0.2

    def test_vector_path_receives_threshold(self, safe_settings, monkeypatch):
        safe_settings.use_hybrid_search = False
        seen = {}

        def fake_similarity(embedding, k=None, threshold=None):
            seen["threshold"] = threshold
            return [{"id": "x"}]

        monkeypatch.setattr(retriever_module, "embed_query", lambda q: [0.0])
        monkeypatch.setattr(retriever_module, "similarity_search", fake_similarity)

        retrieve("q", threshold=0.2)
        assert seen["threshold"] == 0.2


class _FailingClient:
    """Simulates the hybrid RPC being unavailable (migration not applied)."""

    def rpc(self, *args, **kwargs):
        raise RuntimeError("function hybrid_match_docs does not exist")


class TestHybridFallbackThreshold:
    def test_fallback_uses_passed_threshold(self, monkeypatch):
        monkeypatch.setattr(vector_store_module, "get_client", lambda: _FailingClient())
        seen = {}

        def fake_similarity(embedding, k=6, threshold=0.30):
            seen["k"] = k
            seen["threshold"] = threshold
            return []

        monkeypatch.setattr(vector_store_module, "similarity_search", fake_similarity)
        vector_store_module.hybrid_search([0.0], "q", k=3, threshold=0.2)
        assert seen["k"] == 3
        assert seen["threshold"] == 0.2

    def test_fallback_defaults_to_settings_threshold(self, safe_settings, monkeypatch):
        safe_settings.recall_threshold = 0.33
        monkeypatch.setattr(vector_store_module, "get_client", lambda: _FailingClient())
        seen = {}

        def fake_similarity(embedding, k=6, threshold=0.30):
            seen["threshold"] = threshold
            return []

        monkeypatch.setattr(vector_store_module, "similarity_search", fake_similarity)
        vector_store_module.hybrid_search([0.0], "q")
        assert seen["threshold"] == 0.33
