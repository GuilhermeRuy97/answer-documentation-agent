"""API route tests with the graph and external services mocked."""

import pytest
from fastapi.testclient import TestClient

import api.routes as routes_module
from api.main import app


class FakeGraph:
    """Stands in for the compiled LangGraph in route tests."""

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        return {
            **state,
            "final_response": "XML tags structure prompts [1].",
            "answer": "XML tags structure prompts [1].",
            "citations": [
                {"title": "Use XML tags", "url": "https://example.com/xml", "snippet": "XML tags..."}
            ],
            "rewritten_queries": [state["query"], "hypothetical paragraph"],
            "messages": [],
        }


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(routes_module, "compiled_graph", FakeGraph())
    monkeypatch.setattr(routes_module, "health_check", lambda: True)
    return TestClient(app)


class TestAsk:
    def test_ask_returns_answer_and_trace_id(self, client):
        resp = client.post("/ask", json={"question": "What are XML tags?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "XML tags structure prompts [1]."
        assert len(data["citations"]) == 1
        assert data["session_id"]
        assert data["trace_id"]
        assert len(data["rewritten_queries"]) == 2

    def test_ask_reuses_session_id(self, client):
        resp = client.post("/ask", json={"question": "q", "session_id": "fixed-session"})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "fixed-session"

    def test_ask_validates_empty_question(self, client):
        resp = client.post("/ask", json={"question": "   "})
        assert resp.status_code == 422

    def test_ask_sanitizes_internal_errors(self, client, monkeypatch):
        class BrokenGraph:
            def invoke(self, state, config=None):
                raise RuntimeError("secret internal detail")

        monkeypatch.setattr(routes_module, "compiled_graph", BrokenGraph())
        resp = client.post("/ask", json={"question": "q"})
        assert resp.status_code == 500
        assert "secret internal detail" not in resp.text
        assert resp.json()["detail"] == "Internal server error"


class TestAuth:
    def test_auth_disabled_when_no_keys(self, client):
        assert client.post("/ask", json={"question": "q"}).status_code == 200

    def test_rejects_missing_key_when_enabled(self, client, safe_settings):
        safe_settings.api_keys = "secret-key-1, secret-key-2"
        resp = client.post("/ask", json={"question": "q"})
        assert resp.status_code == 401

    def test_accepts_valid_key(self, client, safe_settings):
        safe_settings.api_keys = "secret-key-1, secret-key-2"
        resp = client.post("/ask", json={"question": "q"}, headers={"X-API-Key": "secret-key-2"})
        assert resp.status_code == 200

    def test_rejects_wrong_key(self, client, safe_settings):
        safe_settings.api_keys = "secret-key-1"
        resp = client.post("/ask", json={"question": "q"}, headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_health_is_public_even_with_auth_enabled(self, client, safe_settings):
        safe_settings.api_keys = "secret-key-1"
        assert client.get("/health").status_code == 200


class TestHistory:
    def test_history_404_for_unknown_session(self, client):
        assert client.get("/history/unknown-session").status_code == 404

    def test_history_roundtrip_via_session_store(self, client):
        from langchain_core.messages import AIMessage, HumanMessage

        import agent.session as session_store

        sid = session_store.get_or_create_session("api-history-test")
        session_store.append_messages(sid, [HumanMessage(content="q"), AIMessage(content="a")])

        resp = client.get(f"/history/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["message_count"] == 2
        assert data["messages"][0] == {"role": "human", "content": "q"}

    def test_delete_history(self, client):
        from langchain_core.messages import HumanMessage

        import agent.session as session_store

        sid = session_store.get_or_create_session("api-delete-test")
        session_store.append_messages(sid, [HumanMessage(content="q")])

        resp = client.delete(f"/history/{sid}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert client.get(f"/history/{sid}").status_code == 404


class TestHealth:
    def test_health_connected(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["vector_store"] == "connected"

    def test_health_error_state(self, client, monkeypatch):
        monkeypatch.setattr(routes_module, "health_check", lambda: False)
        assert client.get("/health").json()["vector_store"] == "error"


class TestFeedback:
    def test_feedback_logged_locally_without_langsmith(self, client, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        resp = client.post("/feedback", json={"trace_id": "abc-123", "score": 1.0})
        assert resp.status_code == 200
        assert resp.json()["status"] == "logged_locally"

    def test_feedback_score_validation(self, client):
        resp = client.post("/feedback", json={"trace_id": "abc", "score": 2.0})
        assert resp.status_code == 422


class TestRoot:
    def test_root_serves_ui(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
