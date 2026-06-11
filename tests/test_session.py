"""Unit tests for agent/session.py (short-term memory cache + TTL)."""

import time

from langchain_core.messages import AIMessage, HumanMessage

import agent.session as session_store


class TestSessionLifecycle:
    def test_generates_uuid_when_none(self):
        sid = session_store.get_or_create_session(None)
        assert isinstance(sid, str) and len(sid) == 36

    def test_keeps_provided_session_id(self):
        assert session_store.get_or_create_session("my-session") == "my-session"

    def test_append_and_get_history(self):
        sid = session_store.get_or_create_session("s1")
        session_store.append_messages(sid, [HumanMessage(content="hi"), AIMessage(content="hello")])
        history = session_store.get_history(sid)
        assert len(history) == 2
        assert history[0].content == "hi"
        assert history[1].content == "hello"

    def test_history_is_a_copy(self):
        sid = session_store.get_or_create_session("s2")
        session_store.append_messages(sid, [HumanMessage(content="x")])
        history = session_store.get_history(sid)
        history.append(AIMessage(content="mutated"))
        assert len(session_store.get_history(sid)) == 1

    def test_summary_roundtrip(self):
        sid = session_store.get_or_create_session("s3")
        assert session_store.get_summary(sid) == ""
        session_store.save_summary(sid, "talked about XML tags")
        assert session_store.get_summary(sid) == "talked about XML tags"

    def test_trim_history_keeps_most_recent(self):
        sid = session_store.get_or_create_session("s4")
        msgs = [HumanMessage(content=str(i)) for i in range(10)]
        session_store.append_messages(sid, msgs)
        session_store.trim_history(sid, keep=4)
        history = session_store.get_history(sid)
        assert len(history) == 4
        assert history[0].content == "6"

    def test_clear_history(self):
        sid = session_store.get_or_create_session("s5")
        session_store.append_messages(sid, [HumanMessage(content="x")])
        assert session_store.clear_history(sid) is True
        assert session_store.get_history(sid) == []

    def test_clear_unknown_session(self):
        assert session_store.clear_history("does-not-exist") is False


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    """Minimal chainable stand-in for a supabase table query."""

    def __init__(self, rows):
        self._rows = rows
        self._desc = False
        self._limit = None

    def select(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def order(self, column, desc=False):
        self._desc = desc
        return self

    def limit(self, n):
        self._limit = n
        return self

    def execute(self):
        rows = list(self._rows)
        if self._desc:
            rows = rows[::-1]
        if self._limit is not None:
            rows = rows[: self._limit]
        return _FakeResult(rows)


class _FakeClient:
    def __init__(self, messages, sessions):
        self._tables = {"chat_messages": messages, "chat_sessions": sessions}

    def table(self, name):
        return _FakeQuery(self._tables[name])


class TestDbReload:
    def test_load_from_db_is_bounded_and_chronological(self, safe_settings, monkeypatch):
        """After a TTL eviction, only the last max_history_messages turns are
        reloaded; older turns stay summarized instead of re-entering context."""
        safe_settings.persist_sessions = True
        safe_settings.supabase_url = "http://localhost"
        safe_settings.supabase_service_key = "key"
        safe_settings.max_history_messages = 4

        rows = [{"role": "human", "content": str(i)} for i in range(10)]
        fake = _FakeClient(messages=rows, sessions=[{"summary": "older turns summarized"}])

        import retrieval.vector_store as vector_store_module

        monkeypatch.setattr(vector_store_module, "get_client", lambda: fake)

        entry = session_store._load_from_db("evicted-session")
        assert entry is not None
        assert [m.content for m in entry.messages] == ["6", "7", "8", "9"]
        assert entry.summary == "older turns summarized"

    def test_new_session_skips_db_probe(self, monkeypatch):
        def _fail(session_id):
            raise AssertionError("fresh UUIDs must not be probed in the DB")

        monkeypatch.setattr(session_store, "_load_from_db", _fail)
        sid = session_store.get_or_create_session(None)
        assert sid in session_store._cache

    def test_provided_session_id_still_loads_from_db(self, monkeypatch):
        calls = []
        monkeypatch.setattr(session_store, "_load_from_db", lambda sid: calls.append(sid) or None)
        session_store.get_or_create_session("known-session")
        assert calls == ["known-session"]


class TestTtlEviction:
    def test_idle_sessions_evicted(self, safe_settings):
        safe_settings.session_ttl_seconds = 60
        sid = session_store.get_or_create_session("ttl-test")
        session_store.append_messages(sid, [HumanMessage(content="x")])

        # Simulate idleness past the TTL
        session_store._cache[sid].last_access = time.time() - 120
        session_store._evict_expired()
        assert sid not in session_store._cache

    def test_active_sessions_survive(self, safe_settings):
        safe_settings.session_ttl_seconds = 60
        sid = session_store.get_or_create_session("ttl-active")
        session_store._evict_expired()
        assert sid in session_store._cache
