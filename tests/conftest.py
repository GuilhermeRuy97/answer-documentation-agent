"""Shared test fixtures.

Tests run fully offline: persistence is disabled, auth is reset, and the
session cache is cleared around every test. Settings mutations are restored.
"""

import pytest

import agent.session as session_store
from core.config import get_settings


@pytest.fixture(autouse=True)
def safe_settings():
    """Provide the settings singleton in a safe, restorable test configuration."""
    settings = get_settings()
    snapshot = settings.model_copy()

    settings.persist_sessions = False
    settings.api_keys = ""
    settings.langchain_api_key = ""
    settings.session_ttl_seconds = 3600
    settings.max_history_messages = 12

    session_store._cache.clear()
    yield settings

    for field in type(settings).model_fields:
        setattr(settings, field, getattr(snapshot, field))
    session_store._cache.clear()
