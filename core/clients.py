"""Shared lazily-initialized API clients.

Kept lazy so modules can be imported (and unit-tested) without API keys
in the environment.
"""

from typing import Optional

import anthropic

from core.config import get_settings

_anthropic_client: Optional[anthropic.Anthropic] = None


def get_anthropic_client() -> anthropic.Anthropic:
    """Return the lazily-initialized Anthropic client.

    Returns:
        The shared Anthropic client.

    Raises:
        RuntimeError: If ANTHROPIC_API_KEY is not configured.
    """
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY must be set")

    _anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client
