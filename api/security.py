"""API security primitives: API-key auth dependency and rate limiter."""

import logging
import secrets

from fastapi import Header, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.config import get_settings

logger = logging.getLogger(__name__)

# Shared limiter; attached to the app in api/main.py and used as a decorator in routes.
limiter = Limiter(key_func=get_remote_address)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency enforcing X-API-Key auth.

    Auth is disabled when API_KEYS is empty (local development). Keys are
    compared with constant-time comparison to avoid timing attacks.

    Args:
        x_api_key: Value of the X-API-Key request header.

    Raises:
        HTTPException: 401 when a key is required and missing/invalid.
    """
    valid_keys = get_settings().api_key_list()
    if not valid_keys:
        return

    if x_api_key and any(secrets.compare_digest(x_api_key, k) for k in valid_keys):
        return

    logger.warning("Rejected request with missing/invalid API key")
    raise HTTPException(status_code=401, detail="Invalid or missing API key")
