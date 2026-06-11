"""Logging setup shared by all entry points (API, scripts, evaluation)."""

import logging

from core.config import get_settings

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once, idempotently.

    Args:
        level: Optional log level name; defaults to Settings.log_level.
    """
    settings = get_settings()
    resolved = (level or settings.log_level).upper()
    logging.basicConfig(format=_FORMAT, level=resolved)
    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
