"""Centralized application settings.

Single source of truth for every environment variable and tunable.
All modules must read configuration through `get_settings()` instead of
scattering `os.getenv()` calls with divergent defaults.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- API keys (empty string means "not configured"; clients are lazy) ---
    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    supabase_url: str = ""
    supabase_service_key: str = ""
    firecrawl_api_key: str = ""
    langchain_project: str = "anthropic-rag-agent"

    # --- Models ---
    generation_model: str = "claude-sonnet-4-6"
    judge_model: str = "claude-opus-4-7"
    embedding_model: str = "voyage-4"
    rerank_model: str = "rerank-2"

    # --- Ingestion ---
    docs_url: str = "https://platform.claude.com/docs/en/build-with-claude"
    crawl_limit: int = 80
    cache_path: str = "data/crawl_cache.json"
    crawl_max_retries: int = 2
    # Comma-separated URL substrings. Include: keep only matching URLs (empty = keep all).
    crawl_include_patterns: str = ""
    crawl_exclude_patterns: str = "sitemap.xml,/api/,changelog"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    min_chunk_chars: int = 50

    # --- Retrieval ---
    retrieval_top_k: int = 6
    rerank_top_k: int = 6
    # Recall prefilter for cosine similarity; kept low because the reranker does precision work.
    recall_threshold: float = 0.30
    # Gate for the agent's retry loop (mean rerank/similarity score).
    relevance_threshold: float = 0.45
    max_retry_count: int = 2
    use_hybrid_search: bool = True
    rrf_k: int = 60

    # --- Memory / sessions ---
    session_ttl_seconds: int = 3600
    # When in-context history exceeds this, older turns are summarized into a rolling summary.
    max_history_messages: int = 12
    persist_sessions: bool = True

    # --- API / security ---
    # Comma-separated valid API keys. Empty = auth disabled (local dev).
    api_keys: str = ""
    # Comma-separated allowed CORS origins. "*" = allow all (local dev).
    cors_origins: str = "*"
    rate_limit_ask: str = "20/minute"
    max_question_chars: int = 1000

    # --- App ---
    app_port: int = 8000
    log_level: str = "INFO"

    def api_key_list(self) -> List[str]:
        """Parse the comma-separated API_KEYS env var.

        Returns:
            List of non-empty API keys; empty list means auth is disabled.
        """
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    def cors_origin_list(self) -> List[str]:
        """Parse the comma-separated CORS_ORIGINS env var.

        Returns:
            List of allowed origins; ["*"] allows all.
        """
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()] or ["*"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton.

    Returns:
        The process-wide Settings instance.
    """
    return Settings()
