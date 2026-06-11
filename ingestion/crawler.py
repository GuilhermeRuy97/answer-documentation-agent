"""Firecrawl crawler with explicit seed pages, URL filtering, retries, and caching.

Discovery: map the docs section, merge with an explicit seed list of
prompt-engineering pages (so core pages are never missed by the mapper),
filter with include/exclude patterns, then scrape each page with retry/backoff.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from firecrawl import V1FirecrawlApp

from core.config import get_settings
from ingestion.chunker import content_hash

logger = logging.getLogger(__name__)

# Core prompt-engineering pages, always crawled even if map_url misses them.
SEED_URLS: List[str] = [
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/be-clear-and-direct",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/multishot-prompting",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/chain-of-thought",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/use-xml-tags",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/system-prompts",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prefill-claudes-response",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/chain-prompts",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/long-context-tips",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/extended-thinking-tips",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices",
    "https://platform.claude.com/docs/en/build-with-claude/context-windows",
    "https://platform.claude.com/docs/en/build-with-claude/prompt-caching",
]


def _patterns(raw: str) -> List[str]:
    """Parse a comma-separated pattern string.

    Args:
        raw: Comma-separated substrings.

    Returns:
        List of non-empty patterns.
    """
    return [p.strip() for p in raw.split(",") if p.strip()]


def _is_valid_doc_url(url: str, include: List[str], exclude: List[str]) -> bool:
    """Filter URLs to clean, scrapeable doc pages.

    Args:
        url: Candidate URL.
        include: Keep only URLs containing one of these substrings (empty = all).
        exclude: Drop URLs containing any of these substrings.

    Returns:
        True if the URL should be scraped.
    """
    try:
        parsed = urlparse(url)
        parsed.path.encode("ascii")  # raises UnicodeEncodeError on non-ASCII paths
    except (ValueError, UnicodeEncodeError):
        return False
    if parsed.fragment:
        return False
    if any(pattern in url for pattern in exclude):
        return False
    if include and not any(pattern in url for pattern in include):
        return False
    return True


def _scrape_with_retry(app: V1FirecrawlApp, page_url: str, max_retries: int) -> Optional[Dict[str, Any]]:
    """Scrape one page with retry and linear backoff.

    Args:
        app: Firecrawl client.
        page_url: URL to scrape.
        max_retries: Number of retries after the first attempt.

    Returns:
        Page dict (url, title, markdown, content_hash) or None when the page
        is empty/too short or all attempts failed.
    """
    for attempt in range(max_retries + 1):
        try:
            page = app.scrape_url(page_url, formats=["markdown"])
            markdown = page.markdown or ""
            if len(markdown) < 100:
                logger.debug(f"Skipping short/empty page: {page_url}")
                return None
            title = page.title or page_url.rstrip("/").split("/")[-1].replace("-", " ").title()
            return {
                "url": page_url,
                "title": title,
                "markdown": markdown,
                "content_hash": content_hash(markdown),
            }
        except Exception:
            if attempt < max_retries:
                wait = 1.5 * (attempt + 1)
                logger.warning(f"Scrape failed for {page_url}, retrying in {wait:.1f}s")
                time.sleep(wait)
                continue
            logger.warning(f"Failed to scrape {page_url} after {max_retries + 1} attempts", exc_info=True)
    return None


def crawl_docs(url: str | None = None, limit: int | None = None, force: bool = False) -> List[Dict[str, Any]]:
    """Crawl the docs section and return page dicts, using a local JSON cache.

    Args:
        url: Section root to map; defaults to Settings.docs_url.
        limit: Max pages; defaults to Settings.crawl_limit.
        force: Ignore the cache and re-crawl.

    Returns:
        List of {url, title, markdown, content_hash} dicts.
    """
    settings = get_settings()
    url = url or settings.docs_url
    limit = limit or settings.crawl_limit
    include = _patterns(settings.crawl_include_patterns)
    exclude = _patterns(settings.crawl_exclude_patterns)

    cache_path = Path(settings.cache_path)
    if cache_path.exists() and not force:
        try:
            with cache_path.open(encoding="utf-8") as f:
                pages = json.load(f)
            if pages:
                logger.info(f"Loaded {len(pages)} pages from cache at {cache_path}. Use force=True to re-crawl.")
                return pages
            logger.warning(f"Cache at {cache_path} is empty, re-crawling.")
        except json.JSONDecodeError:
            logger.warning(f"Cache at {cache_path} is invalid JSON, re-crawling.")

    if not settings.firecrawl_api_key:
        logger.error("FIRECRAWL_API_KEY is not set; cannot crawl")
        return []

    try:
        app = V1FirecrawlApp(api_key=settings.firecrawl_api_key)

        map_result = app.map_url(url)
        raw_urls = map_result.links or []

        # Seeds first (never dropped by the limit), then section root, then discovered URLs.
        seen: set = set()
        doc_urls: List[str] = []
        for u in SEED_URLS + [url] + raw_urls:
            if u in seen:
                continue
            if not _is_valid_doc_url(u, include, exclude):
                continue
            seen.add(u)
            doc_urls.append(u)

        doc_urls = doc_urls[:limit]
        logger.info(f"Discovered {len(doc_urls)} URLs to scrape ({len(SEED_URLS)} seeds)")

        pages: List[Dict[str, Any]] = []
        for page_url in doc_urls:
            page = _scrape_with_retry(app, page_url, settings.crawl_max_retries)
            if page is not None:
                pages.append(page)

        logger.info(f"Crawled {len(pages)} pages from {url}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(pages, f, indent=2)
        logger.info(f"Saved crawl cache to {cache_path}")

        return pages

    except Exception:
        logger.exception("Error during crawl")
        return []
