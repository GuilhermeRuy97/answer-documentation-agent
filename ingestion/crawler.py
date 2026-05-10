import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from firecrawl import V1FirecrawlApp

logger = logging.getLogger(__name__)

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
DOCS_URL = os.getenv(
    "DOCS_URL",
    "https://platform.claude.com/docs/en/build-with-claude",
)
CRAWL_LIMIT = int(os.getenv("CRAWL_LIMIT", "80"))
CACHE_PATH = os.getenv("CACHE_PATH", "data/crawl_cache.json")


def _is_valid_doc_url(url: str) -> bool:
    """Keep only clean, scrapeable page URLs — no fragments, sitemaps, or non-ASCII paths."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.fragment:
        return False
    if parsed.path.endswith("sitemap.xml"):
        return False
    parsed.path.encode("ascii")  # raises UnicodeEncodeError if non-ASCII
    return True


def crawl_docs(url: str = None, limit: int = None, force: bool = False) -> List[Dict[str, Any]]:
    url = url or DOCS_URL
    limit = limit or CRAWL_LIMIT

    cache_path = Path(CACHE_PATH)
    if cache_path.exists() and not force:
        try:
            with cache_path.open() as f:
                pages = json.load(f)
            if pages:
                logger.info(f"Loaded {len(pages)} pages from cache at {cache_path}. Use force=True to re-crawl.")
                return pages
            logger.warning(f"Cache at {cache_path} is empty, re-crawling.")
        except json.JSONDecodeError:
            logger.warning(f"Cache at {cache_path} is invalid JSON, re-crawling.")

    try:
        app = V1FirecrawlApp(api_key=FIRECRAWL_API_KEY)

        # Discover all page URLs in the section
        map_result = app.map_url(url)
        raw_urls = map_result.links or []

        # Filter and deduplicate
        seen: set = set()
        doc_urls: List[str] = []
        for u in [url] + raw_urls:  # always include the base URL
            try:
                _is_valid_doc_url(u)
            except UnicodeEncodeError:
                continue
            if _is_valid_doc_url(u) and u not in seen:
                seen.add(u)
                doc_urls.append(u)

        doc_urls = doc_urls[:limit]
        logger.info(f"Discovered {len(doc_urls)} URLs to scrape")

        # Scrape each URL individually so we always have the source URL
        pages: List[Dict[str, Any]] = []
        for page_url in doc_urls:
            try:
                page = app.scrape_url(page_url, formats=["markdown"])
                markdown = page.markdown or ""
                if not markdown or len(markdown) < 100:
                    logger.debug(f"Skipping short/empty page: {page_url}")
                    continue
                # Use scrape response fields; fall back to the requested URL/path for title
                title = page.title or page_url.rstrip("/").split("/")[-1].replace("-", " ").title()
                pages.append({
                    "url": page_url,
                    "title": title,
                    "markdown": markdown,
                })
            except Exception:
                logger.warning(f"Failed to scrape {page_url}", exc_info=True)

        logger.info(f"Crawled {len(pages)} pages from {url}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump(pages, f, indent=2)
        logger.info(f"Saved crawl cache to {cache_path}")

        return pages

    except Exception:
        logger.exception("Error during crawl")
        return []
