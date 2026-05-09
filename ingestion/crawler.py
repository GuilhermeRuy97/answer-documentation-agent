import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from firecrawl import FirecrawlApp

logger = logging.getLogger(__name__)

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
DOCS_URL = os.getenv("DOCS_URL", "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering")
CRAWL_LIMIT = int(os.getenv("CRAWL_LIMIT", "40"))
CACHE_PATH = os.getenv("CACHE_PATH", "data/crawl_cache.json")


def crawl_docs(url: str = None, limit: int = None, force: bool = False) -> List[Dict[str, Any]]:
    url = url or DOCS_URL
    limit = limit or CRAWL_LIMIT

    cache_path = Path(CACHE_PATH)
    if cache_path.exists() and not force:
        with cache_path.open() as f:
            pages = json.load(f)
        logger.info(f"Loaded {len(pages)} pages from cache at {cache_path}. Use force=True to re-crawl.")
        return pages

    try:
        app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        result = app.crawl_url(url, params={"limit": limit, "scrapeOptions": {"formats": ["markdown"]}})

        pages = []
        for page in result.data:
            markdown = page.get("markdown", "")
            if not markdown or len(markdown) < 100:
                continue
            metadata = page.get("metadata", {})
            pages.append({
                "url": page.get("url") or metadata.get("sourceURL", ""),
                "title": metadata.get("title", "Untitled"),
                "markdown": markdown,
            })

        logger.info(f"Crawled {len(pages)} pages from {url}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump(pages, f, indent=2)
        logger.info(f"Saved crawl cache to {cache_path}")

        return pages

    except Exception:
        logger.exception("Error during crawl")
        return []
