# ───────────────────────────────────────────────
# FILE 2: ingestion/crawler.py
# ───────────────────────────────────────────────
"""
Create `ingestion/crawler.py`.

This module uses the Firecrawl Python SDK to crawl the Anthropic docs.
It caches results to disk to avoid re-crawling and wasting API credits.

Requirements:
- Import: firecrawl.FirecrawlApp, os, json, logging, pathlib.Path, List, Dict, Any

- Load from environment:
  - FIRECRAWL_API_KEY
  - DOCS_URL default: "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering"
  - CRAWL_LIMIT default: 40 (cast to int)
  - CACHE_PATH default: "data/crawl_cache.json"

- Define one function:

  def crawl_docs(url: str = None, limit: int = None, force: bool = False) -> List[Dict[str, Any]]:
    - url falls back to DOCS_URL, limit falls back to CRAWL_LIMIT

    CACHE CHECK (runs first unless force=True):
    - cache_path = Path(CACHE_PATH)
    - If cache_path.exists() and not force:
        - Load and return the JSON file contents
        - Log: f"Loaded {len(pages)} pages from cache at {cache_path}. Use force=True to re-crawl."
        - Return early — do NOT call Firecrawl API

    CRAWL (only if no cache or force=True):
    - Create FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    - Call app.crawl_url(url, params={"limit": limit, "scrapeOptions": {"formats": ["markdown"]}})
    - Clean each page into: {url, title, markdown}
      - url: page["url"] or page["metadata"]["sourceURL"]
      - title: page["metadata"].get("title", "Untitled")
      - markdown: page["markdown"]
      - Skip pages where markdown is empty or len(markdown) < 100
    - Log: f"Crawled {len(pages)} pages from {url}"

    SAVE TO CACHE:
    - cache_path.parent.mkdir(parents=True, exist_ok=True)
    - Write pages list to cache_path as JSON (indent=2)
    - Log: f"Saved crawl cache to {cache_path}"

    - Return the cleaned pages list
    - On any error, log the exception and return []
"""