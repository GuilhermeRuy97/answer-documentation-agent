# ───────────────────────────────────────────────
# FILE 6: scripts/ingest.py
# ───────────────────────────────────────────────
"""
Create `scripts/ingest.py`.

CLI entry point for ingestion. Supports a --force-crawl flag to bypass cache.

Requirements:
- Import: dotenv (load_dotenv), argparse, logging, sys
- Import: run_ingestion from ingestion.pipeline
- Load .env at the top
- Configure logging: format="%(asctime)s %(levelname)s %(message)s", level=INFO

- Set up argparse with one optional flag:
  parser.add_argument(
      "--force-crawl",
      action="store_true",
      help="Ignore local cache and re-crawl from Firecrawl API"
  )
  args = parser.parse_args()

- Call run_ingestion(force_crawl=args.force_crawl)

- On success: print "✅ Ingestion complete: {result}"
- On exception: print "❌ Ingestion failed: {e}", sys.exit(1)
"""