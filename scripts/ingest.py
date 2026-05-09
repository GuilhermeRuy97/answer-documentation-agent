import argparse
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from ingestion.pipeline import run_ingestion

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser(description="Run the ingestion pipeline.")
parser.add_argument(
    "--force-crawl",
    action="store_true",
    help="Ignore local cache and re-crawl from Firecrawl API",
)
args = parser.parse_args()

try:
    result = run_ingestion(force_crawl=args.force_crawl)
    print(f"✅ Ingestion complete: {result}")
except Exception as e:
    print(f"❌ Ingestion failed: {e}")
    sys.exit(1)
