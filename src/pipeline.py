"""End-to-end pipeline: Scraper → Preprocessing → Memory Agent."""

import logging
import sys

from src.config import settings
from src.memory.agent import MemoryAgent
from src.preprocessing.agent import PreprocessingAgent
from src.scraper.agent import ScraperAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(max_per_source: int = 20) -> dict:
    """Run the full data pipeline once.

    Returns a summary dict with counts.
    """
    logger.info("Starting pipeline run...")

    scraper = ScraperAgent(settings)
    preprocessor = PreprocessingAgent(settings)
    memory = MemoryAgent(settings)

    try:
        # Step 1: Scrape
        raw_articles = scraper.scrape(max_per_source=max_per_source)
        logger.info("Scraped %d raw articles", len(raw_articles))

        # Step 2: Preprocess + Ingest
        ingested = 0
        skipped = 0
        failed = 0

        for raw in raw_articles:
            try:
                output = preprocessor.process(raw)
                was_new = memory.ingest_preprocessed(output)
                if was_new:
                    ingested += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.error("Failed to process article '%s': %s", raw.title[:50], e)
                failed += 1

        summary = {
            "scraped": len(raw_articles),
            "ingested": ingested,
            "skipped_duplicates": skipped,
            "failed": failed,
        }
        logger.info("Pipeline complete: %s", summary)
        return summary

    finally:
        memory.close()


def main():
    """Entry point for docker CMD."""
    run_pipeline()


if __name__ == "__main__":
    main()
