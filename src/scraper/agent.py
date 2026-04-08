"""Scraper Agent orchestrator — runs all fetchers, deduplicates, returns novel articles."""

import logging

from src.config import Settings
from src.scraper.fetchers.base import BaseFetcher, RawArticle
from src.scraper.fetchers.newsapi import TavilyFetcher
from src.scraper.fetchers.rss import RSSFetcher
from src.scraper.fetchers.telegram import R2Uploader, TelegramFetcher

logger = logging.getLogger(__name__)


class ScraperAgent:
    def __init__(self, settings: Settings):
        self._fetchers: list[BaseFetcher] = []

        # Tavily Search API
        if settings.tavily_api_key:
            self._fetchers.append(TavilyFetcher(api_key=settings.tavily_api_key))

        # RSS feeds (no API key needed)
        self._fetchers.append(RSSFetcher())

        # Telegram
        if settings.telegram_api_id and settings.telegram_api_hash:
            r2 = None
            if settings.r2_account_id and settings.r2_access_key_id:
                r2 = R2Uploader(
                    account_id=settings.r2_account_id,
                    access_key_id=settings.r2_access_key_id,
                    secret_access_key=settings.r2_secret_access_key,
                    bucket_name=settings.r2_bucket_name,
                )
            self._fetchers.append(
                TelegramFetcher(
                    api_id=settings.telegram_api_id,
                    api_hash=settings.telegram_api_hash,
                    r2_uploader=r2,
                )
            )

    def scrape(self, max_per_source: int = 20) -> list[RawArticle]:
        """Run all fetchers and return deduplicated articles.

        Each fetcher is run independently — one failure does not block others.
        Deduplication is done via content_hash within this batch.
        """
        all_articles: list[RawArticle] = []
        seen_hashes: set[str] = set()

        for fetcher in self._fetchers:
            fetcher_name = type(fetcher).__name__
            try:
                articles = fetcher.fetch(max_results=max_per_source)
                for article in articles:
                    if article.content_hash not in seen_hashes:
                        seen_hashes.add(article.content_hash)
                        all_articles.append(article)
            except Exception as e:
                logger.error("Fetcher %s failed: %s", fetcher_name, e)

        logger.info(
            "Scraper completed: %d unique articles from %d fetchers",
            len(all_articles),
            len(self._fetchers),
        )
        return all_articles
