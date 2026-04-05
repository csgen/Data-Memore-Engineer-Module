"""Scraper Agent orchestrator — runs all fetchers, deduplicates, returns novel articles."""

import logging

from src.config import Settings
from src.scraper.fetchers.base import BaseFetcher, RawArticle
from src.scraper.fetchers.newsapi import NewsAPIFetcher
from src.scraper.fetchers.reddit import RedditFetcher
from src.scraper.fetchers.rss import RSSFetcher

logger = logging.getLogger(__name__)


class ScraperAgent:
    def __init__(self, settings: Settings):
        self._fetchers: list[BaseFetcher] = []

        # NewsAPI
        if settings.newsapi_key:
            self._fetchers.append(NewsAPIFetcher(api_key=settings.newsapi_key))

        # RSS feeds (no API key needed)
        self._fetchers.append(RSSFetcher())

        # Reddit
        if settings.reddit_client_id and settings.reddit_client_secret:
            self._fetchers.append(
                RedditFetcher(
                    client_id=settings.reddit_client_id,
                    client_secret=settings.reddit_client_secret,
                    user_agent=settings.reddit_user_agent,
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
