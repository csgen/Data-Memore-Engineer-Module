"""NewsAPI fetcher for standard news articles."""

import logging
from datetime import datetime

import httpx

from src.scraper.dedup import compute_content_hash
from src.scraper.fetchers.base import BaseFetcher, RawArticle

logger = logging.getLogger(__name__)

NEWSAPI_BASE = "https://newsapi.org/v2"


class NewsAPIFetcher(BaseFetcher):
    def __init__(self, api_key: str, query: str = "technology OR politics OR economy"):
        self._api_key = api_key
        self._query = query

    def fetch(self, max_results: int = 20) -> list[RawArticle]:
        articles: list[RawArticle] = []

        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(
                    f"{NEWSAPI_BASE}/everything",
                    params={
                        "q": self._query,
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": min(max_results, 100),
                        "apiKey": self._api_key,
                    },
                )
                response.raise_for_status()
                data = response.json()

            for item in data.get("articles", []):
                # Skip removed articles
                if item.get("title") == "[Removed]":
                    continue

                body = item.get("content") or item.get("description") or ""
                title = item.get("title") or ""

                if not title or not body:
                    continue

                image_urls = []
                if item.get("urlToImage"):
                    image_urls.append(item["urlToImage"])

                published_at = None
                if item.get("publishedAt"):
                    try:
                        published_at = datetime.fromisoformat(
                            item["publishedAt"].replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass

                source_name = item.get("source", {}).get("name", "")
                source_domain = ""
                url = item.get("url", "")
                if url:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    source_domain = parsed.netloc

                raw = RawArticle(
                    url=url,
                    title=title,
                    body_text=body,
                    image_urls=image_urls,
                    source_name=source_name,
                    source_domain=source_domain,
                    published_at=published_at,
                    content_hash=compute_content_hash(title, body),
                )
                articles.append(raw)

        except httpx.HTTPStatusError as e:
            logger.error("NewsAPI HTTP error: %s", e.response.status_code)
        except Exception as e:
            logger.error("NewsAPI fetch failed: %s", e)

        logger.info("NewsAPI fetched %d articles", len(articles))
        return articles
