"""Reddit fetcher using PRAW for social media discussions."""

import logging
from datetime import datetime, timezone

import praw

from src.scraper.dedup import compute_content_hash
from src.scraper.fetchers.base import BaseFetcher, RawArticle

logger = logging.getLogger(__name__)

DEFAULT_SUBREDDITS = ["worldnews", "technology", "politics"]


class RedditFetcher(BaseFetcher):
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        subreddits: list[str] | None = None,
    ):
        self._reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        self._subreddits = subreddits or DEFAULT_SUBREDDITS

    def fetch(self, max_results: int = 20) -> list[RawArticle]:
        articles: list[RawArticle] = []
        per_sub = max(max_results // len(self._subreddits), 5)

        for sub_name in self._subreddits:
            try:
                sub_articles = self._fetch_subreddit(sub_name, per_sub)
                articles.extend(sub_articles)
            except Exception as e:
                logger.error("Reddit fetch failed for r/%s: %s", sub_name, e)

        logger.info("Reddit fetched %d articles total", len(articles))
        return articles[:max_results]

    def _fetch_subreddit(self, sub_name: str, limit: int) -> list[RawArticle]:
        articles: list[RawArticle] = []
        subreddit = self._reddit.subreddit(sub_name)

        for submission in subreddit.hot(limit=limit):
            if submission.stickied:
                continue

            title = submission.title or ""
            if not title:
                continue

            # Self-posts use selftext; link posts need external extraction
            if submission.is_self:
                body = submission.selftext
                url = f"https://reddit.com{submission.permalink}"
            else:
                url = submission.url
                body = self._extract_link_body(url)

            if not body:
                body = title  # Use title as fallback

            image_urls = []
            if hasattr(submission, "preview") and submission.preview:
                images = submission.preview.get("images", [])
                if images:
                    image_urls.append(images[0]["source"]["url"])
            elif submission.url and any(
                submission.url.endswith(ext) for ext in [".jpg", ".png", ".gif"]
            ):
                image_urls.append(submission.url)

            published_at = datetime.fromtimestamp(
                submission.created_utc, tz=timezone.utc
            )

            raw = RawArticle(
                url=url,
                title=title,
                body_text=body,
                image_urls=image_urls,
                source_name=f"r/{sub_name}",
                source_domain="reddit.com",
                published_at=published_at,
                content_hash=compute_content_hash(title, body),
            )
            articles.append(raw)

        return articles

    def _extract_link_body(self, url: str) -> str:
        """Extract article text from a linked URL."""
        try:
            from newspaper import Article as NewspaperArticle

            article = NewspaperArticle(url)
            article.download()
            article.parse()
            return article.text
        except Exception:
            return ""
