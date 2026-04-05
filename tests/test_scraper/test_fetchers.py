"""Tests for news fetchers with mocked HTTP responses."""

import json

import httpx
import pytest
import respx

from src.scraper.fetchers.newsapi import NewsAPIFetcher


class TestNewsAPIFetcher:
    @respx.mock
    def test_fetch_success(self):
        mock_response = {
            "status": "ok",
            "totalResults": 1,
            "articles": [
                {
                    "source": {"id": "reuters", "name": "Reuters"},
                    "title": "Test Article Title",
                    "description": "Test description",
                    "url": "https://reuters.com/test",
                    "urlToImage": "https://example.com/image.jpg",
                    "publishedAt": "2026-04-01T12:00:00Z",
                    "content": "Full article content here for testing purposes.",
                }
            ],
        }

        respx.get("https://newsapi.org/v2/everything").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        fetcher = NewsAPIFetcher(api_key="test_key")
        articles = fetcher.fetch(max_results=10)

        assert len(articles) == 1
        assert articles[0].title == "Test Article Title"
        assert articles[0].source_name == "Reuters"
        assert len(articles[0].image_urls) == 1
        assert articles[0].content_hash.startswith("sha256_")

    @respx.mock
    def test_fetch_skips_removed_articles(self):
        mock_response = {
            "status": "ok",
            "articles": [
                {
                    "title": "[Removed]",
                    "url": "https://removed.com",
                    "content": "removed",
                    "source": {"name": "Unknown"},
                }
            ],
        }

        respx.get("https://newsapi.org/v2/everything").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        fetcher = NewsAPIFetcher(api_key="test_key")
        articles = fetcher.fetch(max_results=10)
        assert len(articles) == 0

    @respx.mock
    def test_fetch_handles_api_error(self):
        respx.get("https://newsapi.org/v2/everything").mock(
            return_value=httpx.Response(429, json={"message": "rate limited"})
        )

        fetcher = NewsAPIFetcher(api_key="test_key")
        articles = fetcher.fetch(max_results=10)
        assert len(articles) == 0
