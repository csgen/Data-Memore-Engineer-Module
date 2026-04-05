"""Tests for news fetchers with mocked HTTP responses."""

import httpx
import respx

from src.scraper.fetchers.newsapi import TavilyFetcher


class TestTavilyFetcher:
    @respx.mock
    def test_fetch_success(self):
        mock_response = {
            "results": [
                {
                    "title": "Test Article Title",
                    "url": "https://reuters.com/test-article",
                    "content": "Short description of the article.",
                    "raw_content": "Full article content here for testing purposes.",
                    "score": 0.95,
                    "images": ["https://example.com/image.jpg"],
                }
            ],
        }

        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        fetcher = TavilyFetcher(api_key="tvly-test_key")
        articles = fetcher.fetch(max_results=10)

        assert len(articles) == 1
        assert articles[0].title == "Test Article Title"
        assert articles[0].source_domain == "reuters.com"
        assert articles[0].body_text == "Full article content here for testing purposes."
        assert len(articles[0].image_urls) == 1
        assert articles[0].content_hash.startswith("sha256_")

    @respx.mock
    def test_fetch_falls_back_to_content(self):
        """When raw_content is missing, should use content field."""
        mock_response = {
            "results": [
                {
                    "title": "Fallback Article",
                    "url": "https://bbc.co.uk/news/test",
                    "content": "Short content used as fallback.",
                    "score": 0.8,
                }
            ],
        }

        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        fetcher = TavilyFetcher(api_key="tvly-test_key")
        articles = fetcher.fetch(max_results=10)

        assert len(articles) == 1
        assert articles[0].body_text == "Short content used as fallback."

    @respx.mock
    def test_fetch_skips_empty_results(self):
        mock_response = {
            "results": [
                {"title": "", "url": "https://example.com", "content": "body"},
                {"title": "No Body", "url": "https://example.com", "content": ""},
            ],
        }

        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        fetcher = TavilyFetcher(api_key="tvly-test_key")
        articles = fetcher.fetch(max_results=10)
        assert len(articles) == 0

    @respx.mock
    def test_fetch_handles_api_error(self):
        respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(429, json={"error": "rate limited"})
        )

        fetcher = TavilyFetcher(api_key="tvly-test_key")
        articles = fetcher.fetch(max_results=10)
        assert len(articles) == 0
