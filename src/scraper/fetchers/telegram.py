"""Telegram fetcher using Telethon for public news channel messages."""

import asyncio
import io
import logging
from datetime import datetime, timezone
from typing import Optional

import boto3
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto

from src.scraper.dedup import compute_content_hash
from src.scraper.fetchers.base import BaseFetcher, RawArticle

logger = logging.getLogger(__name__)

# Public news channels — breaking news, geopolitics, rumors/tips
DEFAULT_CHANNELS = [
    "Breaking911",
    "disclosetv",
    "WatcherGuru",
]


class R2Uploader:
    """Upload images to Cloudflare R2 (S3-compatible)."""

    def __init__(
        self,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
    ):
        self._bucket = bucket_name
        self._client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
        )
        self._public_base = f"https://pub-{account_id}.r2.dev"

    def upload(self, data: bytes, key: str, content_type: str = "image/jpeg") -> str:
        """Upload bytes to R2 and return the public URL."""
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return f"{self._public_base}/{key}"


class TelegramFetcher(BaseFetcher):
    def __init__(
        self,
        api_id: int,
        api_hash: str,
        channels: list[str] | None = None,
        r2_uploader: Optional[R2Uploader] = None,
    ):
        self._api_id = api_id
        self._api_hash = api_hash
        self._channels = channels or DEFAULT_CHANNELS
        self._r2 = r2_uploader

    def fetch(self, max_results: int = 20) -> list[RawArticle]:
        """Fetch messages from public Telegram channels.

        Wraps the async implementation for compatibility with
        the synchronous BaseFetcher interface.
        """
        try:
            return asyncio.run(self._fetch_async(max_results))
        except Exception as e:
            logger.error("Telegram fetch failed: %s", e)
            return []

    async def _fetch_async(self, max_results: int) -> list[RawArticle]:
        articles: list[RawArticle] = []
        per_channel = max(max_results // len(self._channels), 5)

        async with TelegramClient(
            "telegram_session", self._api_id, self._api_hash
        ) as client:
            for channel_name in self._channels:
                try:
                    channel_articles = await self._fetch_channel(
                        client, channel_name, per_channel
                    )
                    articles.extend(channel_articles)
                except Exception as e:
                    logger.error(
                        "Telegram fetch failed for @%s: %s", channel_name, e
                    )

        logger.info("Telegram fetched %d messages total", len(articles))
        return articles[:max_results]

    async def _fetch_channel(
        self,
        client: TelegramClient,
        channel_name: str,
        limit: int,
    ) -> list[RawArticle]:
        articles: list[RawArticle] = []
        entity = await client.get_entity(channel_name)

        async for message in client.iter_messages(entity, limit=limit):
            if not message.text:
                continue

            text = message.text.strip()
            if len(text) < 20:
                continue

            # Use first line as title, full text as body
            lines = text.split("\n", 1)
            title = lines[0][:200]
            body = text

            # Handle image: download → upload to R2 if available
            image_urls: list[str] = []
            if self._r2 and isinstance(message.media, MessageMediaPhoto):
                try:
                    buffer = io.BytesIO()
                    await client.download_media(message, file=buffer)
                    buffer.seek(0)

                    key = f"telegram/{channel_name}/{message.id}.jpg"
                    url = self._r2.upload(buffer.read(), key)
                    image_urls.append(url)
                except Exception as e:
                    logger.warning("Image upload failed for message %d: %s", message.id, e)

            published_at = message.date.astimezone(timezone.utc) if message.date else datetime.now(timezone.utc)

            raw = RawArticle(
                url=f"https://t.me/{channel_name}/{message.id}",
                title=title,
                body_text=body,
                image_urls=image_urls,
                source_name=f"@{channel_name}",
                source_domain="t.me",
                published_at=published_at,
                content_hash=compute_content_hash(title, body),
            )
            articles.append(raw)

        return articles
