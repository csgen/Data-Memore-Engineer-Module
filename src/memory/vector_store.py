"""ChromaDB wrapper managing all 4 collections."""

import logging

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        host: str,
        port: int,
        api_key: str = "",
        tenant: str = "",
        database: str = "",
    ):
        if api_key:
            self._client = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=True,
                headers={"Authorization": f"Bearer {api_key}"},
                tenant=tenant,
                database=database,
            )
        else:
            # Local ChromaDB (for development/testing)
            self._client = chromadb.HttpClient(host=host, port=port)

        self._claims = self._client.get_or_create_collection("claims")
        self._articles = self._client.get_or_create_collection("articles")
        self._verdicts = self._client.get_or_create_collection("verdicts")
        self._image_captions = self._client.get_or_create_collection("image_captions")

    # ── Claims ──────────────────────────────────────────────────────────

    def upsert_claim(
        self,
        claim_id: str,
        embedding: list[float],
        document: str,
        article_id: str,
        source_id: str,
        status: str,
        extracted_at: str,
    ) -> None:
        self._claims.upsert(
            ids=[claim_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[{
                "article_id": article_id,
                "source_id": source_id,
                "status": status,
                "extracted_at": extracted_at,
            }],
        )

    def search_similar_claims(
        self, query_embedding: list[float], top_k: int = 5
    ) -> dict:
        """Cosine similarity search on claims collection."""
        return self._claims.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    def get_claims_by_ids(self, ids: list[str]) -> dict:
        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}
        return self._claims.get(ids=ids)

    # ── Articles ────────────────────────────────────────────────────────

    def upsert_article(
        self,
        article_id: str,
        embedding: list[float],
        document: str,
        source_id: str,
        domain: str,
        content_hash: str,
        published_at: str,
    ) -> None:
        self._articles.upsert(
            ids=[article_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[{
                "source_id": source_id,
                "domain": domain,
                "content_hash": content_hash,
                "published_at": published_at,
            }],
        )

    def check_content_hash_exists(self, content_hash: str) -> bool:
        """Check if an article with this hash already exists."""
        results = self._articles.get(
            where={"content_hash": content_hash},
            limit=1,
        )
        return len(results["ids"]) > 0

    # ── Verdicts ────────────────────────────────────────────────────────

    def upsert_verdict(
        self,
        verdict_id: str,
        embedding: list[float],
        document: str,
        claim_id: str,
        label: str,
        confidence: float,
        bias_score: float,
        image_mismatch: bool,
        verified_at: str,
    ) -> None:
        self._verdicts.upsert(
            ids=[verdict_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[{
                "claim_id": claim_id,
                "label": label,
                "confidence": confidence,
                "bias_score": bias_score,
                "image_mismatch": image_mismatch,
                "verified_at": verified_at,
            }],
        )

    def get_verdict_by_claim(self, claim_id: str) -> dict:
        return self._verdicts.get(where={"claim_id": claim_id})

    # ── Image Captions ──────────────────────────────────────────────────

    def upsert_caption(
        self,
        caption_id: str,
        embedding: list[float],
        document: str,
        article_id: str,
        image_url: str,
    ) -> None:
        self._image_captions.upsert(
            ids=[caption_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[{
                "article_id": article_id,
                "image_url": image_url,
            }],
        )

    def get_caption_by_article(self, article_id: str) -> dict:
        return self._image_captions.get(where={"article_id": article_id})
