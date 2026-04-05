"""Neo4j wrapper for the Knowledge Graph."""

import logging
from datetime import datetime
from typing import Any, Optional

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class GraphStore:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", uri)

    def close(self) -> None:
        self._driver.close()

    # ── Schema Initialization ───────────────────────────────────────────

    def init_schema(self) -> None:
        """Create uniqueness constraints and indexes. Idempotent."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.source_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.article_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Claim) REQUIRE c.claim_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Verdict) REQUIRE v.verdict_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ic:ImageCaption) REQUIRE ic.caption_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (cs:CredibilitySnapshot) REQUIRE cs.snapshot_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Prediction) REQUIRE p.prediction_id IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (c:Claim) ON (c.extracted_at)",
            "CREATE INDEX IF NOT EXISTS FOR (v:Verdict) ON (v.verified_at)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Prediction) ON (p.deadline)",
        ]
        with self._driver.session() as session:
            for stmt in constraints + indexes:
                session.run(stmt)
        logger.info("Neo4j schema initialized (8 constraints, 3 indexes)")

    # ── Write: Sources ──────────────────────────────────────────────────

    def merge_source(
        self,
        source_id: str,
        name: str,
        domain: str,
        category: str,
        base_credibility: float,
    ) -> None:
        """MERGE a source node (upsert — avoids duplicates for known outlets)."""
        with self._driver.session() as session:
            session.run(
                """
                MERGE (s:Source {source_id: $source_id})
                SET s.name = $name,
                    s.domain = $domain,
                    s.category = $category,
                    s.base_credibility = $base_credibility
                """,
                source_id=source_id,
                name=name,
                domain=domain,
                category=category,
                base_credibility=base_credibility,
            )

    # ── Write: Articles ─────────────────────────────────────────────────

    def create_article(
        self,
        article_id: str,
        title: str,
        url: str,
        source_id: str,
        published_at: datetime,
        ingested_at: datetime,
        content_hash: str,
    ) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (s:Source {source_id: $source_id})
                CREATE (a:Article {
                    article_id: $article_id,
                    title: $title,
                    url: $url,
                    source_id: $source_id,
                    published_at: datetime($published_at),
                    ingested_at: datetime($ingested_at),
                    content_hash: $content_hash
                })
                CREATE (s)-[:PUBLISHES]->(a)
                """,
                article_id=article_id,
                title=title,
                url=url,
                source_id=source_id,
                published_at=published_at.isoformat(),
                ingested_at=ingested_at.isoformat(),
                content_hash=content_hash,
            )

    # ── Write: Claims with Entities ─────────────────────────────────────

    def create_claims_with_entities(
        self,
        claims: list[dict[str, Any]],
        article_id: str,
    ) -> None:
        """Create Claim nodes, Entity nodes (MERGE), and relationship edges.

        Each claim dict should have: claim_id, claim_text, claim_type,
        extracted_at, status, and entities (list of {entity_id, name,
        entity_type, sentiment}).
        """
        with self._driver.session() as session:
            for claim in claims:
                # Create Claim node + CONTAINS edge from Article
                session.run(
                    """
                    MATCH (a:Article {article_id: $article_id})
                    CREATE (c:Claim {
                        claim_id: $claim_id,
                        article_id: $article_id,
                        claim_text: $claim_text,
                        claim_type: $claim_type,
                        extracted_at: datetime($extracted_at),
                        status: $status
                    })
                    CREATE (a)-[:CONTAINS]->(c)
                    """,
                    article_id=article_id,
                    claim_id=claim["claim_id"],
                    claim_text=claim["claim_text"],
                    claim_type=claim.get("claim_type", ""),
                    extracted_at=claim["extracted_at"],
                    status=claim.get("status", "pending"),
                )

                # MERGE entities + create MENTIONS edges
                for entity in claim.get("entities", []):
                    session.run(
                        """
                        MATCH (c:Claim {claim_id: $claim_id})
                        MERGE (e:Entity {entity_id: $entity_id})
                        ON CREATE SET
                            e.name = $name,
                            e.entity_type = $entity_type,
                            e.current_credibility = 0.5,
                            e.total_claims = 0,
                            e.accurate_claims = 0,
                            e.first_seen = datetime($extracted_at),
                            e.last_seen = datetime($extracted_at)
                        ON MATCH SET
                            e.last_seen = datetime($extracted_at)
                        CREATE (c)-[:MENTIONS {sentiment: $sentiment}]->(e)
                        """,
                        claim_id=claim["claim_id"],
                        entity_id=entity["entity_id"],
                        name=entity["name"],
                        entity_type=entity["entity_type"],
                        sentiment=entity["sentiment"],
                        extracted_at=claim["extracted_at"],
                    )

    # ── Write: Image Captions ───────────────────────────────────────────

    def create_image_caption(
        self,
        caption_id: str,
        article_id: str,
        image_url: str,
        vlm_caption: str,
    ) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (a:Article {article_id: $article_id})
                CREATE (ic:ImageCaption {
                    caption_id: $caption_id,
                    article_id: $article_id,
                    image_url: $image_url,
                    vlm_caption: $vlm_caption
                })
                CREATE (a)-[:HAS_IMAGE]->(ic)
                """,
                caption_id=caption_id,
                article_id=article_id,
                image_url=image_url,
                vlm_caption=vlm_caption,
            )

    # ── Write: Verdicts (called by Fact-Check Agent) ────────────────────

    def create_verdict(
        self,
        verdict_id: str,
        claim_id: str,
        label: str,
        confidence: float,
        evidence_summary: str,
        bias_score: float,
        image_mismatch: bool,
        verified_at: datetime,
    ) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (c:Claim {claim_id: $claim_id})
                CREATE (v:Verdict {
                    verdict_id: $verdict_id,
                    claim_id: $claim_id,
                    label: $label,
                    confidence: $confidence,
                    evidence_summary: $evidence_summary,
                    bias_score: $bias_score,
                    image_mismatch: $image_mismatch,
                    verified_at: datetime($verified_at)
                })
                CREATE (c)-[:VERIFIED_AS]->(v)
                SET c.status = 'verified'
                """,
                verdict_id=verdict_id,
                claim_id=claim_id,
                label=label,
                confidence=confidence,
                evidence_summary=evidence_summary,
                bias_score=bias_score,
                image_mismatch=image_mismatch,
                verified_at=verified_at.isoformat(),
            )

    # ── Write: Credibility Snapshots (called by Entity Tracker) ─────────

    def create_snapshot(
        self,
        snapshot_id: str,
        entity_id: str,
        credibility_score: float,
        sentiment_score: float,
        snapshot_at: datetime,
    ) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id})
                CREATE (s:CredibilitySnapshot {
                    snapshot_id: $snapshot_id,
                    entity_id: $entity_id,
                    credibility_score: $credibility_score,
                    sentiment_score: $sentiment_score,
                    snapshot_at: datetime($snapshot_at)
                })
                CREATE (e)-[:TRACKED_OVER_TIME]->(s)
                """,
                snapshot_id=snapshot_id,
                entity_id=entity_id,
                credibility_score=credibility_score,
                sentiment_score=sentiment_score,
                snapshot_at=snapshot_at.isoformat(),
            )

    # ── Write: Entity updates (called by Entity Tracker) ────────────────

    def update_entity(self, entity_id: str, updates: dict[str, Any]) -> None:
        """Update specific fields on an Entity node."""
        set_clauses = ", ".join(f"e.{k} = ${k}" for k in updates)
        with self._driver.session() as session:
            session.run(
                f"MATCH (e:Entity {{entity_id: $entity_id}}) SET {set_clauses}",
                entity_id=entity_id,
                **updates,
            )

    # ── Write: Predictions (called by Prediction Agent) ─────────────────

    def create_prediction(
        self,
        prediction_id: str,
        entity_id: str,
        prediction_text: str,
        confidence: float,
        predicted_at: datetime,
        deadline: datetime,
        outcome: Optional[str] = None,
    ) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id})
                CREATE (p:Prediction {
                    prediction_id: $prediction_id,
                    entity_id: $entity_id,
                    prediction_text: $prediction_text,
                    confidence: $confidence,
                    predicted_at: datetime($predicted_at),
                    deadline: datetime($deadline),
                    outcome: $outcome
                })
                CREATE (e)-[:SUBJECT_OF]->(p)
                """,
                prediction_id=prediction_id,
                entity_id=entity_id,
                prediction_text=prediction_text,
                confidence=confidence,
                predicted_at=predicted_at.isoformat(),
                deadline=deadline.isoformat(),
                outcome=outcome,
            )

    def resolve_prediction(self, prediction_id: str, outcome: str) -> None:
        with self._driver.session() as session:
            session.run(
                "MATCH (p:Prediction {prediction_id: $prediction_id}) SET p.outcome = $outcome",
                prediction_id=prediction_id,
                outcome=outcome,
            )

    # ── Read: Entity context ────────────────────────────────────────────

    def get_entity_context(self, claim_id: str) -> list[dict]:
        """Get entities mentioned in a claim with their credibility."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Claim {claim_id: $claim_id})-[m:MENTIONS]->(e:Entity)
                RETURN e.entity_id AS entity_id,
                       e.name AS name,
                       e.entity_type AS entity_type,
                       e.current_credibility AS current_credibility,
                       m.sentiment AS sentiment
                """,
                claim_id=claim_id,
            )
            return [dict(record) for record in result]

    def get_entity_claims(
        self,
        entity_id: str,
        since: Optional[datetime] = None,
    ) -> list[dict]:
        """Get claims mentioning an entity, optionally filtered by time."""
        query = """
            MATCH (e:Entity {entity_id: $entity_id})<-[m:MENTIONS]-(c:Claim)
            -[:VERIFIED_AS]->(v:Verdict)
        """
        params: dict[str, Any] = {"entity_id": entity_id}

        if since:
            query += " WHERE v.verified_at > datetime($since)"
            params["since"] = since.isoformat()

        query += """
            RETURN c.claim_id AS claim_id,
                   c.claim_text AS claim_text,
                   v.label AS verdict_label,
                   v.confidence AS verdict_confidence,
                   m.sentiment AS sentiment
            ORDER BY v.verified_at DESC
        """
        with self._driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def get_entity_snapshots(
        self, entity_id: str, limit: int = 20
    ) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id})
                      -[:TRACKED_OVER_TIME]->(s:CredibilitySnapshot)
                RETURN s.snapshot_id AS snapshot_id,
                       s.credibility_score AS credibility_score,
                       s.sentiment_score AS sentiment_score,
                       s.snapshot_at AS snapshot_at
                ORDER BY s.snapshot_at DESC
                LIMIT $limit
                """,
                entity_id=entity_id,
                limit=limit,
            )
            return [dict(record) for record in result]

    def get_source_credibility(self, article_id: str) -> Optional[float]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (s:Source)-[:PUBLISHES]->(a:Article {article_id: $article_id})
                RETURN s.base_credibility AS base_credibility
                """,
                article_id=article_id,
            )
            record = result.single()
            return record["base_credibility"] if record else None

    def get_trending_entities(
        self, since: datetime, limit: int = 10
    ) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)<-[:MENTIONS]-(c:Claim)
                WHERE c.extracted_at > datetime($since)
                RETURN e.entity_id AS entity_id,
                       e.name AS name,
                       e.entity_type AS entity_type,
                       e.current_credibility AS current_credibility,
                       count(c) AS mention_count
                ORDER BY mention_count DESC
                LIMIT $limit
                """,
                since=since.isoformat(),
                limit=limit,
            )
            return [dict(record) for record in result]

    def get_expired_predictions(self) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (p:Prediction)
                WHERE p.deadline < datetime() AND p.outcome IS NULL
                RETURN p.prediction_id AS prediction_id,
                       p.entity_id AS entity_id,
                       p.prediction_text AS prediction_text,
                       p.confidence AS confidence,
                       p.deadline AS deadline
                """
            )
            return [dict(record) for record in result]
