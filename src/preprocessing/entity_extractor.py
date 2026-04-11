"""Two-stage entity extraction: spaCy NER + LLM refinement (batched)."""

import json
import logging

import spacy
from langfuse.decorators import observe
from langfuse.openai import OpenAI

from src.id_utils import make_entity_id
from src.preprocessing.prompts import ENTITY_EXTRACTION_BATCH_PROMPT

logger = logging.getLogger(__name__)


class EntityExtractor:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._nlp = spacy.load("en_core_web_sm")

    @observe(name="entity_extraction_batch")
    def extract_entities_batch(
        self,
        claims: list[dict],
        article_context: str,
    ) -> list[list[dict]]:
        """Extract entities for ALL claims in a single LLM call.

        Args:
            claims: list of {"text": str, "type": str} dicts
            article_context: the full article body text

        Returns:
            list of entity lists, one per claim (same order as input).
            Each entity is {"entity_id", "name", "entity_type", "sentiment"}.
        """
        if not claims:
            return []

        # Stage 1: spaCy NER per claim (fast, free)
        all_candidates = []
        for claim in claims:
            candidates = self._spacy_ner(claim["text"] + " " + article_context)
            all_candidates.append(candidates)

        # Stage 2: Single LLM call for all claims
        results = self._llm_refine_batch(claims, all_candidates, article_context)

        # If batch fails, fall back to spaCy candidates with neutral sentiment
        if results is None:
            results = [
                [{**c, "sentiment": "neutral"} for c in candidates]
                for candidates in all_candidates
            ]

        # Generate deterministic entity IDs
        for claim_entities in results:
            for entity in claim_entities:
                entity["entity_id"] = make_entity_id(
                    entity["name"], entity["entity_type"]
                )

        return results

    def _spacy_ner(self, text: str) -> list[dict]:
        """Run spaCy NER to get candidate entities."""
        doc = self._nlp(text)
        candidates = []
        seen = set()

        label_map = {
            "PERSON": "person",
            "ORG": "organization",
            "GPE": "country",
            "LOC": "location",
            "EVENT": "event",
            "PRODUCT": "product",
            "NORP": "organization",
            "FAC": "location",
        }

        for ent in doc.ents:
            if ent.label_ in label_map and ent.text.strip() not in seen:
                seen.add(ent.text.strip())
                candidates.append({
                    "name": ent.text.strip(),
                    "entity_type": label_map[ent.label_],
                })

        return candidates

    def _llm_refine_batch(
        self,
        claims: list[dict],
        all_candidates: list[list[dict]],
        article_context: str,
    ) -> list[list[dict]] | None:
        """Use a single LLM call to refine entities for all claims."""
        # Build the claims + candidates block for the prompt
        claims_block = []
        for i, (claim, candidates) in enumerate(zip(claims, all_candidates)):
            claims_block.append(
                f"Claim {i}: \"{claim['text']}\"\n"
                f"NER candidates: {json.dumps(candidates)}"
            )
        claims_text = "\n\n".join(claims_block)

        prompt = ENTITY_EXTRACTION_BATCH_PROMPT.format(
            claims_with_candidates=claims_text,
            article_context=article_context[:500],
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            claim_results = parsed.get("claims", [])

            # Build result list indexed by claim position
            results: list[list[dict]] = [[] for _ in claims]
            for item in claim_results:
                idx = item.get("claim_index", -1)
                if 0 <= idx < len(claims):
                    for e in item.get("entities", []):
                        if isinstance(e, dict) and "name" in e and "entity_type" in e:
                            results[idx].append({
                                "name": e["name"],
                                "entity_type": e["entity_type"],
                                "sentiment": e.get("sentiment", "neutral"),
                            })

            logger.info("Batch entity extraction: %d claims processed in 1 LLM call", len(claims))
            return results

        except Exception as e:
            logger.error("Batch LLM entity refinement failed: %s", e)
            return None
