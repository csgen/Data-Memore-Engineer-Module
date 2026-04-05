"""Two-stage entity extraction: spaCy NER + LLM refinement."""

import json
import logging

import spacy
from openai import OpenAI

from src.id_utils import make_entity_id
from src.preprocessing.prompts import ENTITY_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class EntityExtractor:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._nlp = spacy.load("en_core_web_sm")

    def extract_entities(
        self,
        claim_text: str,
        article_context: str,
    ) -> list[dict]:
        """Extract and refine entities for a single claim.

        Returns list of {"entity_id", "name", "entity_type", "sentiment"}.
        """
        # Stage 1: spaCy NER for fast candidate extraction
        candidates = self._spacy_ner(claim_text + " " + article_context)

        # Stage 2: LLM refinement
        refined = self._llm_refine(claim_text, candidates, article_context)

        # Generate deterministic entity IDs
        for entity in refined:
            entity["entity_id"] = make_entity_id(
                entity["name"], entity["entity_type"]
            )

        return refined

    def _spacy_ner(self, text: str) -> list[dict]:
        """Run spaCy NER to get candidate entities."""
        doc = self._nlp(text)
        candidates = []
        seen = set()

        # Map spaCy labels to our entity types
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

    def _llm_refine(
        self,
        claim_text: str,
        ner_candidates: list[dict],
        article_context: str,
    ) -> list[dict]:
        """Use LLM to refine, merge, and assign sentiment to entities."""
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            claim_text=claim_text,
            ner_candidates=json.dumps(ner_candidates),
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
            entities = parsed.get("entities", [])

            valid = []
            for e in entities:
                if isinstance(e, dict) and "name" in e and "entity_type" in e:
                    valid.append({
                        "name": e["name"],
                        "entity_type": e["entity_type"],
                        "sentiment": e.get("sentiment", "neutral"),
                    })

            return valid

        except Exception as e:
            logger.error("LLM entity refinement failed: %s", e)
            # Fall back to spaCy candidates with neutral sentiment
            return [
                {**c, "sentiment": "neutral"} for c in ner_candidates
            ]
