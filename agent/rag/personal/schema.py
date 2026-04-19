"""
agent/rag/personal/schema.py
────────────────────────────
Typed data structures for Personal RAG chunks (in-memory).

These are distinct from the pymongo / pydantic collection schemas in
``agent/db/models.py`` — these live purely in-process between the
chunker, the embedder and the indexer. Keeping them separate means
tests for the chunker don't need Mongo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SectionType(str, Enum):
    """Section tag attached to every chunk.

    Used as a Mongo filter field in the ``plan_chunks`` collection and
    by the retriever to pick anchor chunks (macros / calorie_calc /
    split_overview) alongside vector-search hits.
    """

    # Shared
    NOTES = "notes"

    # Diet-only
    CALORIE_CALC = "calorie_calc"
    MACROS = "macros"
    MEAL_DAY = "meal_day"
    SNACK_SWAPS = "snack_swaps"
    RULES = "rules"
    TIMELINE = "timeline"
    HYDRATION = "hydration"
    SUPPLEMENTS = "supplements"

    # Workout-only
    SPLIT_OVERVIEW = "split_overview"
    WORKOUT_DAY = "workout_day"
    WARMUP = "warmup"
    MAIN_LIFTS = "main_lifts"
    ACCESSORIES = "accessories"
    CARDIO = "cardio"
    MOBILITY = "mobility"
    PROGRESSION = "progression"


@dataclass
class PlanChunk:
    """One section of a plan, ready to embed + index.

    The chunker produces a list of these; the embedder fills in
    ``embedding``; the indexer upserts them into ``plan_chunks``.
    """

    user_id: str
    plan_id: str
    plan_type: str                    # diet | workout
    plan_status: str                  # draft | confirmed | archived
    plan_version: int
    section_type: str                 # SectionType value
    day_of_week: str | None           # monday..sunday or None
    heading: str
    preamble: str
    chunk_text: str                   # raw markdown slice
    embedded_text: str                # what the embedder sees
    chunk_tokens: int
    source_content_hash: str
    profile_snapshot_digest: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)

    def to_mongo(self) -> dict[str, Any]:
        """Convert to the dict shape consumed by PlanChunksRepository.

        String ObjectIds are left as strings here — the repository
        converts on write via upsert_many (bulk UpdateOne filter keys
        accept either, but to keep types clean we cast before calling).
        """
        from bson import ObjectId

        def _oid(v: Any) -> Any:
            if isinstance(v, str):
                try:
                    return ObjectId(v)
                except Exception:  # noqa: BLE001
                    return v
            return v

        return {
            "user_id": _oid(self.user_id),
            "plan_id": _oid(self.plan_id),
            "plan_type": self.plan_type,
            "plan_status": self.plan_status,
            "plan_version": self.plan_version,
            "section_type": self.section_type,
            "day_of_week": self.day_of_week,
            "heading": self.heading,
            "preamble": self.preamble,
            "chunk_text": self.chunk_text,
            "embedded_text": self.embedded_text,
            "chunk_tokens": self.chunk_tokens,
            "profile_snapshot_digest": self.profile_snapshot_digest,
            "source_content_hash": self.source_content_hash,
            "embedding": self.embedding,
        }
