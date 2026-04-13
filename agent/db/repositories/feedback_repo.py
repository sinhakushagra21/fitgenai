"""
agent/db/repositories/feedback_repo.py
──────────────────────────────────────
Repository for the ``feedback`` collection.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pymongo.collection import Collection

from agent.db.mongo import get_db
from agent.db.models import FeedbackDocument

logger = logging.getLogger("fitgen.db.feedback_repo")


class FeedbackRepository:
    """CRUD operations for the ``feedback`` collection."""

    @staticmethod
    def _col() -> Collection:
        return get_db().feedback

    # ── Create ───────────────────────────────────────────────────

    @classmethod
    def save(
        cls,
        *,
        session_id: str,
        turn_id: str,
        rating: int | None = None,
        comment: str = "",
        user_id: Any = None,
    ) -> None:
        """Save a single feedback entry for a conversation turn."""
        doc = FeedbackDocument(
            session_id=session_id,
            turn_id=turn_id,
            rating=rating,
            comment=comment,
            user_id=user_id,
        )
        cls._col().insert_one(doc.model_dump())

    # ── Read ─────────────────────────────────────────────────────

    @classmethod
    def get_by_session(cls, session_id: str) -> list[dict[str, Any]]:
        """Return all feedback entries for a session, ordered by time."""
        cursor = (
            cls._col()
            .find({"session_id": session_id})
            .sort("created_at", 1)
        )
        return list(cursor)

    @classmethod
    def get_average_rating(cls, session_id: str) -> float | None:
        """Return the average rating for a session, or None if no ratings."""
        pipeline = [
            {"$match": {"session_id": session_id, "rating": {"$ne": None}}},
            {"$group": {"_id": None, "avg": {"$avg": "$rating"}, "count": {"$sum": 1}}},
        ]
        results = list(cls._col().aggregate(pipeline))
        if results and results[0]["count"] > 0:
            return round(results[0]["avg"], 1)
        return None
