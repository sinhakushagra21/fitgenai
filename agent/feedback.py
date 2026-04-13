"""
agent/feedback.py
─────────────────
User feedback collection and persistence for FITGEN.AI.

Delegates all storage to the FeedbackRepository (MongoDB).
Maintains the same public API as the old SQLite-based module.
"""

from __future__ import annotations

from typing import Any

from agent.db.repositories.feedback_repo import FeedbackRepository


def save_feedback(
    context_id: str,
    turn_id: str,
    rating: int | None = None,
    comment: str = "",
) -> None:
    """Save user feedback for a specific turn."""
    FeedbackRepository.save(
        session_id=context_id,
        turn_id=turn_id,
        rating=rating,
        comment=comment,
    )


def get_session_feedback(context_id: str) -> list[dict[str, Any]]:
    """Get all feedback entries for a conversation session."""
    docs = FeedbackRepository.get_by_session(context_id)
    return [
        {
            "id": str(doc.get("_id", "")),
            "context_id": doc.get("session_id", ""),
            "turn_id": doc.get("turn_id", ""),
            "rating": doc.get("rating"),
            "comment": doc.get("comment", ""),
            "created_at": doc.get("created_at"),
        }
        for doc in docs
    ]


def get_average_rating(context_id: str) -> float | None:
    """Get the average rating for a session, or None if no ratings."""
    return FeedbackRepository.get_average_rating(context_id)
