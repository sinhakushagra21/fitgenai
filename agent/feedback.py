"""
agent/feedback.py
─────────────────
User feedback collection and persistence for FITGEN.AI.
"""

from __future__ import annotations

import json
from typing import Any

from agent.persistence import _connect


def _ensure_feedback_table() -> None:
    """Create the user_feedback table if it doesn't exist."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                rating INTEGER,
                comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def save_feedback(
    context_id: str,
    turn_id: str,
    rating: int | None = None,
    comment: str = "",
) -> None:
    """Save user feedback for a specific turn."""
    _ensure_feedback_table()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_feedback (context_id, turn_id, rating, comment)
            VALUES (?, ?, ?, ?)
            """,
            (context_id, turn_id, rating, comment),
        )
        conn.commit()


def get_session_feedback(context_id: str) -> list[dict[str, Any]]:
    """Get all feedback entries for a conversation session."""
    _ensure_feedback_table()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, context_id, turn_id, rating, comment, created_at
            FROM user_feedback
            WHERE context_id = ?
            ORDER BY created_at ASC
            """,
            (context_id,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "context_id": row["context_id"],
            "turn_id": row["turn_id"],
            "rating": row["rating"],
            "comment": row["comment"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_average_rating(context_id: str) -> float | None:
    """Get the average rating for a session, or None if no ratings."""
    _ensure_feedback_table()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT AVG(rating) as avg_rating, COUNT(rating) as count
            FROM user_feedback
            WHERE context_id = ? AND rating IS NOT NULL
            """,
            (context_id,),
        ).fetchone()
    if row and row["count"] > 0:
        return round(row["avg_rating"], 1)
    return None
