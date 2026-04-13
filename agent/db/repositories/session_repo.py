"""
agent/db/repositories/session_repo.py
─────────────────────────────────────
Repository for the ``sessions`` collection.

Replaces the old ``context_states`` SQLite table.  Sessions are
ephemeral workflow trackers with a configurable TTL auto-expiry.

Set ``FITGEN_SESSION_TTL_HOURS`` in ``.env`` to control how long
session documents live before MongoDB's TTL index removes them.
Default is **2 hours** (``FITGEN_SESSION_TTL_HOURS=2``).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId
from pymongo.collection import Collection

from agent.db.mongo import get_db
from agent.db.models import SessionDocument

logger = logging.getLogger("fitgen.db.session_repo")

# Configurable TTL — read from env, default 2 hours
_SESSION_TTL_HOURS = float(os.getenv("FITGEN_SESSION_TTL_HOURS", "2"))


class SessionRepository:
    """CRUD operations for the ``sessions`` collection."""

    @staticmethod
    def _col() -> Collection:
        return get_db().sessions

    # ── Read ─────────────────────────────────────────────────────

    @classmethod
    def find_by_session_id(cls, session_id: str) -> dict[str, Any] | None:
        """Find a session by its UUID session_id."""
        if not session_id:
            return None
        return cls._col().find_one({"session_id": session_id})

    @classmethod
    def find_latest_by_email(cls, user_email: str) -> dict[str, Any] | None:
        """Find the most recently updated session for a given email."""
        if not user_email:
            return None
        return cls._col().find_one(
            {"user_email": user_email},
            sort=[("updated_at", -1)],
        )

    @classmethod
    def find_latest_by_user_id(cls, user_id: ObjectId | str) -> dict[str, Any] | None:
        """Find the most recently updated session for a given user_id."""
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        return cls._col().find_one(
            {"user_id": user_id},
            sort=[("updated_at", -1)],
        )

    # ── Create / Upsert ─────────────────────────────────────────

    @classmethod
    def upsert(
        cls,
        *,
        session_id: str,
        user_email: str = "",
        user_id: ObjectId | str | None = None,
        user_profile: dict[str, Any] | None = None,
        workflow: dict[str, Any] | None = None,
        calendar_sync_requested: bool = False,
    ) -> None:
        """Create or update a session document."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=_SESSION_TTL_HOURS)

        set_fields: dict[str, Any] = {
            "user_email": user_email,
            "calendar_sync_requested": calendar_sync_requested,
            "updated_at": now,
            "expires_at": expires,
        }
        if user_id is not None:
            if isinstance(user_id, str):
                user_id = ObjectId(user_id)
            set_fields["user_id"] = user_id
        if user_profile is not None:
            set_fields["user_profile"] = user_profile
        if workflow is not None:
            set_fields["workflow"] = workflow

        cls._col().update_one(
            {"session_id": session_id},
            {
                "$set": set_fields,
                "$setOnInsert": {
                    "session_id": session_id,
                    "created_at": now,
                },
            },
            upsert=True,
        )

    # ── Delete ───────────────────────────────────────────────────

    @classmethod
    def delete(cls, session_id: str) -> bool:
        """Delete a session by session_id.  Returns True if removed."""
        result = cls._col().delete_one({"session_id": session_id})
        return result.deleted_count > 0
