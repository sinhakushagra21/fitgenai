"""
agent/db/repositories/user_repo.py
──────────────────────────────────
Repository for the ``users`` collection.

Handles user lookup (by email or _id), creation, and profile updates.
Profile is split into three sub-documents: base_profile, diet_profile,
workout_profile — so domain-specific updates don't clobber each other.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo.collection import Collection

from agent.db.mongo import get_db
from agent.db.models import UserDocument

logger = logging.getLogger("fitgen.db.user_repo")


class UserRepository:
    """CRUD operations for the ``users`` collection."""

    @staticmethod
    def _col() -> Collection:
        return get_db().users

    # ── Read ─────────────────────────────────────────────────────

    @classmethod
    def find_by_email(cls, email: str) -> dict[str, Any] | None:
        """Find a user by email.  Returns the raw Mongo document or None."""
        if not email:
            return None
        return cls._col().find_one({"email": email})

    @classmethod
    def find_by_id(cls, user_id: ObjectId | str) -> dict[str, Any] | None:
        """Find a user by ``_id``."""
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        return cls._col().find_one({"_id": user_id})

    # ── Create ───────────────────────────────────────────────────

    @classmethod
    def create(cls, *, email: str, name: str = "", **kwargs: Any) -> ObjectId:
        """Insert a new user document.  Returns the inserted ``_id``."""
        doc = UserDocument(email=email, name=name, **kwargs)
        result = cls._col().insert_one(doc.model_dump())
        logger.info("User created: email=%s _id=%s", email, result.inserted_id)
        return result.inserted_id

    @classmethod
    def find_or_create(cls, email: str, name: str = "") -> dict[str, Any]:
        """Return existing user or create a new one.

        This is the primary entry point used by the tools: if the user
        already exists we get their full profile; otherwise we bootstrap
        an empty document.
        """
        existing = cls.find_by_email(email)
        if existing:
            return existing
        new_id = cls.create(email=email, name=name)
        return cls.find_by_id(new_id)  # type: ignore[return-value]

    # ── Update ───────────────────────────────────────────────────

    @classmethod
    def update_base_profile(cls, email: str, profile: dict[str, Any]) -> None:
        """Merge fields into ``base_profile``."""
        now = datetime.now(timezone.utc)
        cls._col().update_one(
            {"email": email},
            {
                "$set": {
                    **{f"base_profile.{k}": v for k, v in profile.items()},
                    "updated_at": now,
                },
            },
        )

    @classmethod
    def update_diet_profile(cls, email: str, profile: dict[str, Any]) -> None:
        """Merge fields into ``diet_profile``."""
        now = datetime.now(timezone.utc)
        cls._col().update_one(
            {"email": email},
            {
                "$set": {
                    **{f"diet_profile.{k}": v for k, v in profile.items()},
                    "updated_at": now,
                },
            },
        )

    @classmethod
    def update_workout_profile(cls, email: str, profile: dict[str, Any]) -> None:
        """Merge fields into ``workout_profile``."""
        now = datetime.now(timezone.utc)
        cls._col().update_one(
            {"email": email},
            {
                "$set": {
                    **{f"workout_profile.{k}": v for k, v in profile.items()},
                    "updated_at": now,
                },
            },
        )

    @classmethod
    def update_profile(
        cls,
        email: str,
        *,
        base: dict[str, Any] | None = None,
        diet: dict[str, Any] | None = None,
        workout: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        """Batch-update any combination of profile sub-documents."""
        now = datetime.now(timezone.utc)
        set_fields: dict[str, Any] = {"updated_at": now}
        if name is not None:
            set_fields["name"] = name
        if base:
            for k, v in base.items():
                set_fields[f"base_profile.{k}"] = v
        if diet:
            for k, v in diet.items():
                set_fields[f"diet_profile.{k}"] = v
        if workout:
            for k, v in workout.items():
                set_fields[f"workout_profile.{k}"] = v

        cls._col().update_one({"email": email}, {"$set": set_fields})

        # Best-effort Personal-RAG profile re-index. Never raises.
        try:
            from agent.rag.personal.indexer import index_profile
            index_profile(email)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[user_repo] RAG profile reindex failed: %s", exc)

    # ── Delete ───────────────────────────────────────────────────

    @classmethod
    def delete_by_email(cls, email: str) -> bool:
        """Delete user by email.  Returns True if a document was removed."""
        result = cls._col().delete_one({"email": email})
        return result.deleted_count > 0

    # ── Helpers ──────────────────────────────────────────────────

    @classmethod
    def get_merged_profile(cls, email: str, domain: str = "") -> dict[str, Any]:
        """Return a flat profile dict merging base + domain-specific fields.

        This is what the tools consume: a single flat dict with all fields
        the user has ever provided, regardless of which session they came from.
        """
        user = cls.find_by_email(email)
        if not user:
            return {}

        merged: dict[str, Any] = {}
        merged.update(user.get("base_profile") or {})

        if domain in ("diet", ""):
            merged.update(user.get("diet_profile") or {})
        if domain in ("workout", ""):
            merged.update(user.get("workout_profile") or {})

        # Include name from top-level if present
        if user.get("name"):
            merged.setdefault("name", user["name"])

        return merged
