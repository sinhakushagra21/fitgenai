"""
agent/db/repositories/workout_plan_repo.py
──────────────────────────────────────────
Repository for the ``workout_plans`` collection.

Identical structure to DietPlanRepository — separate collection for
cleaner querying and indexing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo.collection import Collection

from agent.db.mongo import get_db
from agent.db.models import PlanDocument

logger = logging.getLogger("fitgen.db.workout_plan_repo")


class WorkoutPlanRepository:
    """CRUD operations for the ``workout_plans`` collection."""

    @staticmethod
    def _col() -> Collection:
        return get_db().workout_plans

    # ── Create ───────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        *,
        user_id: ObjectId | str,
        session_id: str,
        profile_snapshot: dict[str, Any],
        plan_markdown: str,
        structured_data: dict[str, Any] | None = None,
        status: str = "draft",
    ) -> ObjectId:
        """Insert a new workout plan.  Returns the inserted ``_id``."""
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        doc = PlanDocument(
            user_id=user_id,
            session_id=session_id,
            profile_snapshot=profile_snapshot,
            plan_markdown=plan_markdown,
            structured_data=structured_data or {},
            status=status,
        )
        result = cls._col().insert_one(doc.model_dump())
        logger.info("Workout plan created: _id=%s user=%s", result.inserted_id, user_id)
        return result.inserted_id

    # ── Read ─────────────────────────────────────────────────────

    @classmethod
    def find_by_id(cls, plan_id: ObjectId | str) -> dict[str, Any] | None:
        if isinstance(plan_id, str):
            plan_id = ObjectId(plan_id)
        return cls._col().find_one({"_id": plan_id})

    @classmethod
    def find_latest_by_user(
        cls,
        user_id: ObjectId | str,
        *,
        status: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the most recent workout plan for a user."""
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        query: dict[str, Any] = {"user_id": user_id}
        if status:
            query["status"] = status
        return cls._col().find_one(query, sort=[("updated_at", -1)])

    @classmethod
    def find_by_session(cls, session_id: str) -> dict[str, Any] | None:
        """Return the plan associated with a given session."""
        return cls._col().find_one({"session_id": session_id})

    @classmethod
    def find_all_by_user(
        cls,
        user_id: ObjectId | str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return all workout plans for a user, most recent first."""
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        cursor = (
            cls._col()
            .find({"user_id": user_id})
            .sort("updated_at", -1)
            .limit(limit)
        )
        return list(cursor)

    # ── Update ───────────────────────────────────────────────────

    @classmethod
    def update_plan(
        cls,
        plan_id: ObjectId | str,
        *,
        plan_markdown: str | None = None,
        profile_snapshot: dict[str, Any] | None = None,
        status: str | None = None,
        calendar_synced: bool | None = None,
        fit_synced: bool | None = None,
    ) -> None:
        """Update specific fields of a workout plan."""
        if isinstance(plan_id, str):
            plan_id = ObjectId(plan_id)
        now = datetime.now(timezone.utc)
        set_fields: dict[str, Any] = {"updated_at": now}
        if plan_markdown is not None:
            set_fields["plan_markdown"] = plan_markdown
        if profile_snapshot is not None:
            set_fields["profile_snapshot"] = profile_snapshot
        if status is not None:
            set_fields["status"] = status
        if calendar_synced is not None:
            set_fields["calendar_synced"] = calendar_synced
        if fit_synced is not None:
            set_fields["fit_synced"] = fit_synced

        cls._col().update_one(
            {"_id": plan_id},
            {"$set": set_fields, "$inc": {"version": 1}},
        )

    @classmethod
    def confirm(cls, plan_id: ObjectId | str) -> None:
        """Mark a plan as confirmed."""
        cls.update_plan(plan_id, status="confirmed")

    @classmethod
    def archive(cls, plan_id: ObjectId | str) -> None:
        """Archive a plan (soft delete)."""
        cls.update_plan(plan_id, status="archived")

    # ── Delete ───────────────────────────────────────────────────

    @classmethod
    def delete(cls, plan_id: ObjectId | str) -> bool:
        """Hard-delete a plan.  Returns True if removed."""
        if isinstance(plan_id, str):
            plan_id = ObjectId(plan_id)
        result = cls._col().delete_one({"_id": plan_id})
        return result.deleted_count > 0

    @classmethod
    def delete_all_by_user(cls, user_id: ObjectId | str) -> int:
        """Delete all plans for a user.  Returns count deleted."""
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        result = cls._col().delete_many({"user_id": user_id})
        return result.deleted_count
