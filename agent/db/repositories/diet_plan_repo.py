"""
agent/db/repositories/diet_plan_repo.py
───────────────────────────────────────
Repository for the ``diet_plans`` collection.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo.collection import Collection

from agent.db.mongo import get_db
from agent.db.models import PlanDocument

logger = logging.getLogger("fitgen.db.diet_plan_repo")


def _reindex(plan_id: Any) -> None:
    """Best-effort Personal-RAG re-index hook. Never raises."""
    try:
        from agent.rag.personal.indexer import index_plan
        index_plan(str(plan_id), plan_type="diet")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[diet_plan_repo] RAG reindex failed: %s", exc)


def _archive_siblings(
    user_id: ObjectId, *, except_plan_id: ObjectId | None = None,
) -> int:
    """Archive every non-archived diet plan for a user.

    One-active-plan-per-domain is FITGEN's product invariant: a new
    ``create`` or ``confirm`` archives the previous plan (both its
    ``diet_plans`` doc and its ``plan_chunks`` vector rows) so RAG
    retrieval only sees the current plan.
    """
    from agent.db.mongo import get_db
    from datetime import datetime, timezone

    q: dict[str, Any] = {
        "user_id": user_id,
        "status": {"$in": ["draft", "confirmed"]},
    }
    if except_plan_id is not None:
        q["_id"] = {"$ne": except_plan_id}

    sibling_ids = [p["_id"] for p in get_db().diet_plans.find(q, {"_id": 1})]
    if not sibling_ids:
        return 0

    # Update the diet_plans docs in one shot.
    get_db().diet_plans.update_many(
        {"_id": {"$in": sibling_ids}},
        {"$set": {"status": "archived",
                  "updated_at": datetime.now(timezone.utc)}},
    )

    # Archive their plan_chunks too (keeps RAG in sync).
    try:
        from agent.db.repositories.plan_chunks_repo import PlanChunksRepository
        for pid in sibling_ids:
            PlanChunksRepository.archive_by_plan(pid)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[diet_plan_repo] sibling chunk archive failed: %s", exc)

    logger.info(
        "[diet_plan_repo] archived %d sibling diet plan(s) for user=%s",
        len(sibling_ids), user_id,
    )
    return len(sibling_ids)


class DietPlanRepository:
    """CRUD operations for the ``diet_plans`` collection."""

    @staticmethod
    def _col() -> Collection:
        return get_db().diet_plans

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
        name: str = "",
    ) -> ObjectId:
        """Insert a new diet plan.  Returns the inserted ``_id``."""
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)

        # FITGEN product invariant: one active diet plan per user.
        # Archive any prior non-archived plans before inserting the new
        # one so RAG retrieval (which filters on plan_status) only ever
        # surfaces the current plan.
        _archive_siblings(user_id)

        doc = PlanDocument(
            user_id=user_id,
            session_id=session_id,
            name=name,
            profile_snapshot=profile_snapshot,
            plan_markdown=plan_markdown,
            structured_data=structured_data or {},
            status=status,
        )
        result = cls._col().insert_one(doc.model_dump())
        logger.info("Diet plan created: _id=%s user=%s name=%r", result.inserted_id, user_id, name)
        _reindex(result.inserted_id)
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
        """Return the most recent diet plan for a user, optionally filtered by status."""
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
        """Return all diet plans for a user, most recent first."""
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
        name: str | None = None,
        calendar_synced: bool | None = None,
        fit_synced: bool | None = None,
    ) -> None:
        """Update specific fields of a diet plan."""
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
        if name is not None:
            set_fields["name"] = name
        if calendar_synced is not None:
            set_fields["calendar_synced"] = calendar_synced
        if fit_synced is not None:
            set_fields["fit_synced"] = fit_synced

        cls._col().update_one(
            {"_id": plan_id},
            {"$set": set_fields, "$inc": {"version": 1}},
        )
        _reindex(plan_id)

    @classmethod
    def confirm(cls, plan_id: ObjectId | str) -> None:
        """Mark a plan as confirmed."""
        cls.update_plan(plan_id, status="confirmed")

    @classmethod
    def archive(cls, plan_id: ObjectId | str) -> None:
        """Archive a plan (soft delete)."""
        cls.update_plan(plan_id, status="archived")
        # Also archive chunks in the vector collection.
        try:
            from agent.db.repositories.plan_chunks_repo import PlanChunksRepository
            PlanChunksRepository.archive_by_plan(plan_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[diet_plan_repo] archive chunks failed: %s", exc)

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
