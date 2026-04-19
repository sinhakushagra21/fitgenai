"""
agent/db/repositories/plan_chunks_repo.py
─────────────────────────────────────────
Repository for the ``plan_chunks`` collection used by Personal RAG.

Stores one document per markdown section of a diet or workout plan
together with its 1536-dim embedding. Atlas Vector Search (index name:
``plan_chunks_vec``) provides the ANN retrieval. This repo exposes:

  * upsert_many       — bulk insert/replace with idempotency via
                        ``source_content_hash``
  * archive_by_plan   — mark all chunks for a plan as archived
  * set_status        — update ``plan_status`` for all chunks of a plan
  * delete_by_plan    — hard-delete (used by backfill tests)
  * vector_search     — Atlas $vectorSearch with mandatory user_id
                        tenant-isolation guardrail
  * find_anchor       — fetch the latest "macros" / "calorie_calc" /
                        "split_overview" chunk for a user (used by the
                        retriever to always include macro context)
  * exists_hash       — O(1) idempotency check used during indexing
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import UpdateOne
from pymongo.collection import Collection

from agent.db.mongo import get_db

logger = logging.getLogger("fitgen.db.plan_chunks_repo")

# Default Atlas vector-search index name. Override via env if needed.
VECTOR_INDEX_NAME = "plan_chunks_vec"

# Statuses retrievable by default (archived is excluded).
_DEFAULT_STATUS_FILTER = ["draft", "confirmed"]


class PlanChunksRepository:
    """CRUD + vector search for ``plan_chunks``."""

    @staticmethod
    def _col() -> Collection:
        return get_db().plan_chunks

    # ── Writes ───────────────────────────────────────────────────

    @classmethod
    def upsert_many(cls, docs: list[dict[str, Any]]) -> int:
        """Bulk upsert chunks keyed on (plan_id, source_content_hash).

        If a chunk with the same hash already exists for the plan, the
        write is a no-op (saves an embedding round-trip cost on re-index).
        Returns the number of documents written (inserted + modified).
        """
        if not docs:
            return 0
        now = datetime.now(timezone.utc)
        col = cls._col()

        # Try the fast path first (bulk_write). Fall back to a per-doc
        # loop on environments (e.g. mongomock) that don't accept newer
        # pymongo kwargs in UpdateOne.
        ops: list[UpdateOne] = []
        prepared: list[tuple[dict, dict]] = []
        for d in docs:
            d = dict(d)
            d.setdefault("created_at", now)
            d["updated_at"] = now
            key = {
                "plan_id": d["plan_id"],
                "source_content_hash": d["source_content_hash"],
            }
            prepared.append((key, d))
            ops.append(
                UpdateOne(
                    key,
                    {"$set": d, "$setOnInsert": {"_inserted_at": now}},
                    upsert=True,
                )
            )

        try:
            result = col.bulk_write(ops, ordered=False)
            upserted = result.upserted_count or 0
            modified = result.modified_count or 0
        except TypeError:
            # mongomock compat fallback.
            upserted, modified = 0, 0
            for key, d in prepared:
                r = col.update_one(
                    key,
                    {"$set": d, "$setOnInsert": {"_inserted_at": now}},
                    upsert=True,
                )
                if r.upserted_id is not None:
                    upserted += 1
                else:
                    modified += r.modified_count or 0

        written = upserted + modified
        logger.info(
            "plan_chunks.upsert_many: written=%d (upserted=%d, modified=%d) total=%d",
            written, upserted, modified, len(docs),
        )
        return written

    @classmethod
    def archive_by_plan(cls, plan_id: ObjectId | str) -> int:
        """Set ``plan_status='archived'`` for every chunk of a plan."""
        return cls.set_status(plan_id, "archived")

    @classmethod
    def reactivate_by_plan(
        cls, plan_id: ObjectId | str, status: str = "confirmed",
    ) -> int:
        """Flip archived chunks back to ``status`` (used by restore flow)."""
        return cls.set_status(plan_id, status)

    @classmethod
    def set_status(cls, plan_id: ObjectId | str, status: str) -> int:
        """Update ``plan_status`` for all chunks of a given plan."""
        if isinstance(plan_id, str):
            plan_id = ObjectId(plan_id)
        result = cls._col().update_many(
            {"plan_id": plan_id},
            {
                "$set": {
                    "plan_status": status,
                    "updated_at": datetime.now(timezone.utc),
                },
            },
        )
        logger.info(
            "plan_chunks.set_status: plan=%s status=%s modified=%d",
            plan_id, status, result.modified_count,
        )
        return result.modified_count

    @classmethod
    def archive_older_versions(
        cls, plan_id: ObjectId | str, current_version: int,
    ) -> int:
        """Archive chunks of this plan whose ``plan_version`` is stale."""
        if isinstance(plan_id, str):
            plan_id = ObjectId(plan_id)
        result = cls._col().update_many(
            {"plan_id": plan_id, "plan_version": {"$lt": current_version}},
            {
                "$set": {
                    "plan_status": "archived",
                    "updated_at": datetime.now(timezone.utc),
                },
            },
        )
        return result.modified_count

    @classmethod
    def delete_by_plan(cls, plan_id: ObjectId | str) -> int:
        """Hard-delete all chunks for a plan (used by tests / re-index)."""
        if isinstance(plan_id, str):
            plan_id = ObjectId(plan_id)
        return cls._col().delete_many({"plan_id": plan_id}).deleted_count

    @classmethod
    def delete_by_user(cls, user_id: ObjectId | str) -> int:
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        return cls._col().delete_many({"user_id": user_id}).deleted_count

    # ── Reads ────────────────────────────────────────────────────

    @classmethod
    def exists_hash(
        cls, plan_id: ObjectId | str, source_content_hash: str,
    ) -> bool:
        if isinstance(plan_id, str):
            plan_id = ObjectId(plan_id)
        return cls._col().count_documents(
            {"plan_id": plan_id, "source_content_hash": source_content_hash},
            limit=1,
        ) > 0

    @classmethod
    def find_by_plan(cls, plan_id: ObjectId | str) -> list[dict[str, Any]]:
        if isinstance(plan_id, str):
            plan_id = ObjectId(plan_id)
        return list(cls._col().find({"plan_id": plan_id}))

    @classmethod
    def find_anchor(
        cls,
        user_id: ObjectId | str,
        *,
        plan_type: str,
        section_type: str,
        statuses: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Return the most recent chunk of a given section_type for a user.

        Used by the retriever to always include a "macros" / "calorie_calc"
        / "split_overview" anchor chunk alongside vector-search results.
        """
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        query: dict[str, Any] = {
            "user_id": user_id,
            "plan_type": plan_type,
            "section_type": section_type,
            "plan_status": {"$in": statuses or _DEFAULT_STATUS_FILTER},
        }
        return cls._col().find_one(query, sort=[("updated_at", -1)])

    @classmethod
    def vector_search(
        cls,
        *,
        user_id: ObjectId | str,
        query_vector: list[float],
        plan_type: str | None = None,
        section_type: str | None = None,
        day_of_week: str | None = None,
        statuses: list[str] | None = None,
        num_candidates: int = 100,
        limit: int = 20,
        index_name: str = VECTOR_INDEX_NAME,
    ) -> list[dict[str, Any]]:
        """Atlas $vectorSearch with mandatory user_id filter.

        ``user_id`` is non-optional — this is the tenant-isolation
        invariant that tests assert. Additional filters narrow the
        candidate pool before ANN.
        """
        if not user_id:
            raise ValueError("user_id is required for plan_chunks vector search")
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)

        filter_clause: dict[str, Any] = {
            "user_id": user_id,
            "plan_status": {"$in": statuses or _DEFAULT_STATUS_FILTER},
        }
        if plan_type:
            filter_clause["plan_type"] = plan_type
        if section_type:
            filter_clause["section_type"] = section_type
        if day_of_week:
            filter_clause["day_of_week"] = day_of_week

        pipeline: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": num_candidates,
                    "limit": limit,
                    "filter": filter_clause,
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        try:
            return list(cls._col().aggregate(pipeline))
        except Exception as exc:  # noqa: BLE001
            # Atlas unavailable (e.g. local mongomock) — caller should
            # handle empty results gracefully.
            logger.warning(
                "plan_chunks.vector_search failed: %s (filter=%s, limit=%d)",
                exc, filter_clause, limit,
            )
            return []
