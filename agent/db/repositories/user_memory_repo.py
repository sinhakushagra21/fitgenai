"""
agent/db/repositories/user_memory_repo.py
─────────────────────────────────────────
Repository for the ``user_memory`` collection used by Personal RAG.

Stores normalized, embedded snippets of durable personal context:
profile-field groups, extracted preferences, feedback notes. Retrieved
alongside plan chunks to personalize answers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import UpdateOne
from pymongo.collection import Collection

from agent.db.mongo import get_db

logger = logging.getLogger("fitgen.db.user_memory_repo")

VECTOR_INDEX_NAME = "user_memory_vec"


class UserMemoryRepository:
    """CRUD + vector search for ``user_memory``."""

    @staticmethod
    def _col() -> Collection:
        return get_db().user_memory

    # ── Writes ───────────────────────────────────────────────────

    @classmethod
    def upsert_profile_group(
        cls,
        *,
        user_id: ObjectId | str,
        domain: str,
        content: str,
        structured: dict[str, Any],
        embedding: list[float],
        source_content_hash: str,
    ) -> None:
        """Upsert the single ``memory_type=profile_group`` doc for a
        ``(user_id, domain)`` pair. Replaces any existing entry.
        """
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        now = datetime.now(timezone.utc)
        cls._col().update_one(
            {
                "user_id": user_id,
                "memory_type": "profile_group",
                "domain": domain,
            },
            {
                "$set": {
                    "user_id": user_id,
                    "memory_type": "profile_group",
                    "domain": domain,
                    "content": content,
                    "structured": structured,
                    "embedding": embedding,
                    "source": "profile",
                    "source_content_hash": source_content_hash,
                    "confidence": 1.0,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    @classmethod
    def upsert_many(cls, docs: list[dict[str, Any]]) -> int:
        if not docs:
            return 0
        now = datetime.now(timezone.utc)
        col = cls._col()

        ops: list[UpdateOne] = []
        prepared: list[tuple[dict, dict]] = []
        for d in docs:
            d = dict(d)
            d.setdefault("created_at", now)
            d["updated_at"] = now
            key = {
                "user_id": d["user_id"],
                "memory_type": d["memory_type"],
                "source_content_hash": d["source_content_hash"],
            }
            prepared.append((key, d))
            ops.append(UpdateOne(key, {"$set": d}, upsert=True))

        try:
            result = col.bulk_write(ops, ordered=False)
            return (result.upserted_count or 0) + (result.modified_count or 0)
        except TypeError:
            # mongomock fallback
            upserted, modified = 0, 0
            for key, d in prepared:
                r = col.update_one(key, {"$set": d}, upsert=True)
                if r.upserted_id is not None:
                    upserted += 1
                else:
                    modified += r.modified_count or 0
            return upserted + modified

    @classmethod
    def delete_by_user(cls, user_id: ObjectId | str) -> int:
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        return cls._col().delete_many({"user_id": user_id}).deleted_count

    # ── Reads ────────────────────────────────────────────────────

    @classmethod
    def find_by_user(
        cls, user_id: ObjectId | str, *, memory_type: str | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        query: dict[str, Any] = {"user_id": user_id}
        if memory_type:
            query["memory_type"] = memory_type
        return list(cls._col().find(query))

    @classmethod
    def vector_search(
        cls,
        *,
        user_id: ObjectId | str,
        query_vector: list[float],
        memory_type: str | None = None,
        domain: str | None = None,
        num_candidates: int = 50,
        limit: int = 5,
        index_name: str = VECTOR_INDEX_NAME,
    ) -> list[dict[str, Any]]:
        """Atlas $vectorSearch with mandatory user_id filter."""
        if not user_id:
            raise ValueError("user_id is required for user_memory vector search")
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)

        filter_clause: dict[str, Any] = {"user_id": user_id}
        if memory_type:
            filter_clause["memory_type"] = memory_type
        if domain:
            filter_clause["domain"] = domain

        pipeline = [
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
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        ]

        try:
            return list(cls._col().aggregate(pipeline))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "user_memory.vector_search failed: %s (filter=%s)",
                exc, filter_clause,
            )
            return []
