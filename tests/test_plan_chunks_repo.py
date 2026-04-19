"""
tests/test_plan_chunks_repo.py
──────────────────────────────
Unit tests for PlanChunksRepository (non-vector operations + tenant
guardrail). Atlas $vectorSearch is not supported by mongomock, so the
retrieval tests only assert that vector_search fails safely (empty
result) — the tenant-isolation check is the important invariant.
"""

from __future__ import annotations

import pytest
from bson import ObjectId

from agent.db.repositories.plan_chunks_repo import PlanChunksRepository


def _make_doc(user_id, plan_id, *, section_type="macros",
              plan_type="diet", plan_status="draft", plan_version=1,
              chash="h1", heading="Macros") -> dict:
    return {
        "user_id": user_id,
        "plan_id": plan_id,
        "plan_type": plan_type,
        "plan_status": plan_status,
        "plan_version": plan_version,
        "section_type": section_type,
        "day_of_week": None,
        "heading": heading,
        "preamble": "",
        "chunk_text": "P 160 / C 180 / F 70",
        "embedded_text": "Macros …",
        "chunk_tokens": 10,
        "embedding": [0.0] * 1536,
        "profile_snapshot_digest": {},
        "source_content_hash": chash,
    }


class TestUpsertMany:
    def test_insert_then_dedupe(self, mock_mongo_db):
        uid, pid = ObjectId(), ObjectId()
        docs = [_make_doc(uid, pid, chash="a"),
                _make_doc(uid, pid, chash="b", section_type="calorie_calc")]
        n = PlanChunksRepository.upsert_many(docs)
        assert n >= 2
        # Re-running with the same hashes is a no-op (idempotent).
        n2 = PlanChunksRepository.upsert_many(docs)
        # mongomock may report modified==0 second time → still safe
        assert n2 >= 0
        assert len(PlanChunksRepository.find_by_plan(pid)) == 2

    def test_empty_list(self, mock_mongo_db):
        assert PlanChunksRepository.upsert_many([]) == 0


class TestStatusAndArchive:
    def test_archive_by_plan(self, mock_mongo_db):
        uid, pid = ObjectId(), ObjectId()
        PlanChunksRepository.upsert_many([_make_doc(uid, pid)])
        PlanChunksRepository.archive_by_plan(pid)
        docs = PlanChunksRepository.find_by_plan(pid)
        assert all(d["plan_status"] == "archived" for d in docs)

    def test_archive_older_versions(self, mock_mongo_db):
        uid, pid = ObjectId(), ObjectId()
        PlanChunksRepository.upsert_many([
            _make_doc(uid, pid, plan_version=1, chash="v1"),
            _make_doc(uid, pid, plan_version=2, chash="v2"),
        ])
        PlanChunksRepository.archive_older_versions(pid, current_version=2)
        by_ver = {d["plan_version"]: d["plan_status"]
                  for d in PlanChunksRepository.find_by_plan(pid)}
        assert by_ver[1] == "archived"
        assert by_ver[2] == "draft"


class TestFindAnchor:
    def test_find_anchor_latest(self, mock_mongo_db):
        uid, pid = ObjectId(), ObjectId()
        PlanChunksRepository.upsert_many([
            _make_doc(uid, pid, section_type="macros", chash="m1"),
        ])
        doc = PlanChunksRepository.find_anchor(
            uid, plan_type="diet", section_type="macros",
        )
        assert doc is not None
        assert doc["section_type"] == "macros"


class TestVectorSearchGuardrail:
    def test_requires_user_id(self, mock_mongo_db):
        with pytest.raises(ValueError):
            PlanChunksRepository.vector_search(
                user_id="", query_vector=[0.0] * 1536,
            )

    def test_safe_on_mongomock(self, mock_mongo_db):
        # mongomock doesn't support $vectorSearch — repo swallows the
        # error and returns []. That's the contract.
        uid = ObjectId()
        out = PlanChunksRepository.vector_search(
            user_id=uid, query_vector=[0.01] * 1536, plan_type="diet",
        )
        assert out == []


class TestDelete:
    def test_delete_by_plan(self, mock_mongo_db):
        uid, pid = ObjectId(), ObjectId()
        PlanChunksRepository.upsert_many([_make_doc(uid, pid)])
        assert PlanChunksRepository.delete_by_plan(pid) == 1
        assert PlanChunksRepository.find_by_plan(pid) == []

    def test_delete_by_user(self, mock_mongo_db):
        uid = ObjectId()
        PlanChunksRepository.upsert_many([
            _make_doc(uid, ObjectId(), chash="x"),
            _make_doc(uid, ObjectId(), chash="y"),
        ])
        assert PlanChunksRepository.delete_by_user(uid) == 2
