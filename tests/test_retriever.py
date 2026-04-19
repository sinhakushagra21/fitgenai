"""
tests/test_retriever.py
───────────────────────
PersonalRAG retriever tests.

We mock the OpenAI embedder and the rewrite FAST_MODEL so the tests
run offline and deterministically. mongomock has no $vectorSearch, so
the ANN call returns [] — but the anchor-chunk path still runs and we
assert anchors come through.
"""

from __future__ import annotations

from unittest.mock import patch

from bson import ObjectId

from agent.db.repositories.plan_chunks_repo import PlanChunksRepository
from agent.rag.personal.retriever import PersonalRAG, RetrievedChunk


def _seed_anchor(uid, *, plan_type="diet", section_type="macros"):
    PlanChunksRepository.upsert_many([
        {
            "user_id": uid,
            "plan_id": ObjectId(),
            "plan_type": plan_type,
            "plan_status": "confirmed",
            "plan_version": 1,
            "section_type": section_type,
            "day_of_week": None,
            "heading": "Macro Targets",
            "preamble": "",
            "chunk_text": "Protein 160 / Carbs 180 / Fat 70",
            "embedded_text": "Macros …",
            "chunk_tokens": 10,
            "embedding": [0.01] * 1536,
            "profile_snapshot_digest": {},
            "source_content_hash": f"anchor-{section_type}",
        }
    ])


class TestPersonalRAGRetrieve:
    def test_empty_on_missing_user_id(self, mock_mongo_db):
        assert PersonalRAG.retrieve(user_id="", query="anything") == []

    def test_empty_on_blank_query(self, mock_mongo_db):
        assert PersonalRAG.retrieve(user_id=str(ObjectId()), query="   ") == []

    def test_anchor_included_for_diet(self, mock_mongo_db):
        uid = ObjectId()
        _seed_anchor(uid, plan_type="diet", section_type="macros")

        with patch("agent.rag.personal.retriever.embed_query",
                   return_value=[0.01] * 1536), \
             patch("agent.rag.personal.retriever._rewrite_query",
                   return_value={"query": "macros?", "plan_type": "diet",
                                 "day_of_week": "", "section_hint": ""}):
            out = PersonalRAG.retrieve(
                user_id=str(uid), query="what are my macros?",
                plan_type="diet", include_memory=False,
            )

        assert len(out) >= 1
        assert any(c.section_type == "macros" for c in out)
        assert all(isinstance(c, RetrievedChunk) for c in out)

    def test_render_block_format(self, mock_mongo_db):
        c = RetrievedChunk(
            source="plan_chunks", score=0.9, plan_type="diet",
            section_type="macros", heading="Macro Targets",
            text="P 160 / C 180 / F 70",
        )
        r = c.render()
        assert "plan_chunks" in r
        assert "macros" in r
        assert "P 160" in r

    def test_fails_soft_on_embed_error(self, mock_mongo_db):
        uid = ObjectId()
        with patch("agent.rag.personal.retriever.embed_query",
                   side_effect=RuntimeError("boom")), \
             patch("agent.rag.personal.retriever._rewrite_query",
                   return_value={"query": "x", "plan_type": "",
                                 "day_of_week": "", "section_hint": ""}):
            out = PersonalRAG.retrieve(user_id=str(uid), query="hi")
        assert out == []
