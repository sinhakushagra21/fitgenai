"""
tests/test_persistence.py
─────────────────────────
Unit tests for SQLite persistence layer.
All tests use the tmp_db fixture (in-memory SQLite).
"""

import json

import pytest

from agent.persistence import (
    delete_context_state,
    delete_record,
    get_context_state,
    get_latest_context_state_by_email,
    get_record,
    init_db,
    update_calendar_sync,
    upsert_context_state,
    upsert_record,
)


class TestUserRecords:
    def test_init_db_creates_tables(self, tmp_db):
        tables = tmp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row["name"] for row in tables}
        assert "user_records" in table_names
        assert "context_states" in table_names

    def test_upsert_and_get_record(self, tmp_db):
        upsert_record(
            state_id="rec_001",
            domain="workout",
            profile={"age": 25, "goal": "muscle gain"},
            plan_text="Test workout plan",
        )
        record = get_record("rec_001")
        assert record is not None
        assert record["domain"] == "workout"
        assert record["profile"]["age"] == 25
        assert record["plan_text"] == "Test workout plan"
        assert record["calendar_sync"] is False

    def test_upsert_updates_existing(self, tmp_db):
        upsert_record(state_id="rec_002", domain="diet", profile={}, plan_text="v1")
        upsert_record(state_id="rec_002", domain="diet", profile={}, plan_text="v2")
        record = get_record("rec_002")
        assert record["plan_text"] == "v2"

    def test_get_nonexistent_returns_none(self, tmp_db):
        assert get_record("nonexistent") is None

    def test_delete_existing_record(self, tmp_db):
        upsert_record(state_id="rec_del", domain="workout", profile={}, plan_text="x")
        assert delete_record("rec_del") is True
        assert get_record("rec_del") is None

    def test_delete_nonexistent_returns_false(self, tmp_db):
        assert delete_record("nope") is False

    def test_update_calendar_sync(self, tmp_db):
        upsert_record(state_id="rec_cal", domain="workout", profile={}, plan_text="x")
        update_calendar_sync("rec_cal", True)
        record = get_record("rec_cal")
        assert record["calendar_sync"] is True


class TestContextStates:
    def test_upsert_and_get_context_state(self, tmp_db):
        upsert_context_state(
            context_id="ctx_001",
            user_email="user@test.com",
            user_profile={"age": 30},
            workflow={"intent": "create"},
            calendar_sync_requested=False,
        )
        ctx = get_context_state("ctx_001")
        assert ctx is not None
        assert ctx["user_email"] == "user@test.com"
        assert ctx["user_profile"]["age"] == 30
        assert ctx["workflow"]["intent"] == "create"

    def test_get_nonexistent_context(self, tmp_db):
        assert get_context_state("nope") is None

    def test_get_latest_by_email(self, tmp_db):
        upsert_context_state(
            context_id="ctx_old",
            user_email="user@test.com",
            user_profile={"version": 1},
            workflow={},
            calendar_sync_requested=False,
        )
        # Force a later timestamp by updating ctx_new after a manual timestamp bump
        import time
        time.sleep(1.1)
        upsert_context_state(
            context_id="ctx_new",
            user_email="user@test.com",
            user_profile={"version": 2},
            workflow={},
            calendar_sync_requested=True,
        )
        latest = get_latest_context_state_by_email("user@test.com")
        assert latest is not None
        assert latest["context_id"] == "ctx_new"

    def test_get_latest_by_email_empty(self, tmp_db):
        assert get_latest_context_state_by_email("") is None
        assert get_latest_context_state_by_email("nobody@test.com") is None

    def test_delete_context_state(self, tmp_db):
        upsert_context_state(
            context_id="ctx_del",
            user_email="",
            user_profile={},
            workflow={},
            calendar_sync_requested=False,
        )
        assert delete_context_state("ctx_del") is True
        assert get_context_state("ctx_del") is None
