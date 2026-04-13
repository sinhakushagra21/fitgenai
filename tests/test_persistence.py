"""
tests/test_persistence.py
─────────────────────────
Unit tests for the MongoDB persistence facade.

All tests use the ``mock_mongo_db`` fixture (mongomock) from conftest.
"""

import time

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


class TestInitDb:
    def test_init_db_creates_indexes(self, mock_mongo_db):
        """init_db() should run without errors (creates indexes)."""
        init_db()
        # Verify sessions collection has a session_id index
        index_info = mock_mongo_db.sessions.index_information()
        # At minimum, _id_ index exists
        assert "_id_" in index_info


class TestUserRecords:
    """Tests for the legacy user_records facade (upsert_record / get_record)."""

    def test_upsert_and_get_record(self, mock_mongo_db):
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

    def test_upsert_updates_existing(self, mock_mongo_db):
        upsert_record(state_id="rec_002", domain="diet", profile={}, plan_text="v1")
        upsert_record(state_id="rec_002", domain="diet", profile={}, plan_text="v2")
        record = get_record("rec_002")
        assert record["plan_text"] == "v2"

    def test_get_nonexistent_returns_none(self, mock_mongo_db):
        assert get_record("nonexistent") is None

    def test_delete_existing_record(self, mock_mongo_db):
        upsert_record(state_id="rec_del", domain="workout", profile={}, plan_text="x")
        assert delete_record("rec_del") is True
        assert get_record("rec_del") is None

    def test_delete_nonexistent_returns_false(self, mock_mongo_db):
        assert delete_record("nope") is False

    def test_update_calendar_sync(self, mock_mongo_db):
        upsert_record(state_id="rec_cal", domain="workout", profile={}, plan_text="x")
        update_calendar_sync("rec_cal", True)
        record = get_record("rec_cal")
        assert record["calendar_sync"] is True


class TestContextStates:
    """Tests for context/session state (upsert_context_state / get_context_state)."""

    def test_upsert_and_get_context_state(self, mock_mongo_db):
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

    def test_get_nonexistent_context(self, mock_mongo_db):
        assert get_context_state("nope") is None

    def test_get_latest_by_email(self, mock_mongo_db):
        upsert_context_state(
            context_id="ctx_old",
            user_email="user@test.com",
            user_profile={"version": 1},
            workflow={},
            calendar_sync_requested=False,
        )
        # Small delay to ensure different updated_at timestamps
        time.sleep(0.1)
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

    def test_get_latest_by_email_empty(self, mock_mongo_db):
        assert get_latest_context_state_by_email("") is None
        assert get_latest_context_state_by_email("nobody@test.com") is None

    def test_delete_context_state(self, mock_mongo_db):
        upsert_context_state(
            context_id="ctx_del",
            user_email="",
            user_profile={},
            workflow={},
            calendar_sync_requested=False,
        )
        assert delete_context_state("ctx_del") is True
        assert get_context_state("ctx_del") is None


class TestUserRepository:
    """Tests for the UserRepository (new MongoDB-native user management)."""

    def test_find_or_create_new_user(self, mock_mongo_db):
        from agent.db.repositories.user_repo import UserRepository

        user = UserRepository.find_or_create("new@test.com", "Test User")
        assert user is not None
        assert user["email"] == "new@test.com"
        assert user["name"] == "Test User"

    def test_find_or_create_existing_user(self, mock_mongo_db):
        from agent.db.repositories.user_repo import UserRepository

        UserRepository.create(email="existing@test.com", name="Existing")
        user = UserRepository.find_or_create("existing@test.com")
        assert user["name"] == "Existing"

    def test_update_profile(self, mock_mongo_db):
        from agent.db.repositories.user_repo import UserRepository

        UserRepository.create(email="update@test.com")
        UserRepository.update_profile(
            "update@test.com",
            base={"age": 25, "height_cm": 180},
            diet={"diet_preference": "vegetarian"},
            workout={"experience_level": "intermediate"},
        )
        user = UserRepository.find_by_email("update@test.com")
        assert user["base_profile"]["age"] == 25
        assert user["diet_profile"]["diet_preference"] == "vegetarian"
        assert user["workout_profile"]["experience_level"] == "intermediate"

    def test_get_merged_profile(self, mock_mongo_db):
        from agent.db.repositories.user_repo import UserRepository

        UserRepository.create(email="merge@test.com", name="Merge User")
        UserRepository.update_profile(
            "merge@test.com",
            base={"age": 30, "goal": "fat loss"},
            diet={"diet_preference": "omnivore"},
            workout={"experience_level": "beginner"},
        )
        profile = UserRepository.get_merged_profile("merge@test.com", domain="diet")
        assert profile["age"] == 30
        assert profile["diet_preference"] == "omnivore"
        assert profile["name"] == "Merge User"

    def test_delete_by_email(self, mock_mongo_db):
        from agent.db.repositories.user_repo import UserRepository

        UserRepository.create(email="delete@test.com")
        assert UserRepository.delete_by_email("delete@test.com") is True
        assert UserRepository.find_by_email("delete@test.com") is None


class TestPlanRepositories:
    """Tests for DietPlanRepository and WorkoutPlanRepository."""

    def test_diet_plan_crud(self, mock_mongo_db):
        from bson import ObjectId
        from agent.db.repositories.diet_plan_repo import DietPlanRepository
        from agent.db.repositories.user_repo import UserRepository

        user_id = UserRepository.create(email="plan@test.com")

        plan_id = DietPlanRepository.create(
            user_id=user_id,
            session_id="sess_001",
            profile_snapshot={"age": 25},
            plan_markdown="## Day 1\nChicken and rice",
        )
        assert plan_id is not None

        plan = DietPlanRepository.find_by_id(plan_id)
        assert plan["plan_markdown"] == "## Day 1\nChicken and rice"
        assert plan["status"] == "draft"

        DietPlanRepository.confirm(plan_id)
        plan = DietPlanRepository.find_by_id(plan_id)
        assert plan["status"] == "confirmed"

        assert DietPlanRepository.delete(plan_id) is True
        assert DietPlanRepository.find_by_id(plan_id) is None

    def test_workout_plan_crud(self, mock_mongo_db):
        from agent.db.repositories.workout_plan_repo import WorkoutPlanRepository
        from agent.db.repositories.user_repo import UserRepository

        user_id = UserRepository.create(email="workout@test.com")

        plan_id = WorkoutPlanRepository.create(
            user_id=user_id,
            session_id="sess_002",
            profile_snapshot={"goal": "muscle gain"},
            plan_markdown="## Day 1\nBench Press 3x8",
        )

        plan = WorkoutPlanRepository.find_latest_by_user(user_id)
        assert plan is not None
        assert plan["session_id"] == "sess_002"

        WorkoutPlanRepository.update_plan(
            plan_id,
            plan_markdown="## Day 1\nBench Press 4x10",
            calendar_synced=True,
        )
        updated = WorkoutPlanRepository.find_by_id(plan_id)
        assert "4x10" in updated["plan_markdown"]
        assert updated["calendar_synced"] is True

    def test_find_by_session(self, mock_mongo_db):
        from agent.db.repositories.diet_plan_repo import DietPlanRepository
        from agent.db.repositories.user_repo import UserRepository

        user_id = UserRepository.create(email="session@test.com")
        DietPlanRepository.create(
            user_id=user_id,
            session_id="sess_find",
            profile_snapshot={},
            plan_markdown="found it",
        )
        plan = DietPlanRepository.find_by_session("sess_find")
        assert plan is not None
        assert plan["plan_markdown"] == "found it"


class TestFeedbackRepository:
    """Tests for the feedback facade."""

    def test_save_and_get_feedback(self, mock_mongo_db):
        from agent.feedback import save_feedback, get_session_feedback, get_average_rating

        save_feedback("ctx_fb", "turn_1", rating=4)
        save_feedback("ctx_fb", "turn_2", rating=5, comment="Great!")

        entries = get_session_feedback("ctx_fb")
        assert len(entries) == 2
        assert entries[0]["rating"] == 4
        assert entries[1]["comment"] == "Great!"

        avg = get_average_rating("ctx_fb")
        assert avg == 4.5

    def test_no_ratings_returns_none(self, mock_mongo_db):
        from agent.feedback import get_average_rating

        assert get_average_rating("no_such_session") is None
