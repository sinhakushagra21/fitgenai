"""
tests/conftest.py
─────────────────
Shared fixtures for FITGEN.AI test suite.
"""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from agent.persistence import init_db


@pytest.fixture
def sample_user_profile() -> dict:
    """Complete user profile with all diet + workout fields."""
    return {
        # Stats
        "name": "Test User",
        "age": 25,
        "sex": "male",
        "height_cm": 180.0,
        "weight_kg": 80.0,
        "goal": "fat loss",
        "goal_weight": "72",
        "weight_loss_pace": "steady & sustainable",
        # Lifestyle
        "job_type": "desk job",
        "exercise_frequency": "3-4 times",
        "exercise_type": "weights and running",
        "sleep_hours": 7,
        "stress_level": "moderate",
        "alcohol_intake": "4 beers per week",
        # Food Preferences
        "diet_preference": "omnivore",
        "favourite_meals": "Chicken tikka masala, Pasta carbonara, Steak and chips, Sushi, Thai green curry",
        "foods_to_avoid": "none",
        "allergies": "none",
        "cooking_style": "mix of all",
        "food_adventurousness": 7,
        # Snack Habits
        "current_snacks": "crisps, chocolate bars, biscuits",
        "snack_reason": "boredom",
        "snack_preference": "both",
        "late_night_snacking": "sometimes",
        # Workout-specific (new fields)
        "experience_level": "intermediate",
        "training_days_per_week": 5,
        "session_duration": 60,
        "daily_steps": 7000,
        "additional_info": "none",
        # Legacy workout fields (kept for backwards compat)
        "activity_level": "moderate",
        "fitness_level": "intermediate",
        "equipment": "full gym",
        "workout_days": 5,
    }


@pytest.fixture
def sample_agent_state(sample_user_profile) -> dict:
    """Valid AgentState dict."""
    return {
        "messages": [],
        "user_profile": sample_user_profile,
        "user_email": "test@fitgen.ai",
        "context_id": "test_ctx_001",
        "state_id": "test_ctx_001",
        "workflow": {},
        "calendar_sync_requested": False,
    }


@pytest.fixture
def tmp_db():
    """Patch persistence._connect to use an in-memory SQLite DB."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    with patch("agent.persistence._connect", return_value=conn):
        init_db()
        yield conn

    conn.close()


@pytest.fixture
def mock_tool_message():
    """Factory for creating ToolMessage-like objects."""
    from langchain_core.messages import ToolMessage

    def _make(assistant_message: str = "Test response", **state_updates):
        payload = {
            "assistant_message": assistant_message,
            "state_updates": {
                "context_id": "test_ctx",
                "state_id": "test_ctx",
                "user_email": "test@fitgen.ai",
                "workflow": {},
                "user_profile": {},
                **state_updates,
            },
        }
        return ToolMessage(
            content=json.dumps(payload),
            tool_call_id="call_test123",
        )

    return _make
