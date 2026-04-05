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
    """Complete 13-field user profile."""
    return {
        "name": "Test User",
        "age": 25,
        "sex": "male",
        "height_cm": 180.0,
        "weight_kg": 80.0,
        "goal": "muscle gain",
        "activity_level": "moderate",
        "diet_preference": "omnivore",
        "foods_to_avoid": "none",
        "allergies": "none",
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
