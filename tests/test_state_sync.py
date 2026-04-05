"""
tests/test_state_sync.py
────────────────────────
Unit tests for state_sync.apply_tool_state_updates.
"""

import json
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.state_sync import apply_tool_state_updates


def _make_state(tool_msg_content: str | None = None, **extra_state) -> dict:
    """Build a minimal AgentState with an optional ToolMessage."""
    messages = [HumanMessage(content="test query")]
    if tool_msg_content is not None:
        messages.append(
            ToolMessage(content=tool_msg_content, tool_call_id="call_test")
        )
    return {
        "messages": messages,
        "user_profile": extra_state.get("user_profile", {}),
        "user_email": extra_state.get("user_email", "test@fitgen.ai"),
        "context_id": extra_state.get("context_id", "ctx_test"),
        "state_id": extra_state.get("state_id", "ctx_test"),
        "workflow": extra_state.get("workflow", {}),
        "calendar_sync_requested": extra_state.get("calendar_sync_requested", False),
    }


class TestApplyToolStateUpdates:
    @patch("agent.state_sync.upsert_context_state")
    def test_valid_tool_message(self, mock_upsert):
        payload = {
            "assistant_message": "Your plan is ready!",
            "state_updates": {
                "context_id": "ctx_123",
                "user_profile": {"age": 25},
                "workflow": {"intent": "create", "stage": "plan_feedback"},
            },
        }
        state = _make_state(json.dumps(payload))
        updates = apply_tool_state_updates(state)

        assert updates["user_profile"]["age"] == 25
        assert updates["workflow"]["intent"] == "create"

    @patch("agent.state_sync.upsert_context_state")
    def test_malformed_json_doesnt_crash(self, mock_upsert):
        state = _make_state("this is not json {{{")
        updates = apply_tool_state_updates(state)
        assert updates == {} or "context_id" in updates

    @patch("agent.state_sync.upsert_context_state")
    def test_user_profile_merged_not_replaced(self, mock_upsert):
        payload = {
            "assistant_message": "Updated!",
            "state_updates": {
                "user_profile": {"goal": "fat loss"},
            },
        }
        state = _make_state(
            json.dumps(payload),
            user_profile={"age": 25, "name": "Test"},
        )
        updates = apply_tool_state_updates(state)
        # Profile should be merged: existing keys + new keys
        profile = updates.get("user_profile", {})
        assert profile.get("goal") == "fat loss"
        assert profile.get("age") == 25

    @patch("agent.state_sync.upsert_context_state")
    def test_workflow_replaced_not_merged(self, mock_upsert):
        payload = {
            "assistant_message": "OK",
            "state_updates": {
                "workflow": {"intent": "update", "stage": "collect_profile"},
            },
        }
        state = _make_state(
            json.dumps(payload),
            workflow={"intent": "create", "stage": "plan_feedback", "old_key": True},
        )
        updates = apply_tool_state_updates(state)
        wf = updates.get("workflow", {})
        assert wf["intent"] == "update"
        assert "old_key" not in wf  # workflow is replaced, not merged

    @patch("agent.state_sync.upsert_context_state")
    def test_no_tool_message_returns_empty(self, mock_upsert):
        state = _make_state(None)
        updates = apply_tool_state_updates(state)
        # Only context_id might be set from state fallback
        assert "user_profile" not in updates or updates.get("user_profile") == {}
