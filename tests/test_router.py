"""
tests/test_router.py
────────────────────
Tests for the deterministic router (agent/router.py).
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.router import (
    _wants_domain_switch,
    _classify_intent,
    _emit_tool_call,
    router_node,
)


# =====================================================================
# _wants_domain_switch
# =====================================================================

class TestWantsDomainSwitch:
    """Test the keyword heuristic for domain switch detection."""

    # ── Should detect switch ─────────────────────────────────────

    def test_explicit_create_workout_plan(self):
        assert _wants_domain_switch(
            "create a workout plan for me", current_domain="diet"
        )

    def test_no_comma_create_workout(self):
        assert _wants_domain_switch(
            "no, create a workout plan", current_domain="diet"
        )

    def test_make_diet_plan_instead(self):
        assert _wants_domain_switch(
            "make me a diet plan instead", current_domain="workout"
        )

    def test_switch_to_workout(self):
        assert _wants_domain_switch(
            "switch to workout plan", current_domain="diet"
        )

    def test_id_rather_have_diet(self):
        assert _wants_domain_switch(
            "I'd rather have a diet plan", current_domain="workout"
        )

    def test_no_give_me_exercise_routine(self):
        assert _wants_domain_switch(
            "no give me an exercise routine", current_domain="diet"
        )

    def test_get_meal_plan(self):
        assert _wants_domain_switch(
            "get me a meal plan please", current_domain="workout"
        )

    def test_new_nutrition_plan(self):
        assert _wants_domain_switch(
            "I want a new nutrition plan", current_domain="workout"
        )

    # ── Should NOT detect switch ─────────────────────────────────

    def test_profile_data_exercise_frequency(self):
        """exercise_frequency is a profile field, not a domain switch."""
        assert not _wants_domain_switch(
            "exercise_frequency: 5 times a week", current_domain="diet"
        )

    def test_profile_data_diet_preference(self):
        """diet_preference is a profile field, not a domain switch."""
        assert not _wants_domain_switch(
            "diet_preference: omnivore", current_domain="workout"
        )

    def test_yes_confirmation(self):
        assert not _wants_domain_switch("yes", current_domain="diet")

    def test_no_standalone(self):
        """Standalone 'no' without other domain keywords isn't a switch."""
        assert not _wants_domain_switch("no", current_domain="diet")

    def test_profile_with_exercise_mention(self):
        """Mentioning exercise as part of profile data."""
        assert not _wants_domain_switch(
            "I exercise 5 times a week, mostly running",
            current_domain="diet",
        )

    def test_simple_number(self):
        assert not _wants_domain_switch("28", current_domain="diet")

    def test_profile_bulk(self):
        """Full profile data submission."""
        assert not _wants_domain_switch(
            "name: Kushagra, age: 28, weight: 80kg, exercise_type: cardio",
            current_domain="diet",
        )

    def test_i_want_to_exercise_more(self):
        """'I want to exercise more' during diet flow — not a switch."""
        assert not _wants_domain_switch(
            "I want to exercise more", current_domain="diet"
        )


# =====================================================================
# _emit_tool_call
# =====================================================================

class TestEmitToolCall:
    def test_creates_ai_message_with_tool_call(self):
        result = _emit_tool_call("diet_tool", "create a diet plan")
        msgs = result["messages"]
        assert len(msgs) == 1
        assert isinstance(msgs[0], AIMessage)
        assert len(msgs[0].tool_calls) == 1
        assert msgs[0].tool_calls[0]["name"] == "diet_tool"
        assert msgs[0].tool_calls[0]["args"] == {"query": "create a diet plan"}

    def test_unique_call_ids(self):
        r1 = _emit_tool_call("diet_tool", "q1")
        r2 = _emit_tool_call("diet_tool", "q2")
        id1 = r1["messages"][0].tool_calls[0]["id"]
        id2 = r2["messages"][0].tool_calls[0]["id"]
        assert id1 != id2


# =====================================================================
# router_node
# =====================================================================

class TestRouterNode:
    """Integration tests for the router_node function."""

    def _state(self, messages=None, workflow=None):
        return {
            "messages": messages or [],
            "user_profile": {},
            "user_email": "test@fitgen.ai",
            "user_id": "",
            "context_id": "ctx",
            "state_id": "ctx",
            "workflow": workflow or {},
            "calendar_sync_requested": False,
        }

    def test_tool_message_ack(self):
        """ToolMessage → short acknowledgement, no LLM call."""
        state = self._state(
            messages=[
                HumanMessage(content="create a plan"),
                ToolMessage(content='{"assistant_message":"plan done"}', tool_call_id="c1"),
            ]
        )
        result = router_node(state)
        assert len(result["messages"]) == 1
        assert "Done" in result["messages"][0].content
        assert not getattr(result["messages"][0], "tool_calls", None)

    def test_active_diet_workflow_routes_to_diet_tool(self):
        """Active diet workflow → deterministic route to diet_tool."""
        state = self._state(
            messages=[HumanMessage(content="yes confirm")],
            workflow={"domain": "diet", "step_completed": "diet_plan_generated"},
        )
        result = router_node(state)
        msg = result["messages"][0]
        assert msg.tool_calls[0]["name"] == "diet_tool"
        assert msg.tool_calls[0]["args"]["query"] == "yes confirm"

    def test_active_workout_workflow_routes_to_workout_tool(self):
        """Active workout workflow → deterministic route to workout_tool."""
        state = self._state(
            messages=[HumanMessage(content="name: Test, age: 25")],
            workflow={"domain": "workout", "step_completed": "prompted_for_user_profile_data"},
        )
        result = router_node(state)
        assert result["messages"][0].tool_calls[0]["name"] == "workout_tool"

    def test_domain_switch_during_diet_workflow(self):
        """User says 'no create a workout plan' during diet → workout_tool."""
        state = self._state(
            messages=[HumanMessage(content="no, create a workout plan for me")],
            workflow={"domain": "diet", "step_completed": "prompted_for_user_profile_data"},
        )
        result = router_node(state)
        assert result["messages"][0].tool_calls[0]["name"] == "workout_tool"

    def test_domain_switch_during_workout_workflow(self):
        """User says 'make me a diet plan instead' during workout → diet_tool."""
        state = self._state(
            messages=[HumanMessage(content="make me a diet plan instead")],
            workflow={"domain": "workout", "step_completed": "user_profile_mapped"},
        )
        result = router_node(state)
        assert result["messages"][0].tool_calls[0]["name"] == "diet_tool"

    def test_terminal_step_no_active_workflow(self):
        """Terminal step → no active workflow, uses classifier."""
        state = self._state(
            messages=[HumanMessage(content="create a diet plan")],
            workflow={"domain": "diet", "step_completed": "diet_confirmed"},
        )
        with patch("agent.router._classify_intent", return_value="diet_tool"):
            result = router_node(state)
        assert result["messages"][0].tool_calls[0]["name"] == "diet_tool"

    @patch("agent.router._classify_intent", return_value="workout_tool")
    def test_no_workflow_classifies_workout(self, mock_classify):
        """No workflow → classifier says workout_tool."""
        state = self._state(
            messages=[HumanMessage(content="make me a workout plan")],
        )
        result = router_node(state)
        assert result["messages"][0].tool_calls[0]["name"] == "workout_tool"

    @patch("agent.router._classify_intent", return_value="diet_tool")
    def test_no_workflow_classifies_diet(self, mock_classify):
        state = self._state(
            messages=[HumanMessage(content="create a meal plan")],
        )
        result = router_node(state)
        assert result["messages"][0].tool_calls[0]["name"] == "diet_tool"

    @patch("agent.router._classify_intent", return_value="rag_query_tool")
    def test_no_workflow_classifies_rag(self, mock_classify):
        state = self._state(
            messages=[HumanMessage(content="is creatine safe?")],
        )
        result = router_node(state)
        assert result["messages"][0].tool_calls[0]["name"] == "rag_query_tool"

    @patch("agent.router._generate_direct_response")
    @patch("agent.router._classify_intent", return_value="direct")
    def test_no_workflow_classifies_direct(self, mock_classify, mock_direct):
        """Direct intent → calls _generate_direct_response."""
        mock_direct.return_value = {
            "messages": [AIMessage(content="Hi! I'm FITGEN.AI")]
        }
        state = self._state(
            messages=[HumanMessage(content="hello")],
        )
        result = router_node(state)
        assert "FITGEN" in result["messages"][0].content
        mock_direct.assert_called_once()

    def test_profile_data_stays_in_active_workflow(self):
        """Profile data like 'exercise_frequency: 5' stays in diet workflow."""
        state = self._state(
            messages=[HumanMessage(
                content="name: Kushagra, age: 28, exercise_frequency: 5, "
                        "diet_preference: omnivore"
            )],
            workflow={"domain": "diet", "step_completed": "prompted_for_user_profile_data"},
        )
        result = router_node(state)
        # Should route to diet_tool, NOT workout_tool
        assert result["messages"][0].tool_calls[0]["name"] == "diet_tool"


# =====================================================================
# _classify_intent (with mocked LLM)
# =====================================================================

class TestClassifyIntent:

    @patch("agent.router.safe_llm_call")
    @patch("agent.router.ChatOpenAI")
    def test_returns_valid_route(self, mock_openai, mock_safe_call):
        mock_safe_call.return_value = MagicMock(content="diet_tool")
        result = _classify_intent("create a meal plan")
        assert result == "diet_tool"

    @patch("agent.router.safe_llm_call")
    @patch("agent.router.ChatOpenAI")
    def test_invalid_response_defaults_to_direct(self, mock_openai, mock_safe_call):
        mock_safe_call.return_value = MagicMock(content="unknown_thing")
        result = _classify_intent("create a meal plan")
        assert result == "direct"

    @patch("agent.router.safe_llm_call", side_effect=Exception("API error"))
    @patch("agent.router.ChatOpenAI")
    def test_error_defaults_to_direct(self, mock_openai, mock_safe_call):
        result = _classify_intent("create a meal plan")
        assert result == "direct"
