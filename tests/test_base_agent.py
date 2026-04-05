"""
tests/test_base_agent.py
────────────────────────
Tests for base_agent.make_base_agent and node behavior.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.base_agent import make_base_agent
from agent.prompts.techniques import TECHNIQUE_KEYS


class TestMakeBaseAgent:
    def test_valid_keys(self):
        """make_base_agent returns a callable for all valid technique keys."""
        for key in TECHNIQUE_KEYS:
            node = make_base_agent(key)
            assert callable(node)

    def test_invalid_key_raises(self):
        """make_base_agent raises ValueError for unknown keys."""
        with pytest.raises((ValueError, KeyError)):
            make_base_agent("nonexistent_technique")

    @patch("agent.base_agent.ChatOpenAI")
    def test_node_invokes_llm(self, MockChatOpenAI):
        """The node function calls llm_with_tools.invoke with messages."""
        mock_response = AIMessage(content="Hello! How can I help?")
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        node = make_base_agent("zero_shot")
        state = {
            "messages": [HumanMessage(content="hi")],
            "user_profile": {},
            "user_email": "",
            "context_id": "ctx",
            "state_id": "ctx",
            "workflow": {},
            "calendar_sync_requested": False,
        }

        result = node(state)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Hello! How can I help?"

    @patch("agent.llm_utils.safe_llm_call")
    @patch("agent.base_agent.ChatOpenAI")
    def test_node_returns_tool_calls(self, MockChatOpenAI, mock_safe_call):
        """When LLM returns tool_calls, node passes them through."""
        mock_response = AIMessage(
            content="",
            tool_calls=[{"name": "workout_tool", "args": {"query": "test"}, "id": "call_1"}],
        )
        mock_safe_call.return_value = mock_response
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        MockChatOpenAI.return_value = mock_llm

        node = make_base_agent("zero_shot")
        state = {
            "messages": [HumanMessage(content="make me a workout")],
            "user_profile": {},
            "user_email": "",
            "context_id": "ctx",
            "state_id": "ctx",
            "workflow": {},
            "calendar_sync_requested": False,
        }

        result = node(state)
        assert result["messages"][0].tool_calls[0]["name"] == "workout_tool"
