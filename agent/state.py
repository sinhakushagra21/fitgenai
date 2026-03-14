"""
agent/state.py
──────────────
Shared state schema for the FITGEN.AI conversational agent.

The state flows through every node in the LangGraph graph and carries
the full conversation history plus optional user-profile metadata.
Routing is handled by the base LLM's tool-calling decisions — no
explicit routing field is needed.
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State threaded through the FITGEN.AI LangGraph graph.

    Attributes
    ----------
    messages : list
        Full conversation history managed by LangGraph's built-in
        ``add_messages`` reducer.  Includes HumanMessages, AIMessages
        (with optional tool_calls), and ToolMessages (tool responses).
    user_profile : dict
        Optional user fitness context: goals, experience, equipment,
        injuries, dietary preferences, etc.  Accumulated during intake
        and passed as context to specialist tools.
    """

    messages: Annotated[list, add_messages]
    user_profile: dict[str, Any]
    user_email: str
    context_id: str
    state_id: str
    workflow: dict[str, Any]
    calendar_sync_requested: bool
