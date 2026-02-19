"""
agent/state.py
──────────────
Shared state schema for the FITGEN.AI conversational agent.

The state flows through every node in the LangGraph graph and carries
the full conversation history plus optional user-profile metadata.
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State that is threaded through the FITGEN.AI LangGraph graph.

    Attributes
    ----------
    messages : list
        Conversation history managed by LangGraph's built-in message
        reducer (``add_messages``).  Each element is a LangChain
        ``BaseMessage`` (HumanMessage, AIMessage, SystemMessage, …).
    user_profile : dict
        Optional dictionary holding user-supplied fitness context such as
        fitness goals, experience level, dietary preferences, injuries,
        available equipment, etc.  Populated during the intake phase and
        referenced by the chatbot node when generating personalised plans.
    """

    messages: Annotated[list, add_messages]
    user_profile: dict[str, Any]
