"""
agent/tools/workout_tool.py
────────────────────────────
FITGEN.AI Workout Specialist Tool.

Implements a multi-turn intent-aware workflow for create / modify / delete
operations with profile intake, SQL persistence, and calendar sync prompt.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from agent.prompts.workout_prompts import WORKOUT_PROMPTS
from agent.tools.conversation_workflow import execute
from agent.tracing import trace

logger = logging.getLogger("fitgen.workout_tool")


def _get_raw_user_query(state: dict[str, Any]) -> str:
    """Extract the last HumanMessage content from graph state messages."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


@tool
@trace(name="Workout Tool", run_type="tool", tags=["workout", "tool"])
def workout_tool(query: str, state: Annotated[dict[str, Any], InjectedState]) -> str:
    """Multi-turn workout agent with create/modify/delete workflow and SQL persistence.

    Workflow:
    1) Understand user intent (LLM): create / modify / delete
    2) Collect or update profile data (height, weight, etc.)
    3) Persist records by state_id in SQLite
    4) Ask optional Google Calendar sync

    IMPORTANT: The 'query' parameter must be the user's EXACT message, word-for-word.
    Do NOT paraphrase, expand, or add instructions. Just pass the raw user text.

    Returns JSON with assistant_message and state_updates.
    """
    # Always use the raw user message from state, not the LLM's tool-call args.
    # The base agent LLM may elaborate or rewrite the query in its tool call,
    # which pollutes profile extraction and yes/no classification downstream.
    raw_query = _get_raw_user_query(state)
    effective_query = raw_query or query  # fallback to LLM arg only if state has no messages

    logger.info("[WorkoutTool] Multi-turn query: %s", effective_query[:120])
    if raw_query != query:
        logger.debug(
            "[WorkoutTool] Overrode LLM tool arg (len=%d) with raw user message (len=%d)",
            len(query),
            len(raw_query),
        )

    return execute(
        domain="workout",
        query=effective_query,
        state=state,
        plan_system_prompt=WORKOUT_PROMPTS["few_shot"],
    )
