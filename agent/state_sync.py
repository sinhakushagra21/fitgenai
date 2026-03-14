"""
agent/state_sync.py
───────────────────
Applies state updates emitted by tool JSON responses.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import ToolMessage

from agent.state import AgentState
from agent.persistence import upsert_context_state


MERGE_KEYS = {"user_profile", "calendar_sync_requested"}
logger = logging.getLogger("fitgen.state_sync")


def apply_tool_state_updates(state: AgentState) -> dict[str, Any]:
    """Read latest ToolMessage JSON and merge `state_updates` into graph state."""
    updates: dict[str, Any] = {}

    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            continue
        if not msg.content:
            continue

        try:
            payload = json.loads(msg.content)
        except json.JSONDecodeError:
            continue

        state_updates = payload.get("state_updates")
        if not isinstance(state_updates, dict):
            continue

        for key, value in state_updates.items():
            if key == "workflow":
                updates[key] = value if isinstance(value, dict) else {}
            elif key in MERGE_KEYS and isinstance(value, dict):
                current = state.get(key, {})
                if isinstance(current, dict):
                    updates[key] = {**current, **value}
                else:
                    updates[key] = value
            else:
                updates[key] = value

        extra = payload.get("extra")
        if isinstance(extra, dict):
            for key, value in extra.items():
                if key in MERGE_KEYS and isinstance(value, dict):
                    current = updates.get(key, state.get(key, {}))
                    if isinstance(current, dict):
                        updates[key] = {**current, **value}
                    else:
                        updates[key] = value
                else:
                    updates[key] = value

        break

    context_id = (
        updates.get("context_id")
        or updates.get("state_id")
        or state.get("context_id")
        or state.get("state_id")
    )

    if context_id:
        try:
            effective_profile = updates.get("user_profile", state.get("user_profile") or {})
            effective_workflow = updates.get("workflow", state.get("workflow") or {})
            effective_calendar = updates.get(
                "calendar_sync_requested",
                state.get("calendar_sync_requested", False),
            )
            effective_email = str(
                updates.get("user_email")
                or state.get("user_email")
                or ""
            )
            upsert_context_state(
                context_id=str(context_id),
                user_email=effective_email,
                user_profile=dict(effective_profile or {}),
                workflow=dict(effective_workflow or {}),
                calendar_sync_requested=bool(effective_calendar),
            )
            updates.setdefault("context_id", str(context_id))
            updates.setdefault("state_id", str(context_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist context state for context_id=%s: %s", context_id, exc)

    return updates
