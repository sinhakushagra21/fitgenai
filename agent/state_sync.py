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
from agent.error_utils import handle_exception


MERGE_KEYS = {"user_profile", "calendar_sync_requested"}
logger = logging.getLogger("fitgen.state_sync")

# Steps after which the workflow is "done" and should be clean for the
# next operation.  The tool's confirm/get handler may have already
# wiped ctx, but if the base agent handled the response instead, the
# workflow was never cleaned.  This acts as a safety net.
_COMPLETED_STEPS = frozenset({
    "diet_confirmed",
    "workout_confirmed",
    "diet_plan_synced_to_google_calendar",
    "diet_plan_synced_to_google_fit",
    "workout_plan_synced_to_google_calendar",
    "workout_plan_synced_to_google_fit",
})


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

            # ── Safety-net cleanup: don't persist stale workflows ──
            # If the workflow reached a completed step, strip heavy
            # payload (plan_text, structured_data, missing_fields,
            # completed_steps) before persisting.  Keep only plan_id
            # + domain so sync can still locate the plan in MongoDB.
            _ew = dict(effective_workflow or {})
            _step = _ew.get("step_completed")
            if _step in _COMPLETED_STEPS:
                _plan_id = _ew.get("plan_id")
                _domain = _ew.get("domain")
                _ew = {
                    "step_completed": _step,
                    "domain": _domain,
                }
                if _plan_id:
                    _ew["plan_id"] = _plan_id
                effective_workflow = _ew
                updates["workflow"] = _ew
                logger.info(
                    "[state_sync] Cleaned completed workflow before persist "
                    "(step=%s, plan_id=%s)", _step, _plan_id,
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
            handle_exception(
                exc,
                module="state_sync",
                context="persist context state",
                level="WARNING",
                extra={
                    "context_id": str(context_id),
                    "user_email": effective_email,
                    "workflow_step": (effective_workflow or {}).get("step_completed"),
                },
            )

    return updates
