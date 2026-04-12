"""
agent/shared/response_builder.py
────────────────────────────────
JSON response builder and step-tracking utilities extracted from
``conversation_workflow.py``.

The ``build_response`` function constructs the canonical JSON payload that
the Streamlit front-end and ``state_sync.py`` both parse.  Its output
contract is:

.. code-block:: json

   {
       "assistant_message": "<str>",
       "state_updates": {
           "context_id": "<str>",
           "state_id": "<str>",
           "user_email": "<str>",
           "workflow": { ... },
           "user_profile": { ... }
       },
       "extra": { ... }           // optional
   }

``append_completed_step`` is a pure helper that merges workflow overrides
and records a step as completed, used by every workflow handler.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.state_manager import StateManager

__all__ = [
    "build_response",
    "append_completed_step",
]

logger: logging.Logger = logging.getLogger("fitgen.response_builder")


# ──────────────────────────────────────────────────────────────────────────────
# Step tracking
# ──────────────────────────────────────────────────────────────────────────────


def append_completed_step(
    current_workflow: dict[str, Any],
    overrides: dict[str, Any],
    step_name: str,
) -> dict[str, Any]:
    """Merge *overrides* into *current_workflow* and append *step_name* to
    the ``completed_steps`` list (if not already present).

    Parameters
    ----------
    current_workflow:
        The workflow dict carried through the current conversation turn.
    overrides:
        Key/value pairs to merge on top of *current_workflow*.
    step_name:
        Logical step identifier (e.g. ``"intake"``, ``"plan_generated"``).

    Returns
    -------
    dict[str, Any]
        A **new** workflow dict with the overrides applied and
        *step_name* appended to ``completed_steps``.
    """
    next_workflow: dict[str, Any] = dict(current_workflow)
    next_workflow.update(overrides)
    completed: list[str] = list(next_workflow.get("completed_steps") or [])
    if step_name not in completed:
        completed.append(step_name)
    next_workflow["completed_steps"] = completed
    return next_workflow


# ──────────────────────────────────────────────────────────────────────────────
# JSON response builder
# ──────────────────────────────────────────────────────────────────────────────


def build_response(
    *,
    assistant_message: str,
    state_id: str,
    user_email: str,
    workflow: dict[str, Any],
    user_profile: dict[str, Any],
    state_manager: StateManager,
    extra: dict[str, Any] | None = None,
) -> str:
    """Build the canonical JSON response payload returned by every tool.

    This function is the **single exit-point** for all conversation-tool
    handlers.  It:

    1. Assembles the ``assistant_message`` + ``state_updates`` envelope.
    2. Optionally attaches an ``extra`` dict (calendar sync flags, etc.).
    3. Persists the updated state via :pymethod:`StateManager.persist` so
       that subsequent turns see the latest snapshot.

    Parameters
    ----------
    assistant_message:
        The text shown to the user in the chat UI.
    state_id:
        Unique context / state identifier for this conversation.
    user_email:
        Current user's email address.
    workflow:
        Full workflow dict (stage, intent, completed_steps, ...).
    user_profile:
        Full user profile dict (name, age, goal, ...).
    state_manager:
        The :class:`StateManager` instance for the active context.
    extra:
        Optional additional data to include in the response payload
        (e.g. ``calendar_sync_requested``).

    Returns
    -------
    str
        A JSON-encoded string matching the response contract consumed by
        the Streamlit app and ``state_sync.py``.
    """
    payload: dict[str, Any] = {
        "assistant_message": assistant_message,
        "state_updates": {
            "context_id": state_id,
            "state_id": state_id,
            "user_email": user_email,
            "workflow": workflow,
            "user_profile": user_profile,
        },
    }
    if extra:
        payload["extra"] = extra

    state_manager.persist(
        user_profile=user_profile,
        workflow=workflow,
        user_email=user_email,
        calendar_sync_requested=(
            bool(extra.get("calendar_sync_requested"))
            if extra and "calendar_sync_requested" in extra
            else state_manager.calendar_sync_requested
        ),
    )

    logger.debug(
        "build_response  context=%s  step_completed=%s  steps=%s",
        state_id,
        workflow.get("step_completed"),
        workflow.get("completed_steps"),
    )

    return json.dumps(payload, ensure_ascii=False)
