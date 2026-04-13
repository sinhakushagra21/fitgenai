"""
agent/state_manager.py
──────────────────────
Context-aware state manager for multi-turn workflow execution.

This module centralizes:
1) fetching state by context_id,
2) merging persisted + in-memory state,
3) updating state after each workflow step,
4) persisting updated snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.persistence import get_context_state, upsert_context_state


@dataclass
class StateManager:
    """State manager for context-scoped workflow state."""

    context_id: str
    user_email: str
    user_id: str
    user_profile: dict[str, Any]
    workflow: dict[str, Any]
    calendar_sync_requested: bool

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "StateManager":
        """Create manager by merging persisted context snapshot with runtime state."""
        context_id = str(state.get("context_id") or state.get("state_id") or "default")
        persisted = get_context_state(context_id) or {}

        user_profile = dict(persisted.get("user_profile") or {})
        user_profile.update(dict(state.get("user_profile") or {}))

        workflow = dict(persisted.get("workflow") or {})
        workflow.update(dict(state.get("workflow") or {}))

        user_email = str(state.get("user_email") or persisted.get("user_email") or "")
        user_id = str(state.get("user_id") or persisted.get("user_id") or "")
        calendar_sync_requested = bool(
            state.get("calendar_sync_requested", persisted.get("calendar_sync_requested", False))
        )

        return cls(
            context_id=context_id,
            user_email=user_email,
            user_id=user_id,
            user_profile=user_profile,
            workflow=workflow,
            calendar_sync_requested=calendar_sync_requested,
        )

    def persist(
        self,
        *,
        user_profile: dict[str, Any],
        workflow: dict[str, Any],
        user_email: str | None = None,
        calendar_sync_requested: bool | None = None,
    ) -> None:
        """Persist state snapshot after each workflow step."""
        self.user_profile = dict(user_profile or {})
        self.workflow = dict(workflow or {})
        if user_email is not None:
            self.user_email = user_email
        if calendar_sync_requested is not None:
            self.calendar_sync_requested = bool(calendar_sync_requested)

        upsert_context_state(
            context_id=self.context_id,
            user_email=self.user_email,
            user_profile=self.user_profile,
            workflow=self.workflow,
            calendar_sync_requested=self.calendar_sync_requested,
        )
