"""
agent/persistence.py
────────────────────
Persistence facade for FITGEN.AI.

This module provides the **same public API** as the old SQLite-based
persistence layer, but delegates all storage to MongoDB via the
repository classes in ``agent.db.repositories``.

Consumers (state_manager.py, state_sync.py, streamlit_app.py, app.py)
import from here and don't need to know about the underlying DB engine.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.db.mongo import init_indexes
from agent.db.repositories.session_repo import SessionRepository
from agent.db.repositories.user_repo import UserRepository

logger = logging.getLogger("fitgen.persistence")


# ── Initialization ───────────────────────────────────────────────


def init_db() -> None:
    """Initialize the database — creates MongoDB indexes.

    Safe to call on every startup (idempotent).
    """
    init_indexes()
    logger.info("Database initialized (MongoDB indexes ensured).")


# ── Context / Session state ──────────────────────────────────────
# These functions maintain backward compatibility with the old SQLite
# API so that state_manager.py and state_sync.py work unchanged.


def get_context_state(context_id: str) -> dict[str, Any] | None:
    """Retrieve session state by context_id (session_id).

    Returns a dict matching the old contract:
    ``{context_id, state_id, user_email, user_profile, workflow,
       calendar_sync_requested, updated_at}``
    """
    doc = SessionRepository.find_by_session_id(context_id)
    if not doc:
        return None
    return _session_doc_to_context(doc)


def get_latest_context_state_by_email(user_email: str) -> dict[str, Any] | None:
    """Retrieve the most recent session for an email address."""
    if not user_email:
        return None
    doc = SessionRepository.find_latest_by_email(user_email)
    if not doc:
        return None
    return _session_doc_to_context(doc)


def upsert_context_state(
    *,
    context_id: str,
    user_email: str,
    user_profile: dict[str, Any],
    workflow: dict[str, Any],
    calendar_sync_requested: bool,
) -> None:
    """Create or update a session document."""
    SessionRepository.upsert(
        session_id=context_id,
        user_email=user_email,
        user_profile=user_profile,
        workflow=workflow,
        calendar_sync_requested=calendar_sync_requested,
    )


def delete_context_state(context_id: str) -> bool:
    """Delete a session by context_id.  Returns True if removed."""
    return SessionRepository.delete(context_id)


# ── User records (legacy facade) ────────────────────────────────
# The old user_records table stored per-session plan snapshots.
# We keep the function signatures for any callers but route them
# to the new collections.  These are thin wrappers — the real work
# happens in the plan repos + user repo.


def get_record(state_id: str) -> dict[str, Any] | None:
    """Legacy: fetch a user record by state_id.

    Now looks up the session to find the associated plan.
    """
    session = SessionRepository.find_by_session_id(state_id)
    if not session:
        return None

    workflow = session.get("workflow") or {}
    return {
        "state_id": state_id,
        "domain": workflow.get("domain", ""),
        "profile": session.get("user_profile") or {},
        "plan_text": workflow.get("plan_text", ""),
        "calendar_sync": session.get("calendar_sync_requested", False),
        "updated_at": session.get("updated_at"),
    }


def upsert_record(
    *,
    state_id: str,
    domain: str,
    profile: dict[str, Any],
    plan_text: str,
    calendar_sync: bool = False,
) -> None:
    """Legacy: upsert a user record.

    Routes to a session upsert with the plan text stored in workflow.
    """
    SessionRepository.upsert(
        session_id=state_id,
        user_profile=profile,
        workflow={"domain": domain, "plan_text": plan_text},
        calendar_sync_requested=calendar_sync,
    )


def update_calendar_sync(state_id: str, enabled: bool) -> None:
    """Legacy: toggle calendar sync flag on a record."""
    SessionRepository.upsert(
        session_id=state_id,
        calendar_sync_requested=enabled,
    )


def delete_record(state_id: str) -> bool:
    """Legacy: delete a user record by state_id."""
    return SessionRepository.delete(state_id)


# ── Internal helpers ─────────────────────────────────────────────


def _session_doc_to_context(doc: dict[str, Any]) -> dict[str, Any]:
    """Convert a MongoDB session document to the old context_state dict shape."""
    session_id = doc.get("session_id", "")
    return {
        "context_id": session_id,
        "state_id": session_id,
        "user_email": doc.get("user_email", ""),
        "user_profile": doc.get("user_profile") or {},
        "workflow": doc.get("workflow") or {},
        "calendar_sync_requested": doc.get("calendar_sync_requested", False),
        "updated_at": doc.get("updated_at"),
    }
