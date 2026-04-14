"""
agent/db/models.py
──────────────────
Pydantic models for MongoDB document validation.

These models define the canonical shape of each document before it is
written to Mongo.  They are NOT ODM models — pymongo handles the actual
read/write.  We use these for:

  1. Validation before insert/update (fail fast, clear errors).
  2. A single place to see the full schema.
  3. Easy serialization with ``model_dump()``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


# ── Users ────────────────────────────────────────────────────────


class UserDocument(BaseModel):
    """Schema for the ``users`` collection."""

    email: str
    name: str = ""
    base_profile: dict[str, Any] = Field(default_factory=dict)
    diet_profile: dict[str, Any] = Field(default_factory=dict)
    workout_profile: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


# ── Plans ────────────────────────────────────────────────────────


class PlanDocument(BaseModel):
    """Schema for ``diet_plans`` and ``workout_plans`` collections."""

    user_id: Any = None  # ObjectId — set by repo
    session_id: str = ""
    profile_snapshot: dict[str, Any] = Field(default_factory=dict)
    plan_markdown: str = ""
    structured_data: dict[str, Any] = Field(default_factory=dict)
    status: str = "draft"  # draft | confirmed | archived
    calendar_synced: bool = False
    fit_synced: bool = False
    version: int = 1
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


# ── Sessions ─────────────────────────────────────────────────────


class SessionDocument(BaseModel):
    """Schema for the ``sessions`` collection (replaces context_states)."""

    session_id: str
    user_id: Any = None  # ObjectId — set by repo when user is identified
    user_email: str = ""
    user_profile: dict[str, Any] = Field(default_factory=dict)
    workflow: dict[str, Any] = Field(default_factory=dict)
    calendar_sync_requested: bool = False
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    expires_at: datetime = Field(default_factory=_utcnow)  # TTL index


# ── Feedback ─────────────────────────────────────────────────────


class FeedbackDocument(BaseModel):
    """Schema for the ``feedback`` collection."""

    session_id: str
    user_id: Any = None
    turn_id: str = ""
    rating: int | None = None
    comment: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
