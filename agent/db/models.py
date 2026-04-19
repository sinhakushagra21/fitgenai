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
    name: str = ""  # Short human-readable title (e.g. "Lean Muscle Power Plan")
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


# ── Personal RAG — plan chunks ───────────────────────────────────


class PlanChunkDocument(BaseModel):
    """Schema for the ``plan_chunks`` collection.

    One document per markdown section of a diet or workout plan. Used
    by Personal RAG to answer questions about the user's own plans.
    """

    user_id: Any = None               # ObjectId
    plan_id: Any = None               # ObjectId of source plan
    plan_type: str = "diet"           # diet | workout
    plan_status: str = "draft"        # draft | confirmed | archived
    plan_version: int = 1             # bumps when the source plan is updated
    section_type: str = "notes"       # typed tag (see chunker.py for enum)
    day_of_week: str | None = None    # monday..sunday, or None
    heading: str = ""                 # original markdown heading text
    preamble: str = ""                # breadcrumb, e.g. "Diet plan › 7-Day Meal Plan › Monday"
    chunk_text: str = ""              # original markdown slice
    embedded_text: str = ""           # what actually went to the embedder
    chunk_tokens: int = 0
    embedding: list[float] = Field(default_factory=list)
    profile_snapshot_digest: dict[str, Any] = Field(default_factory=dict)
    source_content_hash: str = ""     # sha256 of chunk_text (idempotency)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


# ── Personal RAG — user memory ───────────────────────────────────


class UserMemoryDocument(BaseModel):
    """Schema for the ``user_memory`` collection.

    One document per durable piece of personal context: a normalized
    profile group, an extracted preference, a feedback note, or a
    constraint. Retrieved alongside plan chunks to personalize answers.
    """

    user_id: Any = None               # ObjectId
    memory_type: str = "profile_group"  # profile_group | preference | feedback | constraint
    domain: str = "base"              # base | diet | workout
    content: str = ""                 # normalized English sentence (what we embed)
    structured: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)
    source: str = "profile"           # profile | chat_turn | plan_feedback
    source_ref: str = ""              # message_id / plan_id etc. for audit
    confidence: float = 1.0
    source_content_hash: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
