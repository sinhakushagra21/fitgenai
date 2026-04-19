"""
agent/rag/personal/indexer.py
─────────────────────────────
Personal-RAG indexing orchestration.

Two public entry points:

  * ``index_plan(plan_id, *, plan_type)`` — called after a plan is
    saved/updated/confirmed/archived. Loads the plan from the relevant
    repository, chunks it, embeds each chunk, archives any older
    versions for the same ``plan_id``, and bulk-upserts the chunks into
    ``plan_chunks``.

  * ``index_profile(user_email)`` — called after a user's profile is
    updated. Normalises each profile group (base / diet / workout) into
    a short natural-language sentence, embeds it, and upserts into
    ``user_memory`` under ``memory_type='profile_group'``.

Both functions swallow their own errors and return 0 on failure so that
plan-saving / profile-saving never breaks if the RAG layer hiccups.
The hooks in the repos wrap these in a second try/except for belt-and-
braces safety.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from agent.db.repositories.diet_plan_repo import DietPlanRepository
from agent.db.repositories.plan_chunks_repo import PlanChunksRepository
from agent.db.repositories.user_memory_repo import UserMemoryRepository
from agent.db.repositories.user_repo import UserRepository
from agent.db.repositories.workout_plan_repo import WorkoutPlanRepository
from agent.error_utils import handle_exception
from agent.rag.personal.chunker import chunk_diet_plan, chunk_workout_plan
from agent.rag.personal.embedder import embed_texts
from agent.tracing import log_event

logger = logging.getLogger("fitgen.rag.personal.indexer")


# ─────────────────────────────────────────────────────────────────────
# Plans
# ─────────────────────────────────────────────────────────────────────

def _plan_repo(plan_type: str):
    return DietPlanRepository if plan_type == "diet" else WorkoutPlanRepository


def index_plan(plan_id: str, *, plan_type: str) -> int:
    """Chunk + embed + upsert one plan's markdown into ``plan_chunks``.

    Args:
        plan_id: Mongo ObjectId (str or ObjectId) of the plan.
        plan_type: "diet" or "workout".

    Returns:
        Number of chunks written. Returns 0 on any failure.
    """
    try:
        repo = _plan_repo(plan_type)
        plan = repo.find_by_id(plan_id)
        if not plan:
            logger.warning(
                "[index_plan] plan %s (%s) not found — skipping",
                plan_id, plan_type,
            )
            return 0

        plan_markdown: str = plan.get("plan_markdown", "") or ""
        if not plan_markdown.strip():
            logger.info(
                "[index_plan] plan %s has empty markdown — skipping",
                plan_id,
            )
            return 0

        plan_status: str = plan.get("status", "draft")
        plan_version: int = int(plan.get("version", 1) or 1)
        user_id = str(plan["user_id"])
        profile_snapshot: dict[str, Any] = dict(
            plan.get("profile_snapshot") or {}
        )

        chunk_fn = chunk_diet_plan if plan_type == "diet" else chunk_workout_plan
        chunks = chunk_fn(
            plan_markdown,
            user_id=user_id,
            plan_id=str(plan_id),
            plan_version=plan_version,
            plan_status=plan_status,
            profile_snapshot=profile_snapshot,
        )
        if not chunks:
            logger.info(
                "[index_plan] plan %s produced 0 chunks — skipping",
                plan_id,
            )
            return 0

        # Embed the ``embedded_text`` field of every chunk in one
        # batched API round-trip.
        vectors = embed_texts([c.embedded_text for c in chunks])
        for c, vec in zip(chunks, vectors):
            c.embedding = vec

        docs = [c.to_mongo() for c in chunks]

        # Archive older versions for this plan_id before upserting.
        try:
            PlanChunksRepository.archive_older_versions(
                plan_id, current_version=plan_version,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[index_plan] archive_older_versions failed: %s", exc,
            )

        written = PlanChunksRepository.upsert_many(docs)

        log_event(
            "rag.index_plan.ok",
            module="rag.indexer",
            plan_type=plan_type,
            plan_id=str(plan_id),
            plan_version=plan_version,
            chunks=len(docs),
            written=written,
        )
        return written

    except Exception as exc:  # noqa: BLE001
        handle_exception(
            exc,
            module="rag.indexer",
            context="index_plan",
            level="WARNING",
            extra={"plan_id": str(plan_id), "plan_type": plan_type},
        )
        return 0


# ─────────────────────────────────────────────────────────────────────
# Profile
# ─────────────────────────────────────────────────────────────────────

_BASE_KEYS = (
    "name", "age", "sex", "height_cm", "weight_kg", "goal",
    "sleep_hours", "stress_level", "job_type",
)
_DIET_KEYS = (
    "diet_preference", "allergies", "meals_per_day", "cooking_time",
    "favourite_meals", "foods_to_avoid", "cooking_style",
    "food_adventurousness", "current_snacks", "snack_reason",
    "snack_preference", "late_night_snacking", "goal_weight",
    "weight_loss_pace", "exercise_frequency", "exercise_type",
    "alcohol_intake",
)
_WORKOUT_KEYS = (
    "experience_level", "training_days_per_week", "session_duration",
    "daily_steps", "additional_info",
)


def _flatten_profile_group(
    group: dict[str, Any], *, keys: tuple[str, ...],
) -> dict[str, Any]:
    return {
        k: group[k]
        for k in keys
        if k in group and group[k] not in (None, "")
    }


def _render_sentence(domain: str, fields: dict[str, Any]) -> str:
    """Flat key=value sentence — good enough signal for the embedder.

    We deliberately keep it deterministic (no LLM) so indexing is
    cheap and reproducible. The embedder is strong enough to match
    natural-language queries against these.
    """
    if not fields:
        return ""
    parts = [f"{k.replace('_', ' ')}: {v}" for k, v in fields.items()]
    prefix = {
        "base":    "User profile (base)",
        "diet":    "User profile (diet preferences)",
        "workout": "User profile (workout preferences)",
    }[domain]
    return f"{prefix}. " + "; ".join(parts) + "."


def _hash_fields(domain: str, fields: dict[str, Any]) -> str:
    items = sorted((k, str(v)) for k, v in fields.items())
    h = hashlib.sha256()
    h.update(domain.encode("utf-8"))
    for k, v in items:
        h.update(b"\x00")
        h.update(k.encode("utf-8"))
        h.update(b"=")
        h.update(v.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def index_profile(user_email: str) -> int:
    """Embed + upsert each profile group (base/diet/workout) for a user.

    Idempotent: if the flattened ``(domain, fields)`` hash hasn't
    changed since the last index, we still upsert (cheap) — Atlas
    merges the same vector back in place.

    Returns:
        Number of groups written. 0 on failure.
    """
    try:
        user = UserRepository.find_by_email(user_email)
        if not user:
            logger.warning(
                "[index_profile] user %s not found — skipping", user_email,
            )
            return 0

        user_id = str(user["_id"])
        groups = {
            "base":    _flatten_profile_group(
                user.get("base_profile") or {}, keys=_BASE_KEYS,
            ),
            "diet":    _flatten_profile_group(
                user.get("diet_profile") or {}, keys=_DIET_KEYS,
            ),
            "workout": _flatten_profile_group(
                user.get("workout_profile") or {}, keys=_WORKOUT_KEYS,
            ),
        }

        # Pull the user's top-level name into base if missing.
        if user.get("name") and "name" not in groups["base"]:
            groups["base"]["name"] = user["name"]

        non_empty = {d: f for d, f in groups.items() if f}
        if not non_empty:
            logger.info(
                "[index_profile] user %s has no profile fields — skipping",
                user_email,
            )
            return 0

        sentences = [_render_sentence(d, f) for d, f in non_empty.items()]
        vectors = embed_texts(sentences)

        written = 0
        for (domain, fields), sentence, vec in zip(
            non_empty.items(), sentences, vectors,
        ):
            try:
                UserMemoryRepository.upsert_profile_group(
                    user_id=user_id,
                    domain=domain,
                    content=sentence,
                    structured=fields,
                    embedding=vec,
                    source_content_hash=_hash_fields(domain, fields),
                )
                written += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[index_profile] upsert failed domain=%s: %s",
                    domain, exc,
                )

        log_event(
            "rag.index_profile.ok",
            module="rag.indexer",
            user_email=user_email,
            groups=list(non_empty.keys()),
            written=written,
        )
        return written

    except Exception as exc:  # noqa: BLE001
        handle_exception(
            exc,
            module="rag.indexer",
            context="index_profile",
            level="WARNING",
            extra={"user_email": user_email},
        )
        return 0
