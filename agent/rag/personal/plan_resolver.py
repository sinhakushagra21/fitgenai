"""
agent/rag/personal/plan_resolver.py
───────────────────────────────────
Pick which plan a user's "get" query is referring to.

FITGEN keeps **one active plan per domain per user** (see
``_archive_siblings`` in the plan repos). But archived plans are kept in
MongoDB and remain searchable by their distinguishing descriptors
(``name`` + ``profile_snapshot``: diet_preference, goal, …).

``resolve_plan(user_id, query, plan_type)`` tries, in order:

1. If the user has no archived plans, return the active plan.
2. If the query contains no descriptor word that separates the plans
   (vegan / keto / cut / bulk / strength / hypertrophy / …), return the
   active plan.
3. Otherwise score each plan's descriptors against the query with a
   fast LLM and return the best match — archived plans included.

Return shape: ``{"plan": dict, "is_archived": bool, "match_reason": str}``
or ``None`` if the user has no plans at all. Fails soft — on any error
falls back to the active plan.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.config import FAST_MODEL
from agent.db.repositories.diet_plan_repo import DietPlanRepository
from agent.db.repositories.workout_plan_repo import WorkoutPlanRepository

logger = logging.getLogger("fitgen.rag.personal.plan_resolver")

# Words that typically distinguish one plan from another. If none of
# these appear in the user's query we skip the LLM match entirely and
# return the active plan.
_DESCRIPTOR_WORDS = {
    # diet
    "vegan", "vegetarian", "veg", "non-veg", "nonveg", "non veg",
    "eggetarian", "keto", "paleo", "mediterranean", "indian", "jain",
    "pescatarian", "dairy-free", "gluten-free",
    # goals (shared)
    "cut", "bulk", "cutting", "bulking", "fat loss", "weight loss",
    "muscle gain", "maintenance", "performance", "recomp",
    # workout
    "strength", "hypertrophy", "powerlifting", "ppl", "push pull legs",
    "upper lower", "full body", "calisthenics",
    # generic qualifiers
    "old", "previous", "last", "earlier", "before", "archived",
}


def _repo_for(plan_type: str):
    return (
        DietPlanRepository if plan_type == "diet" else WorkoutPlanRepository
    )


def _has_descriptor(query: str) -> bool:
    q = query.lower()
    return any(w in q for w in _DESCRIPTOR_WORDS)


def _summarize_plan(plan: dict[str, Any]) -> str:
    """Short one-line descriptor for LLM matching."""
    name = plan.get("name") or "(unnamed)"
    status = plan.get("status", "?")
    snap = plan.get("profile_snapshot") or {}
    bits: list[str] = []
    for k in ("diet_preference", "goal", "goal_weight", "experience_level",
              "training_days_per_week"):
        v = snap.get(k)
        if v:
            bits.append(f"{k}={v}")
    return f"name={name!r} status={status} " + " ".join(bits)


_SYSTEM_PROMPT = """You are a disambiguation helper for a fitness app.
Given a user's question and a numbered list of their plans, pick which
plan the user is asking about. Match on descriptors: plan name, diet
preference (vegan, keto, …), goal (fat loss, bulk, …), workout style.

Return ONLY JSON of the form:
  {"choice": <1-based index>, "reason": "<short phrase>"}

If the query does not clearly name a specific plan, choose the plan
whose status is "confirmed" or "draft" (the active one)."""


def _llm_match(query: str, plans: list[dict[str, Any]]) -> tuple[int, str]:
    """Ask the fast LLM which plan matches. Returns (1-based index, reason).

    Falls back to the active plan (first non-archived) on any error.
    """
    numbered = "\n".join(
        f"{i+1}. {_summarize_plan(p)}" for i, p in enumerate(plans)
    )
    try:
        llm = ChatOpenAI(model=FAST_MODEL, temperature=0)
        resp = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {query}\n\nPlans:\n{numbered}"),
        ])
        raw = (resp.content or "").strip()
        # Strip code fences if present.
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.split("\n", 1)[-1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        idx = int(data.get("choice", 1))
        reason = str(data.get("reason", ""))[:120]
        if 1 <= idx <= len(plans):
            return idx, reason
    except Exception as exc:  # noqa: BLE001
        logger.warning("[plan_resolver] LLM match failed: %s", exc)

    # Fallback: first active, else first.
    for i, p in enumerate(plans):
        if p.get("status") in ("draft", "confirmed"):
            return i + 1, "fallback-active"
    return 1, "fallback-first"


def resolve_plan(
    user_id: str,
    query: str,
    plan_type: str,
) -> dict[str, Any] | None:
    """Pick the plan that best matches ``query`` for ``user_id``.

    Returns ``{"plan": dict, "is_archived": bool, "match_reason": str}``
    or ``None`` if the user has no plans. Fails soft — on any error
    returns the active plan (or None).
    """
    if not user_id:
        return None
    repo = _repo_for(plan_type)

    try:
        all_plans = repo.find_all_by_user(user_id, limit=50)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[plan_resolver] find_all_by_user failed: %s", exc)
        all_plans = []

    if not all_plans:
        return None

    active = next(
        (p for p in all_plans if p.get("status") in ("draft", "confirmed")),
        None,
    )
    archived = [p for p in all_plans if p.get("status") == "archived"]

    # No archived plans → active plan is the only choice.
    if not archived:
        if active is None:
            return None
        return {
            "plan": active,
            "is_archived": False,
            "match_reason": "only-active-plan",
        }

    # Query has no distinguishing descriptor → active plan.
    if not _has_descriptor(query) and active is not None:
        return {
            "plan": active,
            "is_archived": False,
            "match_reason": "no-descriptor-in-query",
        }

    # Otherwise rank with LLM over all plans.
    idx, reason = _llm_match(query, all_plans)
    chosen = all_plans[idx - 1]
    return {
        "plan": chosen,
        "is_archived": chosen.get("status") == "archived",
        "match_reason": reason,
    }
