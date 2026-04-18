"""
agent/shared/plan_evaluator.py
──────────────────────────────
Lightweight evaluator for plans returned by ``generate_plan()``.

Two layers:

1. **Hard constraints** (deterministic, ~1ms) — structural checks that
   catch guaranteed-bad output: empty markdown, missing day sections,
   calorie-floor violations, declared-allergen leakage, missing
   ``FITGEN_DATA`` metadata block.  Reuses the existing
   :func:`agent.shared.llm_helpers.validate_plan_json` where possible.

2. **Light eval** (one call to the fast model) — a multi-criteria
   rubric score.  Functionally equivalent to LangChain's
   ``CriteriaEvalChain`` but implemented directly against
   ``ChatOpenAI`` because ``langchain.evaluation`` was removed from
   core ``langchain`` in the 1.x release and moved to
   ``langchain-community`` (uninstalled).  Semantics are identical:
   per-criterion score in [0, 1], brief reasoning, overall verdict.

The evaluator is an internal signal for the retry loop in
:mod:`agent.shared.plan_generation_loop` — it is never surfaced to the
end user.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agent.config import FAST_MODEL
from agent.error_utils import handle_exception
from agent.shared.llm_helpers import validate_plan_json

logger = logging.getLogger("fitgen.plan_evaluator")

# ── Env-driven knobs ────────────────────────────────────────────────
_EVAL_MODEL = os.getenv("PLAN_EVAL_MODEL", FAST_MODEL)


# ── Result objects ──────────────────────────────────────────────────

@dataclass
class HardConstraintResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class LightEvalResult:
    score: float          # aggregate in [0, 1]
    reasons: list[str] = field(default_factory=list)
    criteria: dict[str, float] = field(default_factory=dict)  # per-criterion


@dataclass
class EvalResult:
    hard: HardConstraintResult
    light: LightEvalResult
    combined_score: float    # blends hard-pass + light score

    @property
    def passed(self) -> bool:
        """Convenience — hard pass AND light above default threshold."""
        return self.hard.passed and self.light.score >= 0.6


# ── Hard constraints ────────────────────────────────────────────────

_FITGEN_DATA_MARKER = re.compile(r"<!--\s*FITGEN_DATA", re.IGNORECASE)
_H2_LINE = re.compile(r"^##\s+\S", re.MULTILINE)
_MIN_DAY_SECTIONS = 7


def _extract_plan_json(plan_markdown: str) -> dict[str, Any] | None:
    """Try to find an embedded JSON blob in the FITGEN_DATA comment.

    The markdown format appends a machine-readable JSON block inside an
    HTML comment (``<!-- FITGEN_DATA ... -->``).  Pulling it out lets
    us reuse :func:`validate_plan_json` for diet-specific checks.
    Returns ``None`` if no block is present or parsing fails.
    """
    if not plan_markdown:
        return None
    m = re.search(
        r"<!--\s*FITGEN_DATA\s*(.*?)\s*-->",
        plan_markdown,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None
    blob = m.group(1).strip()
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return None


def check_hard_constraints(
    plan_markdown: str,
    *,
    domain: str,
    profile: dict[str, Any],
) -> HardConstraintResult:
    """Deterministic structural checks on a plan's markdown output.

    No LLM call; cheap enough to run on every attempt.
    """
    reasons: list[str] = []

    # (1) Non-empty
    if not plan_markdown or not plan_markdown.strip():
        reasons.append("Plan markdown is empty.")
        return HardConstraintResult(passed=False, reasons=reasons)

    # (2) Enough day sections — week-long plans need 7 H2 headings
    day_sections = len(_H2_LINE.findall(plan_markdown))
    if day_sections < _MIN_DAY_SECTIONS:
        reasons.append(
            f"Plan has only {day_sections} H2 section(s); expected at least "
            f"{_MIN_DAY_SECTIONS} for a week-long {domain} plan."
        )

    # (3) FITGEN_DATA metadata block required by PDF + RAG pipelines
    if not _FITGEN_DATA_MARKER.search(plan_markdown):
        reasons.append(
            "Missing <!-- FITGEN_DATA ... --> metadata block (required "
            "by PDF export and RAG indexer)."
        )

    # (4) Domain-specific checks
    parsed = _extract_plan_json(plan_markdown)
    if domain == "diet" and parsed is not None:
        # Reuse existing validator for calorie floor / allergens / negatives.
        issues = validate_plan_json(parsed, profile)
        reasons.extend(issues)
    elif domain == "workout":
        # Each H2 day should carry at least one exercise line — a bullet
        # ("- ") or a numbered list ("1. ") under the heading.  Rest days
        # must say the word "rest" in the heading or first line.
        sections = _split_h2_sections(plan_markdown)
        for title, body in sections:
            if re.search(r"\brest\b", title, re.IGNORECASE):
                continue
            has_exercise_line = bool(
                re.search(r"^(\s*[-*]|\s*\d+\.)\s+\S", body, re.MULTILINE)
            )
            if not has_exercise_line:
                reasons.append(
                    f"Workout day '{title.strip()}' has no exercise lines "
                    f"and is not marked as a rest day."
                )

    return HardConstraintResult(passed=not reasons, reasons=reasons)


def _split_h2_sections(md: str) -> list[tuple[str, str]]:
    """Return a list of ``(title, body)`` pairs split by ``## `` headings."""
    parts = re.split(r"^##\s+(.+)$", md, flags=re.MULTILINE)
    # parts = [preamble, title1, body1, title2, body2, ...]
    out: list[tuple[str, str]] = []
    for i in range(1, len(parts) - 1, 2):
        out.append((parts[i], parts[i + 1]))
    return out


# ── Light evaluator (Criteria-style rubric) ─────────────────────────

class _CriteriaScore(BaseModel):
    """Structured output returned by the rubric LLM."""

    relevance_to_goals: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    domain_compliance: float = Field(ge=0.0, le=1.0)
    structural_quality: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(
        default="",
        description="Short (<=2 sentences) explanation of the lowest-scoring criterion."
    )


_CRITERIA_PROMPTS = {
    "diet": {
        "relevance_to_goals": (
            "Does the plan address the user's fitness goal, current diet, "
            "budget, and cooking time shown in their profile?"
        ),
        "completeness": (
            "Does every day have breakfast, lunch, dinner, snacks and a "
            "daily total in kcal + macros?"
        ),
        "domain_compliance": (
            "Does the plan respect declared allergies, foods to avoid, and "
            "keep calories above the safety floor (1500 male / 1200 female)?"
        ),
        "structural_quality": (
            "Is the markdown structured with H2 per day, readable tables, "
            "and a FITGEN_DATA JSON block?"
        ),
    },
    "workout": {
        "relevance_to_goals": (
            "Does the plan match the user's fitness goal, current fitness "
            "level, equipment access and injuries from their profile?"
        ),
        "completeness": (
            "Does every training day list exercises with sets x reps + RPE "
            "(or intensity), warm-up, and cool-down?  Are rest days clearly "
            "flagged?"
        ),
        "domain_compliance": (
            "Does it avoid contraindicated movements given the user's "
            "injuries and stay within their session_duration budget?"
        ),
        "structural_quality": (
            "Is the markdown structured with H2 per day, readable exercise "
            "lists, and a FITGEN_DATA JSON block?"
        ),
    },
}


_SYSTEM_RUBRIC = """\
You are a strict rubric evaluator. Score the plan against the four criteria
below. For each, return a float in [0.0, 1.0] where:
  1.0 = fully satisfies the criterion
  0.6 = acceptable, minor issues
  0.3 = significant gaps
  0.0 = absent or wrong

Be calibrated. Do not default to 1.0. Return ONLY the JSON object.
"""


def _build_rubric_prompt(
    *,
    domain: str,
    profile: dict[str, Any],
    user_request: str,
    plan_markdown: str,
) -> str:
    criteria = _CRITERIA_PROMPTS.get(domain, _CRITERIA_PROMPTS["diet"])
    criterion_block = "\n".join(
        f"- {name}: {desc}" for name, desc in criteria.items()
    )
    # Cap plan size to avoid blowing the eval model context on pathological output.
    plan_preview = plan_markdown
    if len(plan_preview) > 8000:
        plan_preview = plan_preview[:8000] + "\n... [truncated for eval]"
    return (
        f"Domain: {domain}\n"
        f"User profile (JSON): {json.dumps(profile, ensure_ascii=False)}\n"
        f"User request: {user_request!r}\n\n"
        f"Criteria:\n{criterion_block}\n\n"
        f"Plan to evaluate:\n---\n{plan_preview}\n---\n"
    )


# Lazy singleton — built on first use.  Swappable in tests.
_evaluator_llm: Any = None


def _get_evaluator_llm():
    global _evaluator_llm
    if _evaluator_llm is None:
        base = ChatOpenAI(model=_EVAL_MODEL, temperature=0.0)
        _evaluator_llm = base.with_structured_output(_CriteriaScore)
    return _evaluator_llm


def run_light_eval(
    plan_markdown: str,
    *,
    domain: str,
    profile: dict[str, Any],
    user_request: str,
) -> LightEvalResult:
    """Score the plan with the rubric model. Best-effort: never raises."""
    if not plan_markdown or not plan_markdown.strip():
        return LightEvalResult(score=0.0, reasons=["empty plan"])

    try:
        evaluator = _get_evaluator_llm()
        out: _CriteriaScore = evaluator.invoke([
            SystemMessage(content=_SYSTEM_RUBRIC),
            HumanMessage(content=_build_rubric_prompt(
                domain=domain,
                profile=profile,
                user_request=user_request,
                plan_markdown=plan_markdown,
            )),
        ])
    except Exception as exc:  # noqa: BLE001
        handle_exception(
            exc,
            module="plan_evaluator",
            context="run_light_eval",
            extra={"domain": domain},
        )
        # On evaluator failure, return a neutral score so the loop
        # neither retries needlessly nor penalises the generator.
        return LightEvalResult(score=0.5, reasons=[f"evaluator error: {exc!s}"])

    criteria = {
        "relevance_to_goals": float(out.relevance_to_goals),
        "completeness": float(out.completeness),
        "domain_compliance": float(out.domain_compliance),
        "structural_quality": float(out.structural_quality),
    }
    # Equal-weighted average.  Could be tuned later.
    agg = sum(criteria.values()) / len(criteria)
    reasons = [out.reasoning] if out.reasoning else []
    return LightEvalResult(score=agg, reasons=reasons, criteria=criteria)


# ── Combined entry point ────────────────────────────────────────────

def evaluate_plan(
    plan_markdown: str,
    *,
    domain: str,
    profile: dict[str, Any],
    user_request: str,
) -> EvalResult:
    """Run hard constraints + light rubric and return a composite result."""
    hard = check_hard_constraints(
        plan_markdown, domain=domain, profile=profile,
    )
    light = run_light_eval(
        plan_markdown,
        domain=domain,
        profile=profile,
        user_request=user_request,
    )
    # 0.5 * hard_pass + 0.5 * light_score
    # (hard failures effectively halve the composite — strong penalty
    #  without making them an automatic veto, so best-of selection can
    #  still prefer a high-light-score attempt over a hard-fail one).
    combined = 0.5 * (1.0 if hard.passed else 0.0) + 0.5 * light.score
    return EvalResult(hard=hard, light=light, combined_score=combined)


__all__ = [
    "HardConstraintResult",
    "LightEvalResult",
    "EvalResult",
    "check_hard_constraints",
    "run_light_eval",
    "evaluate_plan",
]
