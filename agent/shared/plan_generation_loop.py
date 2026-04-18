"""
agent/shared/plan_generation_loop.py
────────────────────────────────────
Generate-Evaluate-Retry orchestrator for ``generate_plan()``.

Wraps the single-shot plan generator with a feedback loop:

    attempt 1 → evaluate → (pass?) → return
                       ↓ no
    attempt 2 with feedback injected → evaluate →
                       ↓
    select best-scoring attempt (hard-pass preferred)

Gated by the ``PLAN_EVAL_ENABLED`` env var so the entire feature can
be killed without a code change.  All selection logic is
deterministic; the only LLM calls are the plan generation itself plus
one fast-model rubric evaluation per attempt.

Failures in the evaluator never escape — the wrapper swallows and
returns whatever the generator produced.  This is a quality gate,
not a hard dependency.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from agent.error_utils import handle_exception
from agent.shared.llm_helpers import generate_plan
from agent.shared.plan_evaluator import EvalResult, evaluate_plan
from agent.tracing import log_event

logger = logging.getLogger("fitgen.plan_generation_loop")


# ── Env knobs ───────────────────────────────────────────────────────

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


_ENABLED = _env_bool("PLAN_EVAL_ENABLED", True)
_MAX_RETRIES = _env_int("PLAN_EVAL_MAX_RETRIES", 1)
_MIN_SCORE = _env_float("PLAN_EVAL_MIN_SCORE", 0.6)


# ── Attempt bookkeeping ─────────────────────────────────────────────

@dataclass
class _Attempt:
    index: int
    plan_markdown: str
    eval_result: EvalResult | None  # None if the evaluator itself errored


def _build_feedback_block(eval_result: EvalResult) -> str:
    """Turn a failed eval into a short, prompt-safe feedback block."""
    reasons: list[str] = []
    if eval_result.hard.reasons:
        reasons.append("Hard constraint failures:")
        reasons.extend(f"  - {r}" for r in eval_result.hard.reasons)
    if eval_result.light.criteria:
        low = [
            f"{name} = {score:.2f}"
            for name, score in eval_result.light.criteria.items()
            if score < _MIN_SCORE
        ]
        if low:
            reasons.append(
                "Low rubric scores — rewrite to raise these: "
                + ", ".join(low)
            )
    if eval_result.light.reasons:
        reasons.append(
            "Evaluator notes: " + " | ".join(eval_result.light.reasons)
        )
    if not reasons:
        return ""
    return (
        "\n\n--- PREVIOUS ATTEMPT FAILED VALIDATION ---\n"
        + "\n".join(reasons)
        + "\n--- PLEASE ADDRESS THE ISSUES ABOVE AND REGENERATE THE FULL PLAN ---"
    )


def _select_best(attempts: list[_Attempt]) -> _Attempt:
    """Pick the best attempt.

    Policy:
      1. Any attempt that passed hard constraints wins over any that didn't.
      2. Within a group, higher ``combined_score`` wins.
      3. Ties broken by attempt index (earlier wins — cheaper path).
    """
    if not attempts:
        raise ValueError("No attempts to select from")

    def sort_key(a: _Attempt) -> tuple[int, float, int]:
        if a.eval_result is None:
            return (0, 0.0, -a.index)  # evaluator-errored attempts rank low
        hard_pass = 1 if a.eval_result.hard.passed else 0
        return (hard_pass, a.eval_result.combined_score, -a.index)

    return max(attempts, key=sort_key)


# ── Public entry point ──────────────────────────────────────────────

def generate_plan_with_feedback(
    domain: str,
    profile: dict[str, Any],
    user_request: str,
    system_prompt: str,
    *,
    existing_plan: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Drop-in replacement for ``generate_plan()`` with eval + retry.

    Same inputs as the underlying generator plus returns a ``(markdown,
    metadata)`` tuple.  Callers that don't care about the metadata can
    do ``plan, _ = generate_plan_with_feedback(...)``.

    Metadata shape::
        {
            "attempts": 2,
            "chosen_index": 1,
            "hard_passed": True,
            "light_score": 0.82,
            "combined_score": 0.91,
            "eval_enabled": True,
        }
    """
    # Kill switch — fall back to a single-shot call with no eval.
    if not _ENABLED:
        plan = generate_plan(
            domain, profile, user_request, system_prompt,
            existing_plan=existing_plan,
        )
        return plan, {
            "attempts": 1,
            "chosen_index": 0,
            "eval_enabled": False,
        }

    max_attempts = max(1, _MAX_RETRIES + 1)
    attempts: list[_Attempt] = []
    feedback_suffix: str = ""

    for i in range(max_attempts):
        prompt_for_this_attempt = system_prompt + feedback_suffix
        try:
            plan_md = generate_plan(
                domain, profile, user_request, prompt_for_this_attempt,
                existing_plan=existing_plan,
            )
        except Exception as exc:  # noqa: BLE001
            # generate_plan() already logs via handle_exception; if it
            # raises we want the loop to surface the error, same as the
            # single-shot path would have.
            handle_exception(
                exc,
                module="plan_generation_loop",
                context="generate_plan_with_feedback",
                extra={"attempt": i + 1, "domain": domain},
            )
            # If the first attempt itself blows up we cannot recover.
            if not attempts:
                raise
            break

        try:
            result = evaluate_plan(
                plan_md,
                domain=domain,
                profile=profile,
                user_request=user_request,
            )
            attempts.append(_Attempt(index=i, plan_markdown=plan_md, eval_result=result))
        except Exception as exc:  # noqa: BLE001
            handle_exception(
                exc,
                module="plan_generation_loop",
                context="evaluate_plan",
                extra={"attempt": i + 1, "domain": domain},
            )
            # Evaluator failures are non-fatal — keep the attempt,
            # mark eval as missing, and move on.
            attempts.append(_Attempt(index=i, plan_markdown=plan_md, eval_result=None))
            break

        if result.hard.passed and result.light.score >= _MIN_SCORE:
            break  # good enough, stop retrying

        # Prepare feedback for the next attempt (if any left).
        feedback_suffix = _build_feedback_block(result)

    best = _select_best(attempts)

    # One structured event for observability — LangSmith + local log.
    _emit_trace(domain=domain, attempts=attempts, chosen=best)

    return best.plan_markdown, _build_metadata(attempts, best)


def _build_metadata(attempts: list[_Attempt], chosen: _Attempt) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "attempts": len(attempts),
        "chosen_index": chosen.index,
        "eval_enabled": True,
    }
    if chosen.eval_result is not None:
        meta["hard_passed"] = chosen.eval_result.hard.passed
        meta["light_score"] = round(chosen.eval_result.light.score, 4)
        meta["combined_score"] = round(chosen.eval_result.combined_score, 4)
        meta["criteria"] = chosen.eval_result.light.criteria
        if chosen.eval_result.hard.reasons:
            meta["hard_reasons"] = chosen.eval_result.hard.reasons
    else:
        meta["hard_passed"] = None
        meta["light_score"] = None
        meta["combined_score"] = None
    return meta


def _emit_trace(
    *,
    domain: str,
    attempts: list[_Attempt],
    chosen: _Attempt,
) -> None:
    per_attempt = []
    for a in attempts:
        row: dict[str, Any] = {"index": a.index}
        if a.eval_result is not None:
            row["hard_passed"] = a.eval_result.hard.passed
            row["light_score"] = round(a.eval_result.light.score, 4)
            row["combined_score"] = round(a.eval_result.combined_score, 4)
        else:
            row["eval"] = "errored"
        per_attempt.append(row)

    log_event(
        "plan_eval_loop",
        module="plan_generation_loop",
        domain=domain,
        attempts=len(attempts),
        chosen_index=chosen.index,
        per_attempt=per_attempt,
    )


__all__ = [
    "generate_plan_with_feedback",
]
