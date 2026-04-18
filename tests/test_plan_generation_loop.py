"""
tests/test_plan_generation_loop.py
──────────────────────────────────
Integration tests for the generate-evaluate-retry orchestrator.

We stub both the underlying ``generate_plan`` and ``evaluate_plan`` so
the tests run without any LLM calls and focus on the loop's control
flow and selection policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.shared import plan_generation_loop as loop
from agent.shared.plan_evaluator import (
    EvalResult,
    HardConstraintResult,
    LightEvalResult,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _eval(hard: bool, score: float, reasons: list[str] | None = None) -> EvalResult:
    return EvalResult(
        hard=HardConstraintResult(passed=hard, reasons=list(reasons or [])),
        light=LightEvalResult(score=score, reasons=[], criteria={
            "relevance_to_goals": score,
            "completeness": score,
            "domain_compliance": score,
            "structural_quality": score,
        }),
        combined_score=0.5 * (1.0 if hard else 0.0) + 0.5 * score,
    )


class _GenStub:
    """Sequentially returns queued plan markdown strings."""

    def __init__(self, plans):
        self.plans = list(plans)
        self.calls: list[str] = []

    def __call__(self, domain, profile, query, system_prompt, *, existing_plan=None):
        self.calls.append(system_prompt)
        return self.plans.pop(0)


BASE_PROFILE = {"sex": "male", "goal": "fat loss"}


# ── Tests ───────────────────────────────────────────────────────────

def test_first_attempt_passes_no_retry(monkeypatch):
    gen = _GenStub(["# Plan A"])
    evals = iter([_eval(True, 0.9)])

    monkeypatch.setattr(loop, "generate_plan", gen)
    monkeypatch.setattr(loop, "evaluate_plan", lambda *a, **kw: next(evals))
    monkeypatch.setattr(loop, "_ENABLED", True)
    monkeypatch.setattr(loop, "_MAX_RETRIES", 1)
    monkeypatch.setattr(loop, "_MIN_SCORE", 0.6)

    plan, meta = loop.generate_plan_with_feedback(
        "diet", BASE_PROFILE, "req", "SYS",
    )

    assert plan == "# Plan A"
    assert meta["attempts"] == 1
    assert meta["chosen_index"] == 0
    assert meta["hard_passed"] is True
    assert len(gen.calls) == 1


def test_retry_fires_when_first_attempt_fails(monkeypatch):
    gen = _GenStub(["# Bad", "# Good"])
    evals = iter([
        _eval(False, 0.3, reasons=["missing FITGEN_DATA block"]),
        _eval(True, 0.9),
    ])

    monkeypatch.setattr(loop, "generate_plan", gen)
    monkeypatch.setattr(loop, "evaluate_plan", lambda *a, **kw: next(evals))
    monkeypatch.setattr(loop, "_ENABLED", True)
    monkeypatch.setattr(loop, "_MAX_RETRIES", 1)
    monkeypatch.setattr(loop, "_MIN_SCORE", 0.6)

    plan, meta = loop.generate_plan_with_feedback(
        "diet", BASE_PROFILE, "req", "SYS",
    )

    assert plan == "# Good"
    assert meta["attempts"] == 2
    assert meta["chosen_index"] == 1
    # Retry prompt must include the previous failure reason.
    assert "FITGEN_DATA" in gen.calls[1]
    assert "PREVIOUS ATTEMPT FAILED" in gen.calls[1]


def test_both_attempts_fail_returns_highest_scoring(monkeypatch):
    gen = _GenStub(["# Bad1", "# Bad2"])
    evals = iter([
        _eval(False, 0.3),
        _eval(False, 0.7),  # higher light score, still hard-fail
    ])

    monkeypatch.setattr(loop, "generate_plan", gen)
    monkeypatch.setattr(loop, "evaluate_plan", lambda *a, **kw: next(evals))
    monkeypatch.setattr(loop, "_ENABLED", True)
    monkeypatch.setattr(loop, "_MAX_RETRIES", 1)
    monkeypatch.setattr(loop, "_MIN_SCORE", 0.6)

    plan, meta = loop.generate_plan_with_feedback(
        "diet", BASE_PROFILE, "req", "SYS",
    )

    assert plan == "# Bad2"  # higher combined_score
    assert meta["attempts"] == 2
    assert meta["chosen_index"] == 1
    assert meta["hard_passed"] is False


def test_hard_pass_beats_higher_light_score(monkeypatch):
    """Policy: any hard-passing attempt outranks any hard-failing one."""
    gen = _GenStub(["# Passes hard, lowish light", "# Fails hard, great light"])
    evals = iter([
        _eval(True, 0.4),   # combined = 0.5 + 0.2 = 0.7
        _eval(False, 0.95), # combined = 0 + 0.475 = 0.475
    ])

    monkeypatch.setattr(loop, "generate_plan", gen)
    monkeypatch.setattr(loop, "evaluate_plan", lambda *a, **kw: next(evals))
    monkeypatch.setattr(loop, "_ENABLED", True)
    monkeypatch.setattr(loop, "_MAX_RETRIES", 1)
    monkeypatch.setattr(loop, "_MIN_SCORE", 0.6)

    plan, meta = loop.generate_plan_with_feedback(
        "diet", BASE_PROFILE, "req", "SYS",
    )
    assert plan == "# Passes hard, lowish light"
    assert meta["chosen_index"] == 0


def test_disabled_bypass_makes_single_call(monkeypatch):
    gen = _GenStub(["# One-shot"])
    eval_calls = {"n": 0}

    def _never_eval(*a, **kw):
        eval_calls["n"] += 1
        raise AssertionError("evaluator must not run when disabled")

    monkeypatch.setattr(loop, "generate_plan", gen)
    monkeypatch.setattr(loop, "evaluate_plan", _never_eval)
    monkeypatch.setattr(loop, "_ENABLED", False)

    plan, meta = loop.generate_plan_with_feedback(
        "diet", BASE_PROFILE, "req", "SYS",
    )

    assert plan == "# One-shot"
    assert meta == {"attempts": 1, "chosen_index": 0, "eval_enabled": False}
    assert eval_calls["n"] == 0
    assert len(gen.calls) == 1


def test_evaluator_error_falls_through_to_first_attempt(monkeypatch):
    gen = _GenStub(["# A", "# B"])

    def _boom(*a, **kw):
        raise RuntimeError("evaluator down")

    monkeypatch.setattr(loop, "generate_plan", gen)
    monkeypatch.setattr(loop, "evaluate_plan", _boom)
    monkeypatch.setattr(loop, "_ENABLED", True)
    monkeypatch.setattr(loop, "_MAX_RETRIES", 1)

    plan, meta = loop.generate_plan_with_feedback(
        "diet", BASE_PROFILE, "req", "SYS",
    )

    # Evaluator crashed, loop breaks out, attempt 1 is the only survivor.
    assert plan == "# A"
    assert meta["attempts"] == 1
    assert meta["hard_passed"] is None
    assert meta["light_score"] is None


def test_existing_plan_kwarg_propagates(monkeypatch):
    """Update flow must pass ``existing_plan`` to the generator."""
    received = {}

    def _gen(domain, profile, query, sys, *, existing_plan=None):
        received["existing_plan"] = existing_plan
        return "# Updated"

    monkeypatch.setattr(loop, "generate_plan", _gen)
    monkeypatch.setattr(loop, "evaluate_plan", lambda *a, **kw: _eval(True, 0.9))
    monkeypatch.setattr(loop, "_ENABLED", True)

    plan, _ = loop.generate_plan_with_feedback(
        "diet", BASE_PROFILE, "tweak", "SYS", existing_plan="# Old plan body",
    )
    assert plan == "# Updated"
    assert received["existing_plan"] == "# Old plan body"


def test_generator_error_on_first_attempt_propagates(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("openai 500")

    monkeypatch.setattr(loop, "generate_plan", _boom)
    monkeypatch.setattr(loop, "_ENABLED", True)

    with pytest.raises(RuntimeError):
        loop.generate_plan_with_feedback(
            "diet", BASE_PROFILE, "req", "SYS",
        )


def test_log_event_fires_once_per_call(monkeypatch):
    events: list[dict] = []

    def _capture(name, *, module="", **fields):
        events.append({"name": name, "module": module, **fields})

    gen = _GenStub(["# One"])
    monkeypatch.setattr(loop, "generate_plan", gen)
    monkeypatch.setattr(loop, "evaluate_plan", lambda *a, **kw: _eval(True, 0.9))
    monkeypatch.setattr(loop, "log_event", _capture)
    monkeypatch.setattr(loop, "_ENABLED", True)

    loop.generate_plan_with_feedback("diet", BASE_PROFILE, "req", "SYS")

    assert len(events) == 1
    assert events[0]["name"] == "plan_eval_loop"
    assert events[0]["attempts"] == 1
    assert events[0]["chosen_index"] == 0
