"""
tests/test_plan_evaluator.py
────────────────────────────
Unit tests for the plan evaluator's deterministic layer (hard
constraints) and its light-rubric aggregation logic.  The rubric LLM
itself is stubbed — we're testing the wiring and score math, not the
model's taste.
"""

from __future__ import annotations

import pytest

from agent.shared import plan_evaluator as pe


# ── Test fixtures ────────────────────────────────────────────────────

# Minimal well-formed diet plan (7 days, FITGEN_DATA, calorie floor met).
GOOD_DIET_PLAN = """\
# 7-Day Diet Plan

## Monday
| Meal | Food | kcal |
|---|---|---|
| Breakfast | Oats | 400 |
| Lunch | Chicken bowl | 600 |
| Dinner | Fish & rice | 700 |
Total: 1700 kcal

## Tuesday
- Breakfast: Eggs & toast (450)
- Lunch: Paneer wrap (650)
- Dinner: Dal tadka (650)

## Wednesday
- Breakfast: Smoothie (400)
- Lunch: Salmon salad (650)
- Dinner: Chicken curry (700)

## Thursday
- Breakfast: Oatmeal (400)
- Lunch: Tuna salad (600)
- Dinner: Tofu stir-fry (700)

## Friday
- Breakfast: Yogurt bowl (400)
- Lunch: Quinoa bowl (650)
- Dinner: Roast chicken (700)

## Saturday
- Breakfast: Pancakes (500)
- Lunch: Lentil soup (550)
- Dinner: Grilled fish (700)

## Sunday
- Breakfast: Scramble (450)
- Lunch: Chicken wrap (650)
- Dinner: Pasta (650)

<!-- FITGEN_DATA
{
  "calorie_target": 1800,
  "macro_targets": {"protein_g": 140, "carbs_g": 180, "fat_g": 60}
}
-->
"""

GOOD_WORKOUT_PLAN = """\
# 7-Day Workout Plan

## Monday — Push
- Bench press 3x8 @ RPE 7
- Overhead press 3x10
- Triceps pushdown 3x12

## Tuesday — Pull
- Pull-ups 3x8
- Barbell rows 3x10
- Biceps curls 3x12

## Wednesday — Legs
- Squats 3x8
- Romanian deadlifts 3x10
- Calf raises 3x15

## Thursday — Rest
Active recovery, walk 30 min.

## Friday — Push
- Incline press 3x8
- Dumbbell shoulder press 3x10

## Saturday — Pull
- Deadlifts 3x5
- Chin-ups 3x8

## Sunday — Rest
Stretch & mobility.

<!-- FITGEN_DATA
{"split":"PPL", "sessions_per_week": 5}
-->
"""

BASE_PROFILE = {
    "sex": "male",
    "age": 28,
    "weight_kg": 75,
    "height_cm": 178,
    "goal": "lose fat",
    "allergies": "none",
    "foods_to_avoid": "none",
    "fitness_level": "intermediate",
}


# ── Hard constraint tests ───────────────────────────────────────────

def test_hard_pass_on_good_diet_plan():
    res = pe.check_hard_constraints(
        GOOD_DIET_PLAN, domain="diet", profile=BASE_PROFILE,
    )
    assert res.passed, res.reasons


def test_hard_pass_on_good_workout_plan():
    res = pe.check_hard_constraints(
        GOOD_WORKOUT_PLAN, domain="workout", profile=BASE_PROFILE,
    )
    assert res.passed, res.reasons


def test_hard_fail_on_empty_markdown():
    res = pe.check_hard_constraints("", domain="diet", profile=BASE_PROFILE)
    assert not res.passed
    assert any("empty" in r.lower() for r in res.reasons)


def test_hard_fail_on_too_few_days():
    short = "# Plan\n\n## Monday\n- food\n\n<!-- FITGEN_DATA {} -->"
    res = pe.check_hard_constraints(
        short, domain="diet", profile=BASE_PROFILE,
    )
    assert not res.passed
    assert any("H2 section" in r for r in res.reasons)


def test_hard_fail_on_missing_fitgen_data_block():
    no_meta = GOOD_DIET_PLAN.split("<!-- FITGEN_DATA")[0]
    res = pe.check_hard_constraints(
        no_meta, domain="diet", profile=BASE_PROFILE,
    )
    assert not res.passed
    assert any("FITGEN_DATA" in r for r in res.reasons)


def test_hard_fail_on_allergen_leakage():
    profile = {**BASE_PROFILE, "allergies": "peanut"}
    leaky = GOOD_DIET_PLAN.replace(
        "Oats",
        "Peanut butter toast",  # triggers allergen match in json blob text
    )
    # Also inject peanut into the FITGEN_DATA so validate_plan_json
    # (which only looks at the parsed JSON) actually sees it.
    leaky = leaky.replace(
        '"calorie_target": 1800,',
        '"calorie_target": 1800, "notes": "peanut butter preferred",',
    )
    res = pe.check_hard_constraints(
        leaky, domain="diet", profile=profile,
    )
    assert not res.passed
    assert any("peanut" in r.lower() for r in res.reasons)


def test_hard_fail_on_calorie_floor_violation():
    below_floor = GOOD_DIET_PLAN.replace(
        '"calorie_target": 1800,',
        '"calorie_target": 900,',
    )
    res = pe.check_hard_constraints(
        below_floor, domain="diet", profile=BASE_PROFILE,
    )
    assert not res.passed
    assert any("floor" in r.lower() for r in res.reasons)


def test_hard_fail_on_workout_day_without_exercises():
    bad = GOOD_WORKOUT_PLAN.replace(
        "## Monday — Push\n- Bench press 3x8 @ RPE 7\n"
        "- Overhead press 3x10\n- Triceps pushdown 3x12",
        "## Monday — Push\nNo exercises here, just a pep talk.",
    )
    res = pe.check_hard_constraints(
        bad, domain="workout", profile=BASE_PROFILE,
    )
    assert not res.passed
    assert any("no exercise" in r.lower() for r in res.reasons)


def test_rest_day_without_exercises_is_allowed():
    # The good workout plan already has rest days without exercise lines —
    # the above sanity check covers the positive case.
    res = pe.check_hard_constraints(
        GOOD_WORKOUT_PLAN, domain="workout", profile=BASE_PROFILE,
    )
    assert res.passed


# ── Light-eval wiring tests (stubbed LLM) ───────────────────────────

class _FakeScoreObj:
    """Duck-typed replacement for the pydantic _CriteriaScore."""

    def __init__(self, *, rel=1.0, comp=1.0, dom=1.0, struct=1.0, reasoning=""):
        self.relevance_to_goals = rel
        self.completeness = comp
        self.domain_compliance = dom
        self.structural_quality = struct
        self.reasoning = reasoning


class _FakeEvaluator:
    def __init__(self, score_obj):
        self.score_obj = score_obj
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self.score_obj


def test_light_eval_aggregates_to_mean(monkeypatch):
    fake = _FakeEvaluator(
        _FakeScoreObj(rel=1.0, comp=1.0, dom=0.6, struct=0.6, reasoning="ok")
    )
    monkeypatch.setattr(pe, "_get_evaluator_llm", lambda: fake)

    res = pe.run_light_eval(
        GOOD_DIET_PLAN,
        domain="diet",
        profile=BASE_PROFILE,
        user_request="make me a fat-loss diet",
    )
    # Equal-weighted mean of the four criteria.
    assert res.score == pytest.approx((1.0 + 1.0 + 0.6 + 0.6) / 4.0)
    assert res.criteria == {
        "relevance_to_goals": 1.0,
        "completeness": 1.0,
        "domain_compliance": 0.6,
        "structural_quality": 0.6,
    }
    assert fake.calls == 1


def test_light_eval_on_empty_plan_short_circuits(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(
        pe, "_get_evaluator_llm",
        lambda: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    res = pe.run_light_eval(
        "", domain="diet", profile=BASE_PROFILE, user_request="x",
    )
    assert res.score == 0.0
    assert called["n"] == 0


def test_light_eval_swallows_llm_errors(monkeypatch):
    class _Boom:
        def invoke(self, _messages):
            raise RuntimeError("rate limit")

    monkeypatch.setattr(pe, "_get_evaluator_llm", lambda: _Boom())
    res = pe.run_light_eval(
        GOOD_DIET_PLAN,
        domain="diet",
        profile=BASE_PROFILE,
        user_request="go",
    )
    # Neutral score so the loop neither retries needlessly nor penalises.
    assert res.score == 0.5
    assert any("error" in r.lower() for r in res.reasons)


# ── Combined evaluate_plan ──────────────────────────────────────────

def test_evaluate_plan_composite_math(monkeypatch):
    fake = _FakeEvaluator(_FakeScoreObj(rel=0.8, comp=0.8, dom=0.8, struct=0.8))
    monkeypatch.setattr(pe, "_get_evaluator_llm", lambda: fake)

    result = pe.evaluate_plan(
        GOOD_DIET_PLAN,
        domain="diet",
        profile=BASE_PROFILE,
        user_request="fat loss",
    )
    assert result.hard.passed is True
    assert result.light.score == pytest.approx(0.8)
    # combined = 0.5 * hard (1.0) + 0.5 * light (0.8)
    assert result.combined_score == pytest.approx(0.9)
    assert result.passed is True


def test_evaluate_plan_hard_fail_halves_composite(monkeypatch):
    fake = _FakeEvaluator(_FakeScoreObj(rel=0.9, comp=0.9, dom=0.9, struct=0.9))
    monkeypatch.setattr(pe, "_get_evaluator_llm", lambda: fake)

    # Drop the FITGEN_DATA block → hard fail, but light score still 0.9.
    no_meta = GOOD_DIET_PLAN.split("<!-- FITGEN_DATA")[0]
    result = pe.evaluate_plan(
        no_meta,
        domain="diet",
        profile=BASE_PROFILE,
        user_request="fat loss",
    )
    assert not result.hard.passed
    assert result.light.score == pytest.approx(0.9)
    # combined = 0 + 0.5 * 0.9
    assert result.combined_score == pytest.approx(0.45)
    assert result.passed is False
