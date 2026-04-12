"""
tests/test_conversation_workflow.py
───────────────────────────────────
Unit tests for pure logic in shared utilities (formerly conversation_workflow.py).
No LLM calls required — tests deterministic helpers only.
"""

import pytest

from agent.shared.types import (
    BASE_PROFILE_FIELDS,
    DOMAIN_REQUIRED_FIELDS,
    FIELD_QUESTION,
    GOAL_KEYWORDS,
    SEX_MAP,
)
from agent.shared.profile_utils import (
    _parse_single_field,
    build_profile_bulk_question,
    build_profile_confirmation,
    missing_profile_fields,
    required_fields_for_domain,
)
from agent.shared.response_builder import append_completed_step


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# missing_profile_fields
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestMissingProfileFields:
    def test_empty_profile_returns_all(self):
        fields = required_fields_for_domain("workout")
        missing = missing_profile_fields({}, fields)
        assert missing == fields

    def test_complete_profile_returns_none(self, sample_user_profile):
        fields = required_fields_for_domain("workout")
        missing = missing_profile_fields(sample_user_profile, fields)
        assert missing == []

    def test_partial_profile(self):
        profile = {"name": "Test", "age": 25}
        fields = required_fields_for_domain("workout")
        missing = missing_profile_fields(profile, fields)
        assert "name" not in missing
        assert "age" not in missing
        assert "sex" in missing

    def test_none_value_counts_as_missing(self):
        profile = {"name": None, "age": 25}
        fields = ["name", "age"]
        missing = missing_profile_fields(profile, fields)
        assert "name" in missing
        assert "age" not in missing

    def test_empty_string_counts_as_missing(self):
        profile = {"name": "", "age": 25}
        fields = ["name", "age"]
        missing = missing_profile_fields(profile, fields)
        assert "name" in missing


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _parse_single_field
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestParseSingleField:
    def test_age_integer(self):
        assert _parse_single_field("age", "28") == 28

    def test_age_out_of_range(self):
        assert _parse_single_field("age", "5") is None
        assert _parse_single_field("age", "150") is None

    def test_sex_male(self):
        assert _parse_single_field("sex", "m") == "male"
        assert _parse_single_field("sex", "male") == "male"

    def test_sex_female(self):
        assert _parse_single_field("sex", "f") == "female"

    def test_sex_invalid(self):
        assert _parse_single_field("sex", "unknown_value") is None

    def test_height_cm(self):
        assert _parse_single_field("height_cm", "183") == 183.0

    def test_weight_kg(self):
        assert _parse_single_field("weight_kg", "80.5") == 80.5

    def test_goal_fat_loss(self):
        assert _parse_single_field("goal", "lose fat") == "fat loss"
        assert _parse_single_field("goal", "weight loss") == "fat loss"

    def test_goal_muscle_gain(self):
        assert _parse_single_field("goal", "gain muscle") == "muscle gain"
        assert _parse_single_field("goal", "bulk") == "muscle gain"

    def test_goal_invalid(self):
        assert _parse_single_field("goal", "hello world") is None

    def test_activity_level(self):
        assert _parse_single_field("activity_level", "moderate") == "moderate"
        assert _parse_single_field("activity_level", "high") == "high"

    def test_fitness_level(self):
        assert _parse_single_field("fitness_level", "beginner") == "beginner"
        assert _parse_single_field("fitness_level", "advanced") == "advanced"

    def test_workout_days(self):
        assert _parse_single_field("workout_days", "5") == 5

    def test_workout_days_out_of_range(self):
        assert _parse_single_field("workout_days", "0") is None
        assert _parse_single_field("workout_days", "8") is None

    def test_name(self):
        assert _parse_single_field("name", "John") == "John"
        assert _parse_single_field("name", "123abc") is None  # contains digits

    def test_equipment(self):
        assert _parse_single_field("equipment", "dumbbells") == "dumbbells"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Formatting helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestFormattingHelpers:
    def test_build_profile_confirmation_full(self, sample_user_profile):
        fields = required_fields_for_domain("workout")
        result = build_profile_confirmation(sample_user_profile, fields)
        assert "Name: Test User" in result
        assert "Age: 25" in result
        assert "Reply yes to confirm" in result

    def test_build_profile_confirmation_empty(self):
        result = build_profile_confirmation({}, ["name", "age"])
        assert "No fields mapped yet" in result

    def test_build_profile_bulk_question(self):
        result = build_profile_bulk_question(["age", "sex", "height_cm"])
        assert "age" in result.lower()
        assert "sex" in result.lower()
        assert "height" in result.lower()

    def test_append_completed_step(self):
        workflow = {"intent": "create", "completed_steps": []}
        result = append_completed_step(workflow, {"stage": "plan_feedback"}, "profile_confirmed")
        assert "profile_confirmed" in result["completed_steps"]
        assert result["stage"] == "plan_feedback"

    def test_append_completed_step_no_duplicates(self):
        workflow = {"completed_steps": ["step1"]}
        result = append_completed_step(workflow, {}, "step1")
        assert result["completed_steps"].count("step1") == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# required_fields_for_domain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestRequiredFields:
    def test_workout_fields(self):
        fields = required_fields_for_domain("workout")
        assert "fitness_level" in fields
        assert "equipment" in fields
        assert "workout_days" in fields

    def test_diet_fields(self):
        fields = required_fields_for_domain("diet")
        assert "diet_preference" in fields
        assert "foods_to_avoid" in fields
        assert "allergies" in fields

    def test_unknown_domain_returns_base(self):
        fields = required_fields_for_domain("unknown")
        assert fields == list(BASE_PROFILE_FIELDS)
