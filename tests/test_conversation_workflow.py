"""
tests/test_conversation_workflow.py
───────────────────────────────────
Unit tests for pure logic in conversation_workflow.py.
No LLM calls required — tests deterministic helpers only.
"""

import pytest

from agent.tools.conversation_workflow import (
    BASE_PROFILE_FIELDS,
    DOMAIN_REQUIRED_FIELDS,
    FIELD_QUESTION,
    GOAL_KEYWORDS,
    SEX_MAP,
    _parse_single_field,
    append_completed_step,
    build_profile_bulk_question,
    build_profile_confirmation,
    fallback_yes_no_from_text,
    is_generic_modify_request,
    looks_like_modify_request,
    missing_profile_fields,
    required_fields_for_domain,
    resolve_intent,
)


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
# fallback_yes_no_from_text
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestFallbackYesNo:
    def test_yes_variants(self):
        for word in ["yes", "y", "yeah", "yep", "ok", "sure"]:
            assert fallback_yes_no_from_text(word) == "yes"

    def test_no_variants(self):
        for word in ["no", "n", "nope", "nah", "cancel"]:
            assert fallback_yes_no_from_text(word) == "no"

    def test_unknown(self):
        assert fallback_yes_no_from_text("I'm not sure about this") == "unknown"

    def test_positive_feedback_disabled(self):
        assert fallback_yes_no_from_text("looks good") == "unknown"

    def test_positive_feedback_enabled(self):
        assert fallback_yes_no_from_text("looks good", allow_positive_feedback=True) == "yes"
        assert fallback_yes_no_from_text("perfect", allow_positive_feedback=True) == "yes"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# looks_like_modify_request / is_generic_modify_request
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestModifyDetection:
    def test_modify_markers(self):
        assert looks_like_modify_request("change my diet plan") is True
        assert looks_like_modify_request("update the workout") is True
        assert looks_like_modify_request("I want to avoid seafood") is True

    def test_not_modify(self):
        assert looks_like_modify_request("hello") is False
        assert looks_like_modify_request("tell me about protein") is False

    def test_generic_modify(self):
        assert is_generic_modify_request("modify it") is True
        assert is_generic_modify_request("update") is True

    def test_specific_modify_not_generic(self):
        assert is_generic_modify_request("change my calories to 2500") is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# resolve_intent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestResolveIntent:
    def test_direct_intents(self):
        assert resolve_intent("get", None, None, "show my plan") == "get"
        assert resolve_intent("delete", None, None, "remove it") == "delete"
        assert resolve_intent("update", None, None, "change it") == "update"

    def test_modify_in_calendar_sync_stage(self):
        result = resolve_intent("other", "create", "calendar_sync", "change my diet")
        assert result == "update"

    def test_modify_in_plan_feedback_stage(self):
        result = resolve_intent("other", "create", "plan_feedback", "modify the workout")
        assert result == "update"

    def test_fallback_to_active_intent(self):
        result = resolve_intent("other", "create", "collect_profile", "some text")
        assert result == "create"

    def test_fallback_to_detected_when_no_active(self):
        result = resolve_intent("other", None, None, "random text")
        assert result == "other"


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
        assert "What is your age?" in result
        assert "What is your sex" in result
        assert "Example:" in result

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
