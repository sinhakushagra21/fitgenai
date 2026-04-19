"""Shared type definitions, field constants, and lookup maps for FITGEN.AI.

This module centralises every constant that both ``diet_tool`` and
``workout_tool`` depend on so that there is a single source of truth for
field lists, validation ranges, intent literals, and human-readable
question text.

All values that previously lived in ``conversation_workflow.py`` are
re-exported from here; domain-specific tools should import from this
module instead.
"""

from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------------
# Intent type literals
# ---------------------------------------------------------------------------

DietIntent = Literal[
    "create_diet",
    "update_diet",
    "get_diet",
    "delete_diet",
    "confirm_diet",
    "sync_diet_to_google_calendar",
    "sync_diet_to_google_fit",
    "sync_diet_to_both",
    "skip_sync_diet",
    "general_diet_query",
    "restore_diet_plan",
]

WorkoutIntent = Literal[
    "create_workout",
    "update_workout",
    "get_workout",
    "delete_workout",
    "confirm_workout",
    "sync_workout_to_google_calendar",
    "sync_workout_to_google_fit",
    "sync_workout_to_both",
    "skip_sync_workout",
    "general_workout_query",
    "restore_workout_plan",
]

# ---------------------------------------------------------------------------
# Base profile fields (shared across domains)
# ---------------------------------------------------------------------------

BASE_PROFILE_FIELDS: list[str] = [
    "name",
    "age",
    "sex",
    "height_cm",
    "weight_kg",
    "goal",
    "activity_level",
]

# ---------------------------------------------------------------------------
# Diet profile fields (24 fields across 4 sections)
# ---------------------------------------------------------------------------

DIET_PROFILE_FIELDS: list[str] = [
    # Section 1 — Your Stats
    "name",
    "age",
    "sex",
    "height_cm",
    "weight_kg",
    "goal",
    "goal_weight",
    "weight_loss_pace",
    # Section 2 — Your Lifestyle
    "job_type",
    "exercise_frequency",
    "exercise_type",
    "sleep_hours",
    "stress_level",
    "alcohol_intake",
    # Section 3 — Food Preferences
    "diet_preference",
    "favourite_meals",
    "foods_to_avoid",
    "allergies",
    "cooking_style",
    "food_adventurousness",
    # Section 4 — Snack Habits
    "current_snacks",
    "snack_reason",
    "snack_preference",
    "late_night_snacking",
]

# ---------------------------------------------------------------------------
# Required / optional field separation — Diet
# ---------------------------------------------------------------------------

DIET_REQUIRED_FIELDS: list[str] = [
    "name",
    "age",
    "sex",
    "height_cm",
    "weight_kg",
    "goal",
    "diet_preference",
]

DIET_OPTIONAL_FIELDS: list[str] = [
    f for f in DIET_PROFILE_FIELDS if f not in DIET_REQUIRED_FIELDS
]

DIET_ALL_FIELDS: list[str] = DIET_REQUIRED_FIELDS + DIET_OPTIONAL_FIELDS

# ---------------------------------------------------------------------------
# Required / optional field separation — Workout
# ---------------------------------------------------------------------------

WORKOUT_REQUIRED_FIELDS: list[str] = [
    "name",
    "age",
    "sex",
    "height_cm",
    "weight_kg",
    "goal",
    "experience_level",
    "training_days_per_week",
    "session_duration",
    "job_type",
    "daily_steps",
    "sleep_hours",
    "stress_level",
]

WORKOUT_OPTIONAL_FIELDS: list[str] = [
    "additional_info",
]

WORKOUT_ALL_FIELDS: list[str] = WORKOUT_REQUIRED_FIELDS + WORKOUT_OPTIONAL_FIELDS

# ---------------------------------------------------------------------------
# Domain-required fields (legacy mapping)
# ---------------------------------------------------------------------------

DOMAIN_REQUIRED_FIELDS: dict[str, list[str]] = {
    "diet": DIET_PROFILE_FIELDS,
    "workout": WORKOUT_ALL_FIELDS,
}

# ---------------------------------------------------------------------------
# Field → question text (human-readable prompts for intake)
# ---------------------------------------------------------------------------

FIELD_QUESTION: dict[str, str] = {
    # ── Shared / base fields ────────────────────────────────────────
    "name": "What name should I use for your plan?",
    "age": "What is your age?",
    "sex": "What is your sex (male/female/other)?",
    "height_cm": "What is your height in cm?",
    "weight_kg": "What is your current weight in kg?",
    "goal": "What is your primary goal (fat loss, muscle gain, maintenance, performance)?",
    # ── Diet-specific ───────────────────────────────────────────────
    "activity_level": "What is your activity level (sedentary, light, moderate, high, athlete)?",
    "diet_preference": "What diet preference do you follow (omnivore, vegetarian, vegan, eggetarian, etc.)?",
    "foods_to_avoid": "Any foods you want to avoid?",
    "allergies": "Any allergies or intolerances?",
    # ── Workout-specific ────────────────────────────────────────────
    "experience_level": "What is your experience level (beginner, intermediate, advanced)?",
    "training_days_per_week": "How many training days per week can you commit to?",
    "session_duration": "How long can each workout session be (in minutes)?",
    "daily_steps": "Roughly how many steps do you walk daily?",
    # ── Shared lifestyle (used by both diet & workout) ──────────────
    "job_type": "What type of job do you have (desk job, on your feet, manual labour, WFH)?",
    "sleep_hours": "How many hours of sleep do you get per night?",
    "stress_level": "What is your stress level (low, moderate, high)?",
    # ── Legacy (kept for backwards compatibility) ───────────────────
    "fitness_level": "What is your current fitness level (beginner, intermediate, advanced)?",
    "equipment": "What equipment do you have access to?",
    "workout_days": "How many workout days per week can you commit to?",
}

# ---------------------------------------------------------------------------
# Normalisation / lookup maps
# ---------------------------------------------------------------------------

SEX_MAP: dict[str, str] = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female",
    "other": "other",
    "non-binary": "other",
    "nonbinary": "other",
}

GOAL_KEYWORDS: dict[str, str] = {
    "fat loss": "fat loss",
    "lose fat": "fat loss",
    "weight loss": "fat loss",
    "muscle gain": "muscle gain",
    "gain muscle": "muscle gain",
    "bulk": "muscle gain",
    "maintenance": "maintenance",
    "maintain": "maintenance",
    "performance": "performance",
}

ACTIVITY_KEYWORDS: dict[str, str] = {
    "sedentary": "sedentary",
    "light": "light",
    "moderate": "moderate",
    "high": "high",
    "athlete": "athlete",
}

# ---------------------------------------------------------------------------
# Profile validation ranges
# ---------------------------------------------------------------------------

PROFILE_VALIDATION: dict[str, dict[str, int | float | type]] = {
    "age": {"min": 10, "max": 100, "type": int},
    "height_cm": {"min": 50.0, "max": 250.0, "type": float},
    "weight_kg": {"min": 20.0, "max": 300.0, "type": float},
    "workout_days": {"min": 1, "max": 7, "type": int},
    "training_days_per_week": {"min": 1, "max": 7, "type": int},
    "session_duration": {"min": 15, "max": 180, "type": int},
    "daily_steps": {"min": 0, "max": 50000, "type": int},
    "food_adventurousness": {"min": 1, "max": 10, "type": int},
    "sleep_hours": {"min": 3, "max": 12, "type": int},
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Intent literals
    "DietIntent",
    "WorkoutIntent",
    # Base / legacy field lists
    "BASE_PROFILE_FIELDS",
    "DIET_PROFILE_FIELDS",
    "DOMAIN_REQUIRED_FIELDS",
    # Diet fields (required / optional / all)
    "DIET_REQUIRED_FIELDS",
    "DIET_OPTIONAL_FIELDS",
    "DIET_ALL_FIELDS",
    # Workout fields (required / optional / all)
    "WORKOUT_REQUIRED_FIELDS",
    "WORKOUT_OPTIONAL_FIELDS",
    "WORKOUT_ALL_FIELDS",
    # Intake questions
    "FIELD_QUESTION",
    # Normalisation maps
    "SEX_MAP",
    "GOAL_KEYWORDS",
    "ACTIVITY_KEYWORDS",
    # Validation
    "PROFILE_VALIDATION",
]
