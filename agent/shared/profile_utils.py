"""Profile field parsing, validation, and question-building utilities.

Extracted from ``conversation_workflow.py`` so that any module in the
agent package can reuse profile helpers without pulling in the full
workflow machinery (LLM calls, state management, persistence, etc.).

All domain constants (field maps, keyword lookups, validation ranges)
are imported from :pymod:`agent.shared.types`.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.shared.types import (
    FIELD_QUESTION,
    SEX_MAP,
    GOAL_KEYWORDS,
    ACTIVITY_KEYWORDS,
    DOMAIN_REQUIRED_FIELDS,
    BASE_PROFILE_FIELDS,
    PROFILE_VALIDATION,
    DIET_REQUIRED_FIELDS,
    WORKOUT_REQUIRED_FIELDS,
)

__all__ = [
    "_extract_number",
    "_parse_single_field",
    "required_fields_for_domain",
    "missing_profile_fields",
    "build_profile_confirmation",
    "build_profile_bulk_question",
    "validate_profile_field",
]

logger = logging.getLogger("fitgen.profile_utils")


# ── helpers ──────────────────────────────────────────────────────────


def _extract_number(query: str) -> float | None:
    """Extract the first numeric value from *query*.

    Supports integers and decimals (e.g. ``"183"`` → ``183.0``,
    ``"72.5 kg"`` → ``72.5``).

    Args:
        query: Free-text string that may contain a number.

    Returns:
        The extracted number as a ``float``, or ``None`` if no number
        is found or conversion fails.
    """
    match = re.search(r"\d+(?:\.\d+)?", query)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_single_field(field: str, query: str) -> Any | None:
    """Rule-based parser for a single profile field.

    Attempts to extract a normalised value for *field* from the raw
    user text *query* without making an LLM call.  Each field has
    its own heuristic (keyword matching, regex number extraction,
    enum lookup, etc.).

    Supported fields: ``sex``, ``age``, ``height_cm``, ``weight_kg``,
    ``goal``, ``activity_level``, ``fitness_level``, ``workout_days``,
    ``diet_preference``, ``foods_to_avoid``, ``allergies``,
    ``equipment``, ``name``.

    Args:
        field: The profile field key to extract.
        query: The raw user message to parse.

    Returns:
        The normalised value for the field, or ``None`` if it cannot
        be determined from *query*.
    """
    text = query.strip()
    lower = text.lower().strip()

    if field == "sex":
        return SEX_MAP.get(lower)

    if field == "age":
        value = _extract_number(text)
        if value is None:
            return None
        age = int(round(value))
        return age if 10 <= age <= 100 else None

    if field in {"height_cm", "weight_kg"}:
        value = _extract_number(text)
        if value is None:
            return None
        return round(float(value), 1)

    if field == "goal":
        for key, norm in GOAL_KEYWORDS.items():
            if key in lower:
                return norm
        return None

    if field == "activity_level":
        for key, norm in ACTIVITY_KEYWORDS.items():
            if key in lower:
                return norm
        return None

    if field in ("fitness_level", "experience_level"):
        levels = {"beginner", "intermediate", "advanced"}
        for level in levels:
            if level in lower:
                return level
        return None

    if field in ("workout_days", "training_days_per_week"):
        value = _extract_number(text)
        if value is None:
            return None
        days = int(round(value))
        return days if 1 <= days <= 7 else None

    if field == "session_duration":
        value = _extract_number(text)
        if value is None:
            return None
        minutes = int(round(value))
        return minutes if 15 <= minutes <= 180 else None

    if field == "daily_steps":
        value = _extract_number(text)
        if value is None:
            return None
        steps = int(round(value))
        return steps if 0 <= steps <= 50000 else None

    if field == "stress_level":
        for level in ("low", "moderate", "high"):
            if level in lower:
                return level
        return None

    if field == "job_type":
        _job_keywords = {
            "desk": "desk job",
            "office": "desk job",
            "wfh": "work from home",
            "work from home": "work from home",
            "remote": "work from home",
            "retail": "on my feet (retail/teaching)",
            "teaching": "on my feet (retail/teaching)",
            "on my feet": "on my feet (retail/teaching)",
            "manual": "manual labour",
            "labour": "manual labour",
            "labor": "manual labour",
            "construction": "manual labour",
        }
        for key, norm in _job_keywords.items():
            if key in lower:
                return norm
        return None

    if field in {"diet_preference", "foods_to_avoid", "allergies", "equipment"}:
        cleaned = re.sub(r"\s+", " ", text).strip(" .,!?")
        lower_cleaned = cleaned.lower()
        profile_markers = [
            "name", "age", "male", "female", "height", "weight",
            "goal", "activity", "cm", "kg",
        ]
        if len(cleaned.split()) > 8 and any(
            marker in lower_cleaned for marker in profile_markers
        ):
            return None
        return cleaned or None

    if field == "name":
        if re.search(r"\d", text):
            return None
        cleaned = re.sub(r"\s+", " ", text).strip(" .,!?")
        if not cleaned:
            return None
        words = cleaned.split()
        if len(words) > 4:
            return None
        # Reject command/request phrases — these are not names
        _command_words = {
            "create", "make", "build", "show", "get", "delete", "update",
            "modify", "change", "generate", "give", "plan", "diet", "workout",
            "meal", "exercise", "yes", "no", "confirm", "sync", "help",
        }
        if any(w.lower() in _command_words for w in words):
            return None
        return cleaned

    return None


# ── field list helpers ───────────────────────────────────────────────


def required_fields_for_domain(domain: str) -> list[str]:
    """Return the list of required profile fields for *domain*.

    Falls back to :data:`BASE_PROFILE_FIELDS` when the domain is not
    recognised.

    Args:
        domain: ``"diet"`` or ``"workout"``.

    Returns:
        A new list of field-name strings (safe to mutate).
    """
    return list(DOMAIN_REQUIRED_FIELDS.get(domain, BASE_PROFILE_FIELDS))


def missing_profile_fields(
    profile: dict[str, Any],
    required_fields: list[str],
) -> list[str]:
    """Identify which *required_fields* are absent or empty in *profile*.

    Args:
        profile: The current user-profile dictionary.
        required_fields: Ordered list of field names to check.

    Returns:
        A list of field names that are missing or set to ``None`` / ``""``.
    """
    return [
        field
        for field in required_fields
        if field not in profile or profile[field] in (None, "")
    ]


# ── message builders ─────────────────────────────────────────────────


_FIELD_LABELS: dict[str, str] = {
    "name": "Name",
    "age": "Age",
    "sex": "Sex",
    "height_cm": "Height",
    "weight_kg": "Weight",
    "goal": "Goal",
    "goal_weight": "Goal Weight",
    "weight_loss_pace": "Pace",
    "activity_level": "Activity Level",
    "diet_preference": "Diet",
    "job_type": "Job Type",
    "exercise_frequency": "Exercise Freq",
    "exercise_type": "Exercise Type",
    "sleep_hours": "Sleep",
    "stress_level": "Stress",
    "alcohol_intake": "Alcohol",
    "favourite_meals": "Favourite Meals",
    "foods_to_avoid": "Foods to Avoid",
    "allergies": "Allergies",
    "cooking_style": "Cooking Style",
    "food_adventurousness": "Adventurousness",
    "current_snacks": "Current Snacks",
    "snack_reason": "Snack Reason",
    "snack_preference": "Snack Preference",
    "late_night_snacking": "Late-night Snacking",
    "experience_level": "Experience",
    "training_days_per_week": "Training Days",
    "session_duration": "Session Length",
    "daily_steps": "Daily Steps",
    "additional_info": "Notes",
}

_FIELD_SECTIONS: dict[str, list[str]] = {
    "Body & Goals": [
        "name", "age", "sex", "height_cm", "weight_kg",
        "goal", "goal_weight", "weight_loss_pace", "activity_level",
        "experience_level",
    ],
    "Lifestyle": [
        "job_type", "exercise_frequency", "exercise_type",
        "training_days_per_week", "session_duration", "daily_steps",
        "sleep_hours", "stress_level", "alcohol_intake",
    ],
    "Food Preferences": [
        "diet_preference", "favourite_meals", "foods_to_avoid",
        "allergies", "cooking_style", "food_adventurousness",
    ],
    "Snack Habits": [
        "current_snacks", "snack_reason", "snack_preference",
        "late_night_snacking",
    ],
    "Other": [
        "additional_info",
    ],
}

_FIELD_UNITS: dict[str, str] = {
    "height_cm": "cm",
    "weight_kg": "kg",
    "sleep_hours": "hrs",
}


def _format_value(field: str, value: Any) -> str:
    """Format a profile value for display."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    unit = _FIELD_UNITS.get(field, "")
    return f"{value} {unit}".strip() if unit else str(value)


def build_profile_confirmation(
    profile: dict[str, Any],
    required_fields: list[str],
) -> str:
    """Format a styled confirmation message from *profile*.

    Groups fields into logical sections with a markdown table per
    section. Only fields present in both *profile* and *required_fields*
    are shown.
    """
    present = {
        f for f in required_fields
        if f in profile and profile[f] not in (None, "")
    }

    if not present:
        return (
            "I couldn't map any details yet. "
            "Please share your info so I can build your plan."
        )

    parts: list[str] = ["Here's your profile — please confirm or correct:\n"]

    for section_name, section_fields in _FIELD_SECTIONS.items():
        rows = []
        for f in section_fields:
            if f in present:
                label = _FIELD_LABELS.get(f, f.replace("_", " ").title())
                val = _format_value(f, profile[f])
                rows.append(f"| {label} | {val} |")
        if rows:
            parts.append(f"**{section_name}**\n")
            parts.append("| Field | Value |")
            parts.append("|-------|-------|")
            parts.extend(rows)
            parts.append("")  # blank line

    parts.append(
        "Does this look right? Reply **yes** to confirm, "
        "or tell me what to change."
    )
    return "\n".join(parts)


def build_profile_bulk_question(fields: list[str]) -> str:
    """Format a prompt asking the user to supply all *fields* at once.

    Fields without an entry in :data:`FIELD_QUESTION` are silently
    skipped.

    Args:
        fields: List of profile field names the user still needs to
            provide.

    Returns:
        A formatted multi-line prompt string.
    """
    lines = [FIELD_QUESTION[field] for field in fields if field in FIELD_QUESTION]
    checklist = "\n".join(f"- {line}" for line in lines)
    return (
        "Please share the following details in one message so I can build your plan:\n"
        f"{checklist}\n\n"
        "Example: Name Kushagra, age 28, male, height 183 cm, weight 83 kg, "
        "goal fat loss, activity moderate."
    )


# ── validation ───────────────────────────────────────────────────────


def validate_profile_field(field: str, value: Any) -> tuple[bool, str]:
    """Validate a single profile field value against ``PROFILE_VALIDATION`` ranges.

    Checks whether *value* falls within the acceptable range or set of
    allowed values defined in :data:`PROFILE_VALIDATION`.  Fields that
    do not appear in the validation mapping are assumed valid.

    Args:
        field: The profile field name (e.g. ``"age"``, ``"sex"``).
        value: The value to validate.

    Returns:
        A ``(is_valid, error_message)`` tuple.  When the value is
        acceptable *error_message* is an empty string.
    """
    # ── name-specific validation (no spec needed) ─────────────────
    if field == "name":
        name_str = str(value).strip()
        _bad_words = {
            "create", "make", "build", "show", "get", "delete", "update",
            "modify", "change", "generate", "give", "plan", "diet", "workout",
            "meal", "exercise", "yes", "no", "confirm", "sync", "help",
        }
        if any(w.lower() in _bad_words for w in name_str.split()):
            return False, f"name looks like a command, not a person's name: {value!r}."
        if len(name_str.split()) > 4:
            return False, f"name is too long to be a person's name: {value!r}."
        return True, ""

    spec = PROFILE_VALIDATION.get(field)
    if spec is None:
        # No validation rule defined — accept any value.
        logger.debug("No validation spec for field %r; accepting value.", field)
        return True, ""

    # ── range-based numeric validation ───────────────────────────
    if "min" in spec or "max" in spec:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return False, f"{field} must be a number, got {value!r}."

        low = spec.get("min")
        high = spec.get("max")
        if low is not None and num < low:
            return False, f"{field} must be at least {low}, got {num}."
        if high is not None and num > high:
            return False, f"{field} must be at most {high}, got {num}."
        return True, ""

    # ── allowed-values (enum) validation ─────────────────────────
    if "allowed" in spec:
        allowed: set[str] = spec["allowed"]
        normalised = str(value).strip().lower()
        if normalised not in allowed:
            sorted_allowed = ", ".join(sorted(allowed))
            return (
                False,
                f"{field} must be one of [{sorted_allowed}], got {value!r}.",
            )
        return True, ""

    # Spec exists but has no recognised keys — accept.
    logger.warning(
        "PROFILE_VALIDATION[%r] has unrecognised keys %s; accepting value.",
        field,
        set(spec.keys()),
    )
    return True, ""
