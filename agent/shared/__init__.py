"""
agent/shared/__init__.py
────────────────────────
Shared utilities for FITGEN.AI domain tools (diet, workout).

Re-exports public APIs from submodules for convenient access.
"""

from __future__ import annotations

from agent.shared.types import (  # noqa: F401
    BASE_PROFILE_FIELDS,
    DIET_ALL_FIELDS,
    DIET_OPTIONAL_FIELDS,
    DIET_PROFILE_FIELDS,
    DIET_REQUIRED_FIELDS,
    DOMAIN_REQUIRED_FIELDS,
    FIELD_QUESTION,
    GOAL_KEYWORDS,
    ACTIVITY_KEYWORDS,
    PROFILE_VALIDATION,
    SEX_MAP,
    WORKOUT_ALL_FIELDS,
    WORKOUT_OPTIONAL_FIELDS,
    WORKOUT_REQUIRED_FIELDS,
)

from agent.shared.profile_utils import (  # noqa: F401
    build_profile_bulk_question,
    build_profile_confirmation,
    missing_profile_fields,
    required_fields_for_domain,
    validate_profile_field,
)

from agent.shared.response_builder import (  # noqa: F401
    append_completed_step,
    build_response,
)

__all__ = [
    # types
    "BASE_PROFILE_FIELDS",
    "DIET_ALL_FIELDS",
    "DIET_OPTIONAL_FIELDS",
    "DIET_PROFILE_FIELDS",
    "DIET_REQUIRED_FIELDS",
    "DOMAIN_REQUIRED_FIELDS",
    "FIELD_QUESTION",
    "GOAL_KEYWORDS",
    "ACTIVITY_KEYWORDS",
    "PROFILE_VALIDATION",
    "SEX_MAP",
    "WORKOUT_ALL_FIELDS",
    "WORKOUT_OPTIONAL_FIELDS",
    "WORKOUT_REQUIRED_FIELDS",
    # profile_utils
    "build_profile_bulk_question",
    "build_profile_confirmation",
    "missing_profile_fields",
    "required_fields_for_domain",
    "validate_profile_field",
    # response_builder
    "append_completed_step",
    "build_response",
]
