"""
agent/config.py
───────────────
Centralized configuration constants for FITGEN.AI.

Multi-model architecture:
  PLAN_MODEL       — powerful model for plan generation (expensive)
  FAST_MODEL       — lightweight model for routing, classification, validation
  DEFAULT_MODEL    — alias for PLAN_MODEL (backward compat)

Model selection is actually performed by
``agent.shared.llm_helpers.resolve_model(purpose, domain)`` which honours the
following env vars (highest priority first):

  Per-domain overrides (plug a fine-tuned model into just one tool):
    FITGEN_DIET_PLAN_MODEL       FITGEN_WORKOUT_PLAN_MODEL
    FITGEN_DIET_FAST_MODEL       FITGEN_WORKOUT_FAST_MODEL
    FITGEN_DIET_QA_MODEL         FITGEN_WORKOUT_QA_MODEL

  Global overrides (apply to both diet and workout):
    FITGEN_PLAN_MODEL            FITGEN_FAST_MODEL            FITGEN_QA_MODEL

  Registry: fine_tuning/data/finetuned_model_ids.json is consulted next.

  Hardcoded defaults: the constants below.

Leave the env vars unset to use the defaults. To roll out a fine-tuned diet
plan model without touching tool code::

    export FITGEN_DIET_PLAN_MODEL=ft:gpt-4.1-mini:acme:fitgen-diet-plan:abc123
"""

from __future__ import annotations

import os

# ── Model tiers ─────────────────────────────────────────────────
PLAN_MODEL = os.getenv("FITGEN_PLAN_MODEL", "gpt-5.1")
FAST_MODEL = os.getenv("FITGEN_FAST_MODEL", "gpt-4.1-mini")

# Backward compatibility alias
DEFAULT_MODEL = PLAN_MODEL
