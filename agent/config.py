"""
agent/config.py
───────────────
Centralized configuration constants for FITGEN.AI.

Multi-model architecture:
  PLAN_MODEL       — powerful model for plan generation (expensive)
  FAST_MODEL       — lightweight model for routing, classification, validation
  DEFAULT_MODEL    — alias for PLAN_MODEL (backward compat)
"""

from __future__ import annotations

import os

# ── Model tiers ─────────────────────────────────────────────────
PLAN_MODEL = os.getenv("FITGEN_PLAN_MODEL", "gpt-5.1")
FAST_MODEL = os.getenv("FITGEN_FAST_MODEL", "gpt-4.1-mini")

# Backward compatibility alias
DEFAULT_MODEL = PLAN_MODEL
