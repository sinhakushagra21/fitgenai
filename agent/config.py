"""
agent/config.py
───────────────
Centralized configuration constants for FITGEN.AI.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = os.getenv("FITGEN_LLM_MODEL", "gpt-4o-mini")
