"""
agent/prompts/__init__.py
─────────────────────────
Public API for the FITGEN.AI prompts package.
"""

from agent.prompts.base_prompts import BASE_PROMPTS
from agent.prompts.diet_prompts import DIET_PROMPTS
from agent.prompts.techniques import TECHNIQUE_KEYS, TECHNIQUE_META
from agent.prompts.workout_prompts import WORKOUT_PROMPTS

__all__ = [
    "TECHNIQUE_KEYS",
    "TECHNIQUE_META",
    "BASE_PROMPTS",
    "WORKOUT_PROMPTS",
    "DIET_PROMPTS",
]
