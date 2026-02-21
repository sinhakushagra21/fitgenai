"""
agent/tools/workout_tool.py
────────────────────────────
FITGEN.AI Workout Specialist Tool.

Runs the user's query through all 5 prompt technique variants in parallel
and returns a JSON-encoded dict keyed by technique name:
  {"zero_shot": "...", "few_shot": "...", "cot": "...", ...}

The Streamlit frontend renders each response in a separate tab.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agent.prompts.techniques import TECHNIQUE_KEYS
from agent.prompts.workout_prompts import WORKOUT_PROMPTS

logger = logging.getLogger("fitgen.workout_tool")


def _call_technique(technique: str, query: str) -> tuple[str, str]:
    """Invoke one prompt variant and return (technique_key, response_text)."""
    t0 = time.perf_counter()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    response = llm.invoke(
        [SystemMessage(content=WORKOUT_PROMPTS[technique]), HumanMessage(content=query)]
    )
    elapsed = time.perf_counter() - t0
    logger.info("[WorkoutTool] %-22s → %d chars  (%.2fs)", technique, len(response.content), elapsed)
    return technique, response.content


@tool
def workout_tool(query: str) -> str:
    """Expert workout and training specialist for FITGEN.AI.

    Use this tool for ANY query related to:
    - Exercise selection, technique, or form
    - Workout plans, training programmes, or splits (PPL, Upper/Lower, Full Body, etc.)
    - Sets, reps, rest periods, tempo, or progressive overload
    - Cardio, HIIT, endurance, or sport-specific training
    - Mobility, flexibility, stretching, or warm-up/cool-down
    - Recovery from training, deload weeks, or overtraining
    - Strength, hypertrophy, or athletic performance goals

    Returns a JSON object with one response per prompt technique
    (zero_shot, few_shot, cot, analogical, generate_knowledge).

    Args:
        query: The user's workout or training question.
    """
    logger.info("[WorkoutTool] Starting 5 parallel technique calls for: %s", query[:80])
    t_start = time.perf_counter()
    results: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_call_technique, technique, query): technique
            for technique in TECHNIQUE_KEYS
        }
        for future in as_completed(futures):
            technique, response = future.result()
            results[technique] = response

    ordered = {k: results[k] for k in TECHNIQUE_KEYS if k in results}
    logger.info("[WorkoutTool] All 5 techniques done in %.2fs", time.perf_counter() - t_start)
    return json.dumps(ordered)
