"""
agent/tools/diet_tool.py
─────────────────────────
FITGEN.AI Diet & Nutrition Specialist Tool.

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
from agent.prompts.diet_prompts import DIET_PROMPTS

logger = logging.getLogger("fitgen.diet_tool")


def _call_technique(technique: str, query: str) -> tuple[str, str]:
    """Invoke one prompt variant and return (technique_key, response_text)."""
    t0 = time.perf_counter()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    response = llm.invoke(
        [SystemMessage(content=DIET_PROMPTS[technique]), HumanMessage(content=query)]
    )
    elapsed = time.perf_counter() - t0
    logger.info("[DietTool] %-22s → %d chars  (%.2fs)", technique, len(response.content), elapsed)
    return technique, response.content


@tool
def diet_tool(query: str) -> str:
    """Expert nutrition and diet specialist for FITGEN.AI.

    Use this tool for ANY query related to:
    - Calorie targets, TDEE, BMR, or energy balance
    - Macronutrients (protein, carbs, fats) or micronutrients
    - Meal planning, meal prep, or food suggestions
    - Weight loss, fat loss, bulking, or body recomposition through diet
    - Dietary restrictions or preferences (vegan, vegetarian, gluten-free,
      dairy-free, halal, keto, paleo, intermittent fasting, etc.)
    - Pre-workout, post-workout, or peri-workout nutrition
    - Supplements (protein powder, creatine, vitamins, minerals, etc.)
    - Hydration or water intake guidelines
    - Sports nutrition or performance fuelling

    Args:
        query: The user's nutrition or diet question.

    Returns a JSON object with one response per prompt technique
    (zero_shot, few_shot, cot, analogical, generate_knowledge).
    """
    logger.info("[DietTool] Starting 5 parallel technique calls for: %s", query[:80])
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
    logger.info("[DietTool] All 5 techniques done in %.2fs", time.perf_counter() - t_start)
    return json.dumps(ordered)
