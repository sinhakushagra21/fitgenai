"""
agent/shared/plan_data.py
─────────────────────────
Extract and strip the structured FITGEN_DATA JSON block that the LLM
appends at the end of every generated plan.

The LLM is instructed to include:
    <!-- FITGEN_DATA
    {"macros": {...}, "hydration": {...}}
    -->

This module:
  1. Extracts the JSON into a Python dict.
  2. Returns the plan markdown with the block stripped (clean for display).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("fitgen.plan_data")

_FITGEN_DATA_RE = re.compile(
    r"<!--\s*FITGEN_DATA\s*\n(.*?)\n\s*-->",
    re.DOTALL,
)


def extract_plan_structured_data(
    plan_text: str,
) -> tuple[str, dict[str, Any]]:
    """Parse the hidden FITGEN_DATA block from a plan.

    Returns
    -------
    (clean_markdown, structured_data)
        *clean_markdown* — plan text with the data block removed.
        *structured_data* — parsed JSON dict, or ``{}`` if not found / invalid.
    """
    match = _FITGEN_DATA_RE.search(plan_text)
    if not match:
        logger.debug("No FITGEN_DATA block found in plan text")
        return plan_text, {}

    raw_json = match.group(1).strip()
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("FITGEN_DATA JSON parse failed: %s — raw=%s", exc, raw_json[:200])
        return plan_text, {}

    # Strip the block (and any trailing blank lines it leaves)
    clean = _FITGEN_DATA_RE.sub("", plan_text).rstrip() + "\n"
    logger.info("Extracted FITGEN_DATA: %s", list(data.keys()))
    return clean, data
