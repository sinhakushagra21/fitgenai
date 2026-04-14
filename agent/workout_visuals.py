"""
agent/workout_visuals.py
────────────────────────
Visual / parsing helpers for workout plan data in FITGEN.AI.

Provides:
  1. Today's workout extraction from plan markdown
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


_DAY_ABBR_MAP = {
    "Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday",
    "Thu": "Thursday", "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday",
}

_DAY_ABBRS = set(_DAY_ABBR_MAP.keys())


def extract_todays_workout(plan_markdown: str) -> dict[str, Any] | None:
    """Extract today's workout session from the Training Schedule table.

    Returns
    -------
    dict with keys ``day_abbr``, ``day_name``, ``session_name``, ``exercises``
    (list of ``{"name": str, "sets_reps": str}``), or ``None`` if today is
    a rest day.
    """
    today_abbr = datetime.now().strftime("%a")  # Mon, Tue, Wed …

    lines = plan_markdown.split("\n")
    in_schedule = False
    found_today = False
    session_name = ""
    exercises: list[dict[str, str]] = []

    for line in lines:
        stripped = line.strip()

        # Detect the start of the Training Schedule section
        if re.search(r"\*?\*?Training\s+Schedule\*?\*?", stripped, re.IGNORECASE):
            in_schedule = True
            continue

        if not in_schedule:
            continue

        # Stop parsing when we leave the table (next section header, blank
        # line after exercises collected, or markdown heading)
        if found_today and (
            stripped.startswith("**") and not stripped.startswith("|")
            or stripped.startswith("#")
        ):
            break

        # Skip non-table lines
        if not stripped.startswith("|"):
            if found_today and exercises:
                break  # blank line after our exercises → done
            continue

        # Skip header / separator rows
        # A separator row contains ONLY pipes, dashes, colons, and whitespace
        if re.fullmatch(r"[|\s:\-]+", stripped):
            continue
        if re.search(r"\|\s*Day\s*\|", stripped, re.IGNORECASE):
            continue

        # Parse table cells — split by pipe and strip whitespace
        parts = [c.strip() for c in stripped.split("|")]
        # parts[0] = '' (before first |), parts[1] = day, parts[2] = session, etc.
        if len(parts) < 5:
            continue

        day_cell = parts[1]
        session_cell = parts[2]
        exercise_cell = parts[3] if len(parts) > 3 else ""
        sets_cell = parts[4] if len(parts) > 4 else ""

        # New day row
        if day_cell in _DAY_ABBRS:
            if found_today and day_cell != today_abbr:
                break  # moved past today's block
            if day_cell == today_abbr:
                found_today = True
                session_name = session_cell or session_name
                # Only add if it's a real exercise (not warm-up/cool-down, has sets)
                if _is_exercise_row(exercise_cell, sets_cell):
                    exercises.append({"name": exercise_cell, "sets_reps": sets_cell})
        elif found_today and not day_cell:
            # Continuation row for the current day
            if session_cell and not session_name:
                session_name = session_cell
            if _is_exercise_row(exercise_cell, sets_cell):
                exercises.append({"name": exercise_cell, "sets_reps": sets_cell})

    if not found_today:
        return None

    return {
        "day_abbr": today_abbr,
        "day_name": _DAY_ABBR_MAP.get(today_abbr, today_abbr),
        "session_name": session_name or "Workout",
        "exercises": exercises,
    }


def _is_exercise_row(exercise: str, sets: str) -> bool:
    """Return True if this row represents an actual exercise (not warm-up/cool-down)."""
    if not exercise:
        return False
    low = exercise.lower()
    if low.startswith("warm") or low.startswith("cool"):
        return False
    if sets in ("—", "-", "–", ""):
        return False
    return True
