"""
agent/tools/youtube_service.py
──────────────────────────────
YouTube video enrichment for FITGEN.AI workout plans.

Extracts exercise names from a generated workout plan, searches YouTube
for tutorial videos, and injects a "Tutorial" column directly into each
exercise schedule table in the plan markdown.

Three-tier graceful degradation:
  1. Redis cache (avoids redundant API calls; 30-day TTL)
  2. YouTube Data API v3 (requires YOUTUBE_API_KEY)
  3. Fallback YouTube search URLs (always works, no API key needed)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import quote_plus

import requests

from agent.cache.redis_client import youtube_cache_get, youtube_cache_set
from agent.config import DEFAULT_MODEL

logger = logging.getLogger("fitgen.youtube")

# ── Config ───────────────────────────────────────────────────────
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
YOUTUBE_CACHE_TTL_DAYS = int(os.getenv("YOUTUBE_CACHE_TTL_DAYS", "30"))
_YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
_MAX_RESULTS_PER_EXERCISE = 1
_MAX_EXERCISES = 50  # Safety ceiling; typical plans have 18-25 exercises
_KNOWN_SHORT_EXERCISES = {
    "plank", "crunch", "curl", "squat", "lunge", "press",
    "dip", "dips", "shrug", "shrugs", "row", "rows",
    "clean", "snatch", "jerk", "thruster", "pullup", "push-up",
}


# ── Redis cache wrappers ─────────────────────────────────────────

def _get_cached_video(exercise: str) -> dict[str, str] | None:
    """Return cached video dict from Redis, or None if missing."""
    return youtube_cache_get(exercise)


def _cache_video(exercise: str, title: str, url: str, channel: str = "") -> None:
    """Store a video in the Redis cache with TTL."""
    youtube_cache_set(exercise, title, url, channel)


# ── Exercise extraction ──────────────────────────────────────────

def extract_exercise_names(plan_text: str) -> list[str]:
    """Extract unique exercise names from a workout plan's markdown tables.

    Strategy: Only extract from tables whose header row contains
    "Exercise" (case-insensitive). This avoids pulling text from
    overview/summary tables that contain goals, equipment, etc.
    Also catches bullet-point exercise lists.
    """
    exercises: list[str] = []
    seen: set[str] = set()

    # ── Pattern 1: Targeted table extraction ─────────────────────
    # Split the plan into table blocks. Only process tables whose
    # header row contains "Exercise".
    lines = plan_text.split("\n")
    in_exercise_table = False
    exercise_col_idx = -1

    for line in lines:
        stripped = line.strip()

        # Detect table header row containing "Exercise"
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]  # skip empty first/last

            # Check if this is a header row with "Exercise"
            if any(re.match(r"(?i)^exercise", c) for c in cells):
                in_exercise_table = True
                exercise_col_idx = next(
                    i for i, c in enumerate(cells) if re.match(r"(?i)^exercise", c)
                )
                continue

            # Separator row (e.g. |---|---|---|)
            if all(re.match(r"^[-:]+$", c) for c in cells if c):
                continue

            # Data row inside an exercise table
            if in_exercise_table and exercise_col_idx < len(cells):
                cell = cells[exercise_col_idx].strip()
                if _is_exercise_name(cell):
                    _add_exercise(cell, exercises, seen)
                continue

        # Non-table line → end of current table
        if in_exercise_table and not stripped.startswith("|"):
            in_exercise_table = False
            exercise_col_idx = -1

    # ── Pattern 2: Bullet / numbered list items ──────────────────
    # e.g. "- Barbell Squat — 3×8" or "1. Deadlift (3 sets)"
    list_pattern = re.compile(
        r"^[\s]*[-*\d.]+\s+(.+?)(?:\s*[—–-]\s*\d|$)", re.MULTILINE
    )
    for match in list_pattern.finditer(plan_text):
        cell = match.group(1).strip()
        if _is_exercise_name(cell):
            _add_exercise(cell, exercises, seen)

    return exercises[:_MAX_EXERCISES]


def _is_exercise_name(text: str) -> bool:
    """Heuristic: returns True if the text looks like an exercise name."""
    if not text or len(text) < 3 or len(text) > 60:
        return False
    # Skip table headers, numbers-only, and common non-exercise strings
    skip_patterns = [
        r"^-+$",                       # Separator lines
        r"^\d+$",                       # Pure numbers
        r"^[\d×x\s.]+$",              # Sets/reps like "3×10"
        r"^\d+\s*(sets?|reps?)$",      # "18 sets", "3 reps"
        r"^\d+s$",                      # Rest durations like "90s", "60s"
        r"^\d+\s*s$",                   # "90 s"
        r"^\d+\s*(min|sec|seconds|minutes)$",  # Time durations
        r"(?i)^(day|rest|sets|reps|tempo|duration|exercise|muscle|notes|time|warm|cool)",
        r"(?i)^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"(?i)^(week|phase|block|session|programme|overview|progression|disclaimer)",
        r"(?i)^(upper|lower|push|pull|legs|back|chest|arms|core|cardio|full)\s*(body|push|pull)?$",
        r"(?i)^(focus|volume|target|split|frequency|intensity|load|deload)",
        r"(?i)^(#|##|###)",            # Markdown headers
        r"(?i)^(total|sum|average|calories|protein|macros|recovery|flexibility)",
    ]
    for pat in skip_patterns:
        if re.match(pat, text):
            return False
    # Must contain at least one letter
    if not re.search(r"[a-zA-Z]", text):
        return False
    # Must contain at least two words or be a known compound exercise form
    # Single generic words are unlikely to be exercises
    words = text.split()
    if len(words) == 1 and len(text) < 8:
        if text.lower() not in _KNOWN_SHORT_EXERCISES:
            return False
    return True


def _add_exercise(raw: str, exercises: list[str], seen: set[str]) -> None:
    """Clean an exercise name and add it if not a duplicate."""
    # Strip trailing sets/reps info: "Bench Press 3×8" → "Bench Press"
    cleaned = re.sub(r"\s*\d+\s*[×x]\s*\d+.*$", "", raw).strip()
    # Strip trailing parenthetical: "Deadlift (barbell)" → keep as is
    # Strip leading emoji
    cleaned = re.sub(r"^[\U0001F300-\U0001FAFF\u2600-\u27BF]+\s*", "", cleaned).strip()
    # Remove bold markdown
    cleaned = cleaned.replace("**", "").strip()
    # Strip trailing parentheticals: "Face Pull (light)" → "Face Pull"
    cleaned = re.sub(r"\s*\(.*?\)\s*$", "", cleaned).strip()

    if not cleaned or cleaned.lower() in seen:
        return
    seen.add(cleaned.lower())
    exercises.append(cleaned)


# ── YouTube API search ───────────────────────────────────────────

def _search_youtube_api(exercise: str) -> dict[str, str] | None:
    """Search YouTube Data API v3 for an exercise tutorial.

    Returns {"title": ..., "url": ..., "channel": ...} or None.
    """
    if not YOUTUBE_API_KEY:
        return None

    params = {
        "part": "snippet",
        "q": f"{exercise} exercise tutorial form",
        "type": "video",
        "maxResults": _MAX_RESULTS_PER_EXERCISE,
        "key": YOUTUBE_API_KEY,
        "relevanceLanguage": "en",
        "videoDuration": "medium",  # 4-20 min — ideal for tutorials
    }

    try:
        resp = requests.get(_YOUTUBE_SEARCH_URL, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        if not items:
            return None

        item = items[0]
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        channel = item["snippet"].get("channelTitle", "")

        return {
            "title": title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "channel": channel,
        }

    except Exception as e:
        logger.warning("[YouTube] API error for '%s': %s", exercise, e)
        return None


def _fallback_search_url(exercise: str) -> dict[str, str]:
    """Generate a YouTube search URL (no API key needed)."""
    query = quote_plus(f"{exercise} exercise tutorial form")
    return {
        "title": f"{exercise} — YouTube Search",
        "url": f"https://www.youtube.com/results?search_query={query}",
        "channel": "",
    }


# ── Single exercise lookup (cache → API → fallback) ─────────────

def _lookup_video(exercise: str) -> tuple[str, dict[str, str]]:
    """Return (exercise_name, video_dict) using 3-tier strategy."""
    # Tier 1: Cache
    cached = _get_cached_video(exercise)
    if cached:
        logger.debug("[YouTube] Cache hit: %s", exercise)
        return exercise, cached

    # Tier 2: API
    result = _search_youtube_api(exercise)
    if result:
        _cache_video(exercise, result["title"], result["url"], result["channel"])
        logger.debug("[YouTube] API hit: %s", exercise)
        return exercise, result

    # Tier 3: Fallback search URL
    fallback = _fallback_search_url(exercise)
    logger.debug("[YouTube] Fallback: %s", exercise)
    return exercise, fallback


# ── Main enrichment function ─────────────────────────────────────

def _build_video_link(exercise: str, videos: dict[str, dict[str, str]]) -> str:
    """Build a markdown link for an exercise, or empty string if not found."""
    # Try exact match first
    video = videos.get(exercise)
    if not video:
        # Try cleaned version (without parentheticals)
        clean = re.sub(r"\s*\(.*?\)\s*$", "", exercise).strip()
        video = videos.get(clean)
    if not video:
        return ""
    label = "▶️ Tutorial"
    return f"[{label}]({video['url']})"


def enrich_plan_with_videos(plan_text: str) -> str:
    """Extract exercises from a workout plan and inject a Tutorial column
    directly into each exercise schedule table.

    This is the main entry point. Call it after `generate_plan()` for
    workout domain plans.

    Parameters
    ----------
    plan_text : str
        The generated workout plan markdown.

    Returns
    -------
    str
        The plan with a "Tutorial" column added to every exercise table.
        If no exercises are found, returns the plan unchanged.
    """
    exercises = extract_exercise_names(plan_text)
    if not exercises:
        logger.info("[YouTube] No exercises extracted; skipping enrichment")
        return plan_text

    logger.info("[YouTube] Enriching plan with videos for %d exercises: %s", len(exercises), exercises)

    # Concurrent lookups (API calls are I/O-bound)
    videos: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=min(len(exercises), 5)) as pool:
        futures = {pool.submit(_lookup_video, ex): ex for ex in exercises}
        for future in as_completed(futures):
            try:
                name, video = future.result()
                videos[name] = video
            except Exception as e:
                logger.warning("[YouTube] Lookup failed for '%s': %s", futures[future], e)

    if not videos:
        return plan_text

    # ── Inject "Tutorial" column into exercise tables ────────────
    lines = plan_text.split("\n")
    result_lines: list[str] = []
    in_exercise_table = False
    exercise_col_idx = -1
    link_count = 0
    _missing: list[str] = []  # exercises that didn't get a link

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.split("|")]
            # cells[0] and cells[-1] are empty strings from split

            inner_cells = cells[1:-1]  # actual cell content

            # Detect header row with "Exercise"
            if any(re.match(r"(?i)^exercise", c) for c in inner_cells):
                in_exercise_table = True
                exercise_col_idx = next(
                    i for i, c in enumerate(inner_cells) if re.match(r"(?i)^exercise", c)
                )
                # Add "Tutorial" header column
                result_lines.append(stripped + " Tutorial |")
                continue

            # Separator row (e.g. |---|---|---|)
            if in_exercise_table and all(re.match(r"^[-:]+$", c) for c in inner_cells if c):
                result_lines.append(stripped + "----------|")
                continue

            # Data row inside an exercise table
            if in_exercise_table and exercise_col_idx < len(inner_cells):
                cell = inner_cells[exercise_col_idx].strip()
                # Clean the exercise name the same way extraction does
                cleaned = re.sub(r"\s*\d+\s*[×x]\s*\d+.*$", "", cell).strip()
                cleaned = re.sub(r"^[\U0001F300-\U0001FAFF\u2600-\u27BF]+\s*", "", cleaned).strip()
                cleaned = cleaned.replace("**", "").strip()
                cleaned = re.sub(r"\s*\(.*?\)\s*$", "", cleaned).strip()

                link = _build_video_link(cleaned, videos)
                if link:
                    link_count += 1
                else:
                    # Track exercises that didn't get a link (skip warm-ups/cool-downs)
                    if _is_exercise_name(cleaned):
                        _missing.append(cleaned)
                result_lines.append(stripped + f" {link} |")
                continue

        # Non-table line → end of current exercise table
        if in_exercise_table and not stripped.startswith("|"):
            in_exercise_table = False
            exercise_col_idx = -1

        result_lines.append(line)

    logger.info("[YouTube] Injected %d tutorial links into exercise tables", link_count)
    if _missing:
        logger.warning("[YouTube] %d exercise(s) missing tutorial links: %s", len(_missing), _missing)
    else:
        logger.info("[YouTube] All exercise rows have tutorial links ✓")
    return "\n".join(result_lines)
