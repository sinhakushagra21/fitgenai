"""
agent/tools/google_fit_integration.py
──────────────────────────────────────
Google Fit integration for FITGEN.AI.

Provides:
  1.  LLM-powered plan → structured nutrition / activity extractor
  2.  Google Fit REST API v1 data pushers (nutrition + activity sessions)

Reuses OAuth tokens from calendar_integration.py (shared scopes).
"""

from __future__ import annotations

import calendar
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.config import FAST_MODEL
from agent.tracing import trace, get_langsmith_config

logger = logging.getLogger("fitgen.google_fit")

_LLM_MODEL = FAST_MODEL

# Google Fit API scopes (must match calendar_integration._SCOPES).
_FIT_SCOPES = [
    "https://www.googleapis.com/auth/fitness.nutrition.write",
    "https://www.googleapis.com/auth/fitness.activity.write",
]

# ── Activity type mapping ───────────────────────────────────────
# Maps common exercise names to Google Fit integer activity type codes.
# Full reference: https://developers.google.com/fit/rest/v1/reference/activity-list
_ACTIVITY_TYPE_MAP: dict[str, int] = {
    "walking": 7,
    "running": 8,
    "jogging": 8,
    "cycling": 1,
    "biking": 1,
    "swimming": 82,
    "yoga": 100,
    "pilates": 101,
    "weight_training": 80,
    "weights": 80,
    "strength_training": 80,
    "resistance_training": 80,
    "hiit": 113,
    "crossfit": 113,
    "cardio": 9,
    "elliptical": 25,
    "rowing": 103,
    "jump_rope": 55,
    "stretching": 75,
    "dance": 22,
    "boxing": 87,
    "kickboxing": 87,
    "martial_arts": 52,
    "stair_climbing": 68,
    "hiking": 37,
    "rock_climbing": 17,
    "skiing": 67,
    "basketball": 14,
    "football": 28,
    "soccer": 28,
    "tennis": 87,
    "badminton": 10,
    "cricket": 21,
    "rest": 72,
    "other": 108,
}

# Meal type codes for Google Fit nutrition data.
_MEAL_TYPE_MAP: dict[str, int] = {
    "breakfast": 2,
    "lunch": 3,
    "dinner": 4,
    "snack": 5,
    "snacks": 5,
    "unknown": 1,
}


# ── LLM-powered extractors ─────────────────────────────────────

_NUTRITION_EXTRACTION_PROMPT = """\
You are a structured-data extractor for FITGEN.AI.

Given a diet plan and the user's profile, extract a list of daily nutrition entries.
For a 7-day plan, extract EACH unique meal across the week (not repeated entries).

Rules:
- Extract one entry per UNIQUE meal (Breakfast, Lunch, Dinner, Snack).
- Include the day of the week for each entry.
- Estimate realistic macros based on the meal description.
- If exact macros are provided in the plan, use those values.
- All numeric values should be reasonable for a single meal.

Return ONLY a JSON array of objects with these keys:
  day: string ("Monday", "Tuesday", etc.)
  meal_type: string ("breakfast" | "lunch" | "dinner" | "snack")
  food_name: string (brief description of the meal, e.g. "Grilled chicken with quinoa")
  calories_kcal: float
  protein_g: float
  carbs_g: float
  fat_g: float

User profile:
{profile}

Diet plan:
{plan_text}
"""

_ACTIVITY_EXTRACTION_PROMPT = """\
You are a structured-data extractor for FITGEN.AI.

Given a workout plan and the user's profile, extract a list of exercise sessions.

Rules:
- Extract one entry per unique workout session in the plan.
- Include the day of the week.
- Estimate duration and calories burned realistically based on the exercises listed.
- Map each session to the closest activity type from this list:
  walking, running, cycling, swimming, yoga, pilates, weight_training,
  strength_training, hiit, crossfit, cardio, elliptical, rowing, stretching,
  dance, boxing, martial_arts, hiking, basketball, football, tennis, cricket, other

Return ONLY a JSON array of objects with these keys:
  day: string ("Monday", "Tuesday", etc.)
  session_name: string (e.g. "Chest & Triceps", "HIIT Cardio")
  activity_type: string (from the list above)
  duration_minutes: int
  estimated_calories_burned: float

User profile:
{profile}

Workout plan:
{plan_text}
"""


@trace(name="Extract Nutrition Data", run_type="chain", tags=["google_fit", "nutrition"])
def extract_nutrition_data(
    plan_text: str,
    domain: str,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Use the LLM to extract structured nutrition entries from a diet plan."""
    logger.info("[GoogleFit] extract_nutrition_data called: plan_len=%d, domain=%s, profile_keys=%s",
                len(plan_text), domain, sorted(profile.keys()) if profile else "empty")

    if not plan_text or len(plan_text) < 50:
        logger.error("[GoogleFit] Plan text too short (%d chars) — cannot extract", len(plan_text))
        return []

    try:
        llm = ChatOpenAI(model=_LLM_MODEL, temperature=0)
        prompt = _NUTRITION_EXTRACTION_PROMPT.format(
            profile=json.dumps(profile, indent=2),
            plan_text=plan_text,
        )
        logger.info("[GoogleFit] Sending extraction prompt to LLM (%d chars)...", len(prompt))
        resp = llm.invoke(
            [
                SystemMessage(content="You are a JSON extractor. Return ONLY valid JSON."),
                HumanMessage(content=prompt),
            ],
            config=get_langsmith_config("Google Fit Nutrition Extraction", tags=["google_fit"]),
        )
        text = resp.content.strip()
        logger.info("[GoogleFit] LLM response received: %d chars", len(text))
    except Exception as e:
        logger.error("[GoogleFit] LLM extraction call failed: %s", e, exc_info=True)
        return []

    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        entries = json.loads(text)
    except json.JSONDecodeError:
        logger.error("[GoogleFit] Failed to parse nutrition JSON: %s", text[:500])
        return []

    if not isinstance(entries, list):
        logger.error("[GoogleFit] Expected list, got %s", type(entries).__name__)
        return []

    logger.info("[GoogleFit] Extracted %d nutrition entries from diet plan", len(entries))
    return entries


@trace(name="Extract Activity Sessions", run_type="chain", tags=["google_fit", "activity"])
def extract_activity_sessions(
    plan_text: str,
    domain: str,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Use the LLM to extract structured activity sessions from a workout plan."""
    llm = ChatOpenAI(model=_LLM_MODEL, temperature=0)
    prompt = _ACTIVITY_EXTRACTION_PROMPT.format(
        profile=json.dumps(profile, indent=2),
        plan_text=plan_text,
    )
    resp = llm.invoke(
        [
            SystemMessage(content="You are a JSON extractor. Return ONLY valid JSON."),
            HumanMessage(content=prompt),
        ],
        config=get_langsmith_config("Google Fit Activity Extraction", tags=["google_fit"]),
    )
    text = resp.content.strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        sessions = json.loads(text)
    except json.JSONDecodeError:
        logger.error("[GoogleFit] Failed to parse activity JSON: %s", text[:200])
        return []

    if not isinstance(sessions, list):
        logger.error("[GoogleFit] Expected list, got %s", type(sessions).__name__)
        return []

    logger.info("[GoogleFit] Extracted %d activity sessions from workout plan", len(sessions))
    return sessions


# ── Google Fit API pushers ──────────────────────────────────────

def _get_fit_service(tokens: dict[str, Any]):
    """Build a Google Fit API v1 service from OAuth tokens."""
    creds = Credentials(
        token=tokens["token"],
        refresh_token=tokens.get("refresh_token"),
        token_uri=tokens.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=tokens.get("client_id"),
        client_secret=tokens.get("client_secret"),
        scopes=tokens.get("scopes", _FIT_SCOPES),
    )
    return build("fitness", "v1", credentials=creds)


def _day_offset(day_name: str) -> int:
    """Return days until next occurrence of the given day name (starting tomorrow)."""
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    try:
        target = days.index(day_name.lower())
    except ValueError:
        return 1  # Default to tomorrow
    today = datetime.now().weekday()
    diff = (target - today) % 7
    return diff if diff > 0 else 7


def _time_nanos(dt: datetime) -> int:
    """Convert a datetime to nanoseconds since epoch — **fully float-free**.

    Uses ``calendar.timegm`` which works on integer time-tuples and
    returns an integer, completely avoiding IEEE-754 precision drift
    that ``datetime.timestamp()`` can introduce (e.g. trailing :002).
    """
    return calendar.timegm(dt.timetuple()) * 1_000_000_000


def _time_millis(dt: datetime) -> int:
    """Convert a datetime to milliseconds since epoch — float-free."""
    return calendar.timegm(dt.timetuple()) * 1_000


# Cached data source ID — populated after first successful creation/lookup.
_cached_nutrition_ds_id: str | None = None


def _ensure_nutrition_data_source(service) -> str:
    """Create or find the FITGEN.AI nutrition data source.

    Google Fit auto-generates the dataStreamId — we must NOT set it manually.
    For standard data types (com.google.nutrition), do NOT specify field
    definitions; Google already knows the schema.
    """
    global _cached_nutrition_ds_id
    if _cached_nutrition_ds_id:
        return _cached_nutrition_ds_id

    # Step 1: Search existing data sources for one we created earlier.
    try:
        result = service.users().dataSources().list(userId="me").execute()
        for ds in result.get("dataSource", []):
            ds_type = ds.get("dataType", {}).get("name", "")
            app_name = ds.get("application", {}).get("name", "")
            if ds_type == "com.google.nutrition" and "FITGEN" in app_name.upper():
                _cached_nutrition_ds_id = ds["dataStreamId"]
                logger.info("[GoogleFit] Found existing nutrition data source: %s", _cached_nutrition_ds_id)
                return _cached_nutrition_ds_id
    except Exception as e:
        logger.warning("[GoogleFit] Failed to list data sources: %s", e)

    # Step 2: Create a new data source (let Google generate the ID).
    body = {
        "dataStreamName": "FitgenNutrition",
        "type": "raw",
        "application": {
            "name": "FITGEN.AI",
            "version": "1",
        },
        "dataType": {
            "name": "com.google.nutrition",
        },
    }
    try:
        created = service.users().dataSources().create(
            userId="me", body=body,
        ).execute()
        _cached_nutrition_ds_id = created["dataStreamId"]
        logger.info("[GoogleFit] Created nutrition data source: %s", _cached_nutrition_ds_id)
        return _cached_nutrition_ds_id
    except Exception as e:
        logger.error("[GoogleFit] Could not create data source: %s", e, exc_info=True)
        raise RuntimeError(f"Failed to create Google Fit nutrition data source: {e}") from e


@trace(name="Push Nutrition to Google Fit", run_type="chain", tags=["google_fit", "push"])
def push_nutrition_to_google_fit(
    nutrition_data: list[dict[str, Any]],
    tokens: dict[str, Any],
) -> tuple[int, list[str]]:
    """Push extracted nutrition entries to Google Fit.

    Returns (pushed_count, list_of_error_messages).
    """
    errors: list[str] = []

    try:
        service = _get_fit_service(tokens)
    except Exception as e:
        msg = f"Failed to connect to Google Fit: {e}"
        logger.error("[GoogleFit] %s", msg)
        return 0, [msg]

    try:
        data_source_id = _ensure_nutrition_data_source(service)
    except Exception as e:
        return 0, [str(e)]

    logger.info("[GoogleFit] Using data source: %s", data_source_id)
    logger.info("[GoogleFit] Pushing %d nutrition entries...", len(nutrition_data))
    pushed = 0

    # Push each entry individually so one bad data point doesn't kill
    # all the others.  Each PATCH uses a tight dataset window that
    # exactly spans the single data point.
    anchor = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Track used timestamps to add small offsets for duplicates
    _used_timestamps: set[int] = set()

    for idx, entry in enumerate(nutrition_data):
        try:
            day = entry.get("day", "Monday")
            offset = _day_offset(day)
            meal_type_str = entry.get("meal_type", "unknown").lower()
            meal_type_code = _MEAL_TYPE_MAP.get(meal_type_str, 1)

            # Determine meal time based on type.
            meal_times = {"breakfast": 8, "lunch": 13, "dinner": 19, "snack": 16, "snacks": 16}
            hour = meal_times.get(meal_type_str, 12)

            meal_start = anchor + timedelta(days=offset, hours=hour)
            meal_end = meal_start + timedelta(minutes=30)

            start_ns = _time_nanos(meal_start)
            end_ns = _time_nanos(meal_end)

            # Avoid duplicate timestamps — shift by 1 second if collision
            while start_ns in _used_timestamps:
                start_ns += 1_000_000_000   # +1 second (clean nanos)
                end_ns += 1_000_000_000
            _used_timestamps.add(start_ns)

            nutrients = {}
            if entry.get("calories_kcal"):
                nutrients["calories"] = float(entry["calories_kcal"])
            if entry.get("protein_g"):
                nutrients["protein"] = float(entry["protein_g"])
            if entry.get("carbs_g"):
                nutrients["carbohydrates.total"] = float(entry["carbs_g"])
            if entry.get("fat_g"):
                nutrients["fat.total"] = float(entry["fat_g"])

            data_point = {
                "dataTypeName": "com.google.nutrition",
                "startTimeNanos": str(start_ns),
                "endTimeNanos": str(end_ns),
                "value": [
                    {"mapVal": [{"key": k, "value": {"fpVal": v}} for k, v in nutrients.items()]},
                    {"intVal": meal_type_code},
                    {"stringVal": entry.get("food_name", "Meal")},
                ],
            }

            # Individual PATCH — tight window around this single point.
            dataset_id = f"{start_ns}-{end_ns}"
            service.users().dataSources().datasets().patch(
                userId="me",
                dataSourceId=data_source_id,
                datasetId=dataset_id,
                body={
                    "dataSourceId": data_source_id,
                    "minStartTimeNs": str(start_ns),
                    "maxEndTimeNs": str(end_ns),
                    "point": [data_point],
                },
            ).execute()
            pushed += 1
            logger.debug("[GoogleFit] Pushed %d/%d: %s (%s)",
                         pushed, len(nutrition_data),
                         entry.get("food_name"), day)

        except Exception as e:
            err = f"{entry.get('food_name', '?')} ({entry.get('day', '?')}): {e}"
            logger.error("[GoogleFit] Failed to push nutrition — %s", err)
            errors.append(err)

    logger.info("[GoogleFit] Successfully pushed %d/%d nutrition entries", pushed, len(nutrition_data))
    return pushed, errors


@trace(name="Push Activities to Google Fit", run_type="chain", tags=["google_fit", "push"])
def push_activities_to_google_fit(
    sessions: list[dict[str, Any]],
    tokens: dict[str, Any],
) -> tuple[int, list[str]]:
    """Push extracted activity sessions to Google Fit.

    Returns (pushed_count, list_of_error_messages).
    """
    errors: list[str] = []
    try:
        service = _get_fit_service(tokens)
    except Exception as e:
        msg = f"Failed to connect to Google Fit: {e}"
        logger.error("[GoogleFit] %s", msg)
        return 0, [msg]

    pushed = 0

    for session in sessions:
        try:
            day = session.get("day", "Monday")
            offset = _day_offset(day)
            duration = session.get("duration_minutes", 60)

            anchor = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            session_start = anchor + timedelta(days=offset, hours=7)  # Default 7 AM
            session_end = session_start + timedelta(minutes=duration)

            activity_name = session.get("activity_type", "other").lower().replace(" ", "_")
            activity_type = _ACTIVITY_TYPE_MAP.get(activity_name, 108)  # 108 = other

            session_name = session.get("session_name", f"FITGEN.AI Workout ({day})")
            # Create unique session ID.
            session_id = f"fitgen-{day.lower()}-{int(time.time())}-{pushed}"

            session_body = {
                "id": session_id,
                "name": session_name,
                "description": f"FITGEN.AI generated workout: {session_name}",
                "startTimeMillis": str(_time_millis(session_start)),
                "endTimeMillis": str(_time_millis(session_end)),
                "activeTimeMillis": str(duration * 60 * 1000),
                "activityType": activity_type,
                "application": {
                    "name": "FITGEN.AI",
                    "version": "1",
                },
            }

            service.users().sessions().update(
                userId="me",
                sessionId=session_id,
                body=session_body,
            ).execute()

            pushed += 1
            logger.info("[GoogleFit] Pushed session: %s (%s, %d min)", session_name, day, duration)

        except Exception as e:
            err = f"{session.get('session_name', '?')} ({session.get('day', '?')}): {e}"
            logger.error("[GoogleFit] Failed to push session — %s", err)
            errors.append(err)

    logger.info("[GoogleFit] Successfully pushed %d/%d activity sessions", pushed, len(sessions))
    return pushed, errors
