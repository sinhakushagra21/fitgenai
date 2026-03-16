"""
agent/tools/calendar_integration.py
─────────────────────────────────────
Google Calendar integration for FITGEN.AI.

Provides:
  1.  OAuth 2.0 helpers (browser redirect flow)
  2.  LLM-powered plan → structured events extractor
  3.  Google Calendar API event pusher
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger("fitgen.calendar")

_LLM_MODEL = os.getenv("FITGEN_LLM_MODEL", "gpt-4o-mini")
_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501")


# ── OAuth 2.0 helpers ────────────────────────────────────────────


def _get_client_config() -> dict[str, Any]:
    """Build the OAuth client config dict from environment variables."""
    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise ValueError(
            "Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET in .env. "
            "Create OAuth 2.0 credentials at https://console.cloud.google.com"
        )
    return {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [_REDIRECT_URI],
        }
    }


def get_authorization_url() -> tuple[str, str]:
    """Generate the Google OAuth consent URL.

    Returns
    -------
    (authorization_url, state)
        The URL to redirect the user to, and the CSRF state token.
    """
    flow = Flow.from_client_config(
        _get_client_config(),
        scopes=_SCOPES,
        redirect_uri=_REDIRECT_URI,
    )
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return authorization_url, state


def exchange_code_for_tokens(code: str) -> dict[str, Any]:
    """Exchange the authorization code for access + refresh tokens.

    Parameters
    ----------
    code : str
        The authorization code from Google's redirect.

    Returns
    -------
    dict
        Serializable token dictionary with access_token, refresh_token, etc.
    """
    flow = Flow.from_client_config(
        _get_client_config(),
        scopes=_SCOPES,
        redirect_uri=_REDIRECT_URI,
    )
    flow.fetch_token(code=code)
    creds = flow.credentials
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or []),
    }


# ── LLM-powered plan → events extractor ─────────────────────────


_EVENT_EXTRACTION_PROMPT = """\
You are a structured-data extractor for FITGEN.AI.

Given a {domain} plan and the user's profile, extract a list of calendar events.

Rules:
- For DIET plans: extract meals (Breakfast, Lunch, Dinner, Snacks) as daily
  recurring events with reasonable default times (Breakfast 08:00, Lunch 13:00,
  Snacks 16:00, Dinner 19:00). Duration 30 min each.
- For WORKOUT plans: extract training sessions as weekly recurring events on
  the user's workout days. Duration 60 min each. Default start time 07:00.
- Each event summary should start with an emoji prefix:
  Diet → 🍎  Workout → 💪
- Keep descriptions concise (max 2 sentences summarising what to eat/do).

Return ONLY a JSON array of event objects with these keys:
  summary: string (e.g. "🍎 Breakfast — Oats with banana")
  description: string (brief details)
  start_time: string (HH:MM, 24-hour format)
  duration_minutes: int
  recurrence: "DAILY" or "WEEKLY"
  days_of_week: list of 2-letter day codes ["MO","TU","WE","TH","FR","SA","SU"]
                (only required for WEEKLY recurrence)

User profile:
{profile}

Plan:
{plan_text}
"""


def extract_calendar_events(
    plan_text: str,
    domain: str,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Use the LLM to extract structured calendar events from a plan.

    Returns a list of event dicts ready for `push_events_to_calendar`.
    """
    llm = ChatOpenAI(model=_LLM_MODEL, temperature=0)
    prompt = _EVENT_EXTRACTION_PROMPT.format(
        domain=domain,
        profile=json.dumps(profile, indent=2),
        plan_text=plan_text,
    )
    resp = llm.invoke([
        SystemMessage(content="You are a JSON extractor. Return ONLY valid JSON."),
        HumanMessage(content=prompt),
    ])
    text = resp.content.strip()

    # Parse JSON — handle markdown code fences if present.
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        events = json.loads(text)
    except json.JSONDecodeError:
        logger.error("[Calendar] Failed to parse events JSON: %s", text[:200])
        return []

    if not isinstance(events, list):
        logger.error("[Calendar] Expected list, got %s", type(events).__name__)
        return []

    logger.info("[Calendar] Extracted %d events from %s plan", len(events), domain)
    return events


# ── Google Calendar API event pusher ─────────────────────────────


def _build_rrule(event: dict[str, Any]) -> list[str]:
    """Build an RFC 5545 RRULE from an event dict."""
    recurrence = event.get("recurrence", "DAILY").upper()
    if recurrence == "WEEKLY":
        days = event.get("days_of_week", ["MO", "WE", "FR"])
        byday = ",".join(days)
        return [f"RRULE:FREQ=WEEKLY;BYDAY={byday}"]
    return ["RRULE:FREQ=DAILY"]


def push_events_to_calendar(
    events: list[dict[str, Any]],
    tokens: dict[str, Any],
) -> int:
    """Push extracted events to Google Calendar.

    Parameters
    ----------
    events : list[dict]
        Events from `extract_calendar_events`.
    tokens : dict
        Token dictionary from `exchange_code_for_tokens`.

    Returns
    -------
    int
        Number of events successfully created.
    """
    creds = Credentials(
        token=tokens["token"],
        refresh_token=tokens.get("refresh_token"),
        token_uri=tokens.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=tokens.get("client_id"),
        client_secret=tokens.get("client_secret"),
        scopes=tokens.get("scopes", _SCOPES),
    )

    service = build("calendar", "v3", credentials=creds)

    # Anchor events starting tomorrow.
    anchor = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    created = 0

    for ev in events:
        try:
            start_time = ev.get("start_time", "08:00")
            duration = ev.get("duration_minutes", 30)
            hour, minute = map(int, start_time.split(":"))

            event_start = anchor.replace(hour=hour, minute=minute)
            event_end = event_start + timedelta(minutes=duration)

            body = {
                "summary": ev.get("summary", "FITGEN.AI Event"),
                "description": ev.get("description", ""),
                "start": {
                    "dateTime": event_start.isoformat(),
                    "timeZone": "America/New_York",
                },
                "end": {
                    "dateTime": event_end.isoformat(),
                    "timeZone": "America/New_York",
                },
                "recurrence": _build_rrule(ev),
                "reminders": {
                    "useDefault": False,
                    "overrides": [
                        {"method": "popup", "minutes": 10},
                    ],
                },
            }

            service.events().insert(calendarId="primary", body=body).execute()
            created += 1
            logger.info("[Calendar] Created event: %s", ev.get("summary"))

        except Exception as e:
            logger.error("[Calendar] Failed to create event %s: %s", ev.get("summary"), e)

    logger.info("[Calendar] Successfully created %d/%d events", created, len(events))
    return created
