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

# ── oauthlib scope-relaxation (MUST be set before importing oauthlib) ─
# When the user is already signed in with Google, the OAuth response
# includes their login scopes (openid / email / profile) in addition to
# the scopes we requested. oauthlib's strict check raises a Warning on
# any scope mismatch, which breaks the token exchange. Google's own
# docs recommend relaxing this check; we also request those scopes
# explicitly below so they match most of the time anyway.
import os as _os  # noqa: E402
_os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")
_os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger("fitgen.calendar")

from agent.config import FAST_MODEL

_LLM_MODEL = FAST_MODEL
# NOTE: openid + userinfo.email + userinfo.profile are included so the
# scopes returned by Google (which always includes these when the user
# is signed in) match what we asked for. Without this, oauthlib raises
# "Scope has changed" on token exchange.
_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/fitness.nutrition.write",
    "https://www.googleapis.com/auth/fitness.activity.write",
]
_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501")

# ── Plan data persistence for OAuth redirect ────────────────────
# Streamlit session state is LOST when the browser navigates away to
# Google OAuth and back.  We persist plan data (plan_text, domain,
# profile) to a JSON file so the callback can extract calendar events
# even after the session state is wiped.
#
# PKCE is DISABLED (autogenerate_code_verifier=False) so we don't need
# to persist a code_verifier at all — eliminating the main source of bugs.
_OAUTH_CONTEXT_FILE = Path(tempfile.gettempdir()) / "fitgen_oauth_context.json"


def save_oauth_context(
    plan_text: str = "",
    domain: str = "diet",
    profile: dict[str, Any] | None = None,
    sync_target: str = "calendar",
) -> None:
    """Persist plan data to a temp file before the OAuth redirect.

    sync_target: "calendar", "google_fit", or "both"
    """
    data = {
        "plan_text": plan_text,
        "domain": domain,
        "profile": profile or {},
        "sync_target": sync_target,
    }
    _OAUTH_CONTEXT_FILE.write_text(json.dumps(data), encoding="utf-8")
    logger.info("[Calendar] Saved plan context to %s", _OAUTH_CONTEXT_FILE)


def load_oauth_context() -> dict[str, Any]:
    """Load plan data from temp file (does NOT delete — idempotent reads)."""
    if _OAUTH_CONTEXT_FILE.exists():
        try:
            data = json.loads(_OAUTH_CONTEXT_FILE.read_text(encoding="utf-8"))
            logger.info("[Calendar] Loaded plan context: plan_len=%d, domain=%s",
                        len(data.get("plan_text", "")), data.get("domain", "?"))
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.error("[Calendar] Failed to load plan context: %s", e)
    return {}


def clear_oauth_context() -> None:
    """Delete the context file after successful calendar sync."""
    _OAUTH_CONTEXT_FILE.unlink(missing_ok=True)
    logger.debug("[Calendar] Cleared plan context file")


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
    """Generate the Google OAuth consent URL (PKCE disabled).

    Returns
    -------
    (authorization_url, state)
        The URL to redirect the user to, and the CSRF state token.
    """
    flow = Flow.from_client_config(
        _get_client_config(),
        scopes=_SCOPES,
        redirect_uri=_REDIRECT_URI,
        autogenerate_code_verifier=False,  # ← disable PKCE
    )
    # UX: don't force the consent screen on every click. If the user has
    # already granted these scopes, Google will SSO-approve silently; if
    # not, they'll see the consent screen once. We rely on the immediate
    # token exchange (no long-lived refresh_token needed) so omitting
    # `prompt="consent"` is safe.
    authorization_url, state = flow.authorization_url(
        access_type="online",
        include_granted_scopes="true",
    )
    logger.info("[Calendar] Generated OAuth URL (PKCE disabled, SSO-friendly)")
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
        autogenerate_code_verifier=False,  # ← must match get_authorization_url
    )
    flow.fetch_token(code=code)
    creds = flow.credentials
    logger.info("[Calendar] Token exchange successful")

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
