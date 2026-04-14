"""
streamlit_app.py
────────────────
FITGEN.AI — Streamlit chat frontend.

Runs a persistent, multi-turn conversation with the LangGraph agent.
The full agent state (including message history) is stored in
st.session_state so context is retained across turns within a session.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
# Fix protobuf C++ extension issue on Anaconda (must be set before any protobuf import)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import json
import logging
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as _components  # noqa: F401 — needed for st.components.v1.html
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent import AgentState, create_graph
from agent.feedback import get_average_rating, save_feedback
from agent.persistence import get_context_state, get_latest_context_state_by_email, init_db
from agent.config import DEFAULT_MODEL, FAST_MODEL
from agent.prompts.base_prompts import BASE_PROMPTS
from agent.prompts.techniques import TECHNIQUE_KEYS, TECHNIQUE_META
from agent.shared.types import DOMAIN_REQUIRED_FIELDS
from agent.db.repositories.user_repo import UserRepository
from agent.db.repositories.diet_plan_repo import DietPlanRepository
from agent.db.repositories.workout_plan_repo import WorkoutPlanRepository


# ── Log capture ───────────────────────────────────────────────────
_LOG_BUFFER_MAX = int(os.getenv("FITGEN_LOG_BUFFER", "1000"))
_LOG_BUFFER: deque[str] = deque(maxlen=_LOG_BUFFER_MAX)


class _BufferHandler(logging.Handler):
    """Thread-safe in-process log buffer for UI display."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _LOG_BUFFER.append(self.format(record))
        except Exception:  # noqa: BLE001
            pass


def _configure_logging() -> None:
    """Configure console + UI-buffer logging exactly once per process."""
    level_name = os.getenv("FITGEN_LOG_LEVEL", "DEBUG").upper()
    level = getattr(logging, level_name, logging.DEBUG)

    root = logging.getLogger()
    root.setLevel(level)

    if not any(getattr(h, "_fitgen_console", False) for h in root.handlers):
        console = logging.StreamHandler()
        console._fitgen_console = True  # type: ignore[attr-defined]
        console.setLevel(level)
        console.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "%H:%M:%S",
        ))
        root.addHandler(console)

    fitgen = logging.getLogger("fitgen")
    fitgen.setLevel(level)
    fitgen.propagate = True
    if not any(isinstance(h, _BufferHandler) for h in fitgen.handlers):
        buffer_handler = _BufferHandler()
        buffer_handler.setLevel(level)
        buffer_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "%H:%M:%S",
        ))
        fitgen.addHandler(buffer_handler)


_configure_logging()
_ui_logger = logging.getLogger("fitgen.streamlit")

_fitgen_logger = logging.getLogger("fitgen")
_fitgen_logger.setLevel(logging.DEBUG)

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="FITGEN.AI — AI Fitness Coach",
    page_icon="🔥",
    layout="wide",
)

# ── Custom CSS: Dark fitness theme ────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Fonts ────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Root variables ─────────────────────────────────── */
:root {
    --bg-dark: #0a0a0a;
    --bg-card: #1a1a1a;
    --bg-card-hover: #222222;
    --accent-orange: #ff6b2b;
    --accent-orange-glow: rgba(255, 107, 43, 0.3);
    --accent-red: #e63946;
    --accent-green: #2ecc71;
    --text-primary: #f5f5f5;
    --text-secondary: #a0a0a0;
    --text-muted: #666666;
    --border-subtle: #2a2a2a;
    --gradient-hero: linear-gradient(135deg, #ff6b2b 0%, #e63946 100%);
}

/* ── Global overrides ───────────────────────────────── */
.stApp {
    background-color: var(--bg-dark) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Sidebar styling ────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111111 0%, #0d0d0d 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

section[data-testid="stSidebar"] h2 {
    color: var(--accent-orange) !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Chat messages ──────────────────────────────────── */
.stChatMessage {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.75rem !important;
}

/* ── Chat input ─────────────────────────────────────── */
.stChatInput {
    border-color: var(--border-subtle) !important;
}

.stChatInput > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px !important;
}

.stChatInput textarea {
    color: var(--text-primary) !important;
}

/* ── Buttons ────────────────────────────────────────── */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px var(--accent-orange-glow) !important;
}

.stFormSubmitButton > button {
    background: var(--gradient-hero) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s ease !important;
}

.stFormSubmitButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px var(--accent-orange-glow) !important;
}

/* ── Forms ──────────────────────────────────────────── */
[data-testid="stForm"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}

/* ── Input fields ───────────────────────────────────── */
.stTextInput > div > div,
.stNumberInput > div > div,
.stSelectbox > div > div {
    background: #111111 !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

.stTextInput input,
.stNumberInput input {
    color: var(--text-primary) !important;
}

/* ── Progress bar ───────────────────────────────────── */
.stProgress > div > div > div {
    background: var(--gradient-hero) !important;
    border-radius: 10px !important;
}

/* ── Expanders ──────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}

/* ── Status widget ──────────────────────────────────── */
[data-testid="stStatusWidget"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
}

/* ── Dividers ───────────────────────────────────────── */
hr {
    border-color: var(--border-subtle) !important;
}

/* ── Custom hero section ────────────────────────────── */
.hero-container {
    background: linear-gradient(135deg, rgba(255,107,43,0.08) 0%, rgba(230,57,70,0.05) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(255,107,43,0.15) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ff6b2b 0%, #ff8f5e 50%, #e63946 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
    line-height: 1.1;
}

.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 1.25rem;
    max-width: 600px;
}

.hero-tags {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.hero-tag {
    background: rgba(255,107,43,0.12);
    color: var(--accent-orange);
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid rgba(255,107,43,0.2);
    letter-spacing: 0.02em;
}

/* ── Tool badges ────────────────────────────────────── */
.tool-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
    letter-spacing: 0.03em;
    margin-bottom: 8px;
}

.tool-badge-workout { background: linear-gradient(135deg, #2e7d32, #4caf50); }
.tool-badge-diet    { background: linear-gradient(135deg, #1565c0, #42a5f5); }
.tool-badge-general { background: linear-gradient(135deg, #6a1b9a, #ab47bc); }
.tool-badge-rag     { background: linear-gradient(135deg, #e65100, #ff9800); }

/* ── Sidebar stats cards ────────────────────────────── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}

.stat-card-label {
    color: var(--text-muted);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.stat-card-value {
    color: var(--accent-orange);
    font-size: 1.3rem;
    font-weight: 800;
}

/* ── Sidebar progress section ───────────────────────── */
.progress-stage {
    color: var(--text-secondary);
    font-size: 0.8rem;
    padding: 0.4rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.progress-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}

.progress-dot-done     { background: var(--accent-green); }
.progress-dot-active   { background: var(--accent-orange); box-shadow: 0 0 8px var(--accent-orange-glow); }
.progress-dot-pending  { background: var(--text-muted); }

/* ── Metrics override ───────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
}

[data-testid="stMetricValue"] {
    color: var(--accent-orange) !important;
    font-weight: 800 !important;
}

/* ── Scrollbar ──────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #555; }

/* ── AI Response Content Enhancement ───────────────── */

/* Section headings inside responses */
.stChatMessage h2, .stChatMessage h3, .stChatMessage h4 {
    color: var(--accent-orange) !important;
    font-weight: 700 !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.4rem !important;
    padding-bottom: 0.3rem !important;
    border-bottom: 1px solid rgba(255, 107, 43, 0.12) !important;
    font-size: 1rem !important;
    letter-spacing: 0.01em !important;
}
.stChatMessage h2 { font-size: 1.1rem !important; }

/* Bold exercise / food names */
.stChatMessage strong {
    color: #f0f0f0 !important;
    font-weight: 700 !important;
}

/* Ordered lists (exercises) */
.stChatMessage ol {
    counter-reset: item;
    list-style: none !important;
    padding-left: 0 !important;
    margin: 0.5rem 0 !important;
}
.stChatMessage ol > li {
    counter-increment: item;
    position: relative;
    padding: 0.65rem 0.9rem 0.65rem 2.8rem !important;
    margin-bottom: 0.35rem !important;
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    border-radius: 12px !important;
    border-left: 3px solid var(--accent-orange) !important;
    transition: background 0.2s ease;
}
.stChatMessage ol > li:hover {
    background: rgba(255, 107, 43, 0.04) !important;
}
.stChatMessage ol > li::before {
    content: counter(item);
    position: absolute;
    left: 0.75rem;
    top: 0.7rem;
    width: 1.4rem;
    height: 1.4rem;
    background: linear-gradient(135deg, rgba(255,107,43,0.2), rgba(230,57,70,0.15));
    color: var(--accent-orange);
    font-size: 0.75rem;
    font-weight: 800;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    line-height: 1;
}

/* Nested bullet points (exercise details) */
.stChatMessage ul {
    padding-left: 1.2rem !important;
    margin: 0.3rem 0 0.2rem 0 !important;
}
.stChatMessage ul > li {
    color: var(--text-secondary) !important;
    font-size: 0.88rem !important;
    line-height: 1.55 !important;
    padding: 0.1rem 0 !important;
    list-style-type: none !important;
    position: relative;
    padding-left: 1rem !important;
}
.stChatMessage ul > li::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0.55rem;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #555;
}

/* Paragraph text */
.stChatMessage p {
    line-height: 1.65 !important;
    margin-bottom: 0.5rem !important;
}

/* Streaming cursor */
.streaming-cursor {
    display: inline-block;
    width: 2px;
    height: 1.1em;
    background: var(--accent-orange);
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: blink 0.8s step-end infinite;
}
@keyframes blink {
    50% { opacity: 0; }
}

/* ── Profile avatar ────────────────────────────────── */
.profile-avatar {
    width: 38px; height: 38px; border-radius: 50%;
    background: linear-gradient(135deg, #ff6b2b, #e63946);
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 800; font-size: 1.05rem;
    flex-shrink: 0;
}
.profile-row {
    display: flex; align-items: center; gap: 0.65rem;
    padding: 0.3rem 0;
}
.profile-name {
    color: var(--text-primary); font-size: 0.88rem; font-weight: 600;
    line-height: 1.2;
}
.profile-email {
    color: var(--text-muted); font-size: 0.7rem; line-height: 1.2;
}

/* ── Workout card ──────────────────────────────────── */
.workout-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.4rem;
}
.workout-card-session {
    color: var(--accent-orange);
    font-size: 1rem; font-weight: 700;
}
.workout-card-day {
    color: var(--text-muted);
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.06em; font-weight: 600;
}
.workout-exercise {
    color: var(--text-secondary);
    font-size: 0.8rem; padding: 0.15rem 0;
    display: flex; justify-content: space-between;
}
.workout-exercise-sets {
    color: var(--text-muted); font-size: 0.75rem;
}

/* ── Water tracker ─────────────────────────────────── */
.water-progress-bar {
    background: #1a1a1a;
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    height: 18px;
    overflow: hidden;
    margin: 0.4rem 0;
}
.water-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #1a73e8, #34a0f5);
    border-radius: 8px;
    transition: width 0.3s ease;
}
.water-label {
    color: var(--text-secondary);
    font-size: 0.78rem;
    display: flex; justify-content: space-between;
    align-items: center;
}

/* ── Daily tip ─────────────────────────────────────── */
.tip-card {
    background: linear-gradient(135deg, rgba(255,107,43,0.08), rgba(230,57,70,0.05));
    border: 1px solid rgba(255,107,43,0.15);
    border-radius: 12px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.4rem;
}
.tip-text {
    color: var(--text-secondary);
    font-size: 0.82rem;
    line-height: 1.5;
    font-style: italic;
}
.tip-label {
    color: var(--accent-orange);
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.25rem;
}

/* ── Sidebar button & text spacing ────────────────── */
[data-testid="stSidebar"] .stButton {
    margin-bottom: 0.15rem !important;
    margin-top: 0.15rem !important;
}
[data-testid="stSidebar"] .stButton > button {
    padding: 0.35rem 0.75rem !important;
    font-size: 0.82rem !important;
    min-height: 2rem !important;
}
[data-testid="stSidebar"] .stPopover > div > button {
    padding: 0.35rem 0.75rem !important;
    font-size: 0.82rem !important;
    min-height: 2rem !important;
}
[data-testid="stSidebar"] h2 {
    font-size: 0.95rem !important;
    margin-top: 0.6rem !important;
    margin-bottom: 0.3rem !important;
}
[data-testid="stSidebar"] .stDivider {
    margin-top: 0.4rem !important;
    margin-bottom: 0.4rem !important;
}
[data-testid="stSidebar"] .stMarkdown {
    margin-bottom: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stPopover"] {
    margin-top: 0.3rem !important;
    margin-bottom: 0.2rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load env ──────────────────────────────────────────────────────
load_dotenv(override=True)
init_db()

if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    st.error(
        "**API key not found.**  "
        "Copy `.env.example` → `.env` and add your `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`), then restart."
    )
    st.stop()

# ── Auto-detect server port for OAuth redirect ──────────────────
# When multiple Streamlit apps run concurrently (8501, 8502, …), the
# redirect URI MUST match the port *this* app is listening on.
try:
    from streamlit import config as _st_config
    _server_port = _st_config.get_option("server.port") or 8501
except Exception:
    _server_port = 8501
_OAUTH_REDIRECT_URI = f"http://localhost:{_server_port}"

# ── Google OAuth callback handler ───────────────────────────────
# When Google redirects back with ?code=..., intercept it here.
# The ``state`` parameter distinguishes login vs. calendar callbacks.

_gcal_params = st.query_params
if "code" in _gcal_params:
    _gcal_code = _gcal_params.get("code", "")
    _oauth_state = _gcal_params.get("state", "")

    # ── Login OAuth callback ────────────────────────────────────
    if _oauth_state.startswith("login_"):
        st.query_params.clear()
        _login_logger = logging.getLogger("fitgen.auth.streamlit")
        _login_logger.info("[Auth] Received login OAuth code (state=%s…)", _oauth_state[:20])
        try:
            from agent.auth.google_auth import GoogleAuthProvider
            _auth_provider = GoogleAuthProvider(redirect_uri=_OAUTH_REDIRECT_URI)
            _auth_user = _auth_provider.handle_callback(_gcal_code)
            # Store authenticated user in session
            st.session_state["authenticated"] = True
            st.session_state["auth_user_email"] = _auth_user.email
            st.session_state["auth_user_name"] = _auth_user.name
            st.session_state["auth_user_picture"] = _auth_user.picture
            _login_logger.info("[Auth] Login successful: %s", _auth_user.email)
            st.rerun()
        except Exception as _auth_err:
            _login_logger.error("[Auth] Login failed: %s", _auth_err, exc_info=True)
            st.error(f"Login failed: {_auth_err}")
            st.stop()

    # ── Calendar / Fit OAuth callback ───────────────────────────
    else:
        # Clear query params immediately to prevent re-processing on rerun.
        st.query_params.clear()

        _gcal_logger = logging.getLogger("fitgen.calendar.streamlit")
        _gcal_logger.info("[Calendar] Received OAuth code, exchanging for tokens…")

        try:
            from agent.tools.calendar_integration import (
                clear_oauth_context,
                exchange_code_for_tokens,
                extract_calendar_events,
                load_oauth_context,
                push_events_to_calendar,
            )

            with st.spinner("🔐 Exchanging authorization code..."):
                # Exchange auth code for tokens (PKCE is disabled — no code_verifier needed).
                tokens = exchange_code_for_tokens(_gcal_code)

            st.info("✅ Google authorization successful! Processing your plan...")

            # Load plan data from persisted context file (survives the redirect).
            # Fall back to session state if context file was empty.
            _oauth_ctx = load_oauth_context()
            _wf = st.session_state.get("agent_state", {}).get("workflow", {})
            _plan_text = _oauth_ctx.get("plan_text") or _wf.get("plan_text", "")
            _domain = _oauth_ctx.get("domain") or _wf.get("domain", "diet")
            _profile = _oauth_ctx.get("profile") or st.session_state.get("agent_state", {}).get("user_profile", {})

            _sync_target = _oauth_ctx.get("sync_target", "calendar")
            _gcal_logger.info("[Calendar] sync_target=%s, plan_len=%d, domain=%s, profile_keys=%s",
                              _sync_target, len(_plan_text), _domain,
                              sorted(_profile.keys()) if _profile else "empty")
            _gcal_logger.info("[Calendar] plan_text preview: %s", _plan_text[:200] if _plan_text else "(empty)")

            if _plan_text:
                # ── Google Calendar sync ──
                if _sync_target in ("calendar", "both"):
                    with st.spinner("📅 Extracting calendar events from your plan..."):
                        events = extract_calendar_events(_plan_text, _domain, _profile)
                    if events:
                        with st.spinner(f"📅 Pushing {len(events)} events to Google Calendar..."):
                            created_count = push_events_to_calendar(events, tokens)
                        st.success(
                            f"📅 **{created_count} events** synced to your Google Calendar! "
                            f"Check your calendar for recurring {_domain} reminders starting tomorrow."
                        )
                        _gcal_logger.info("[Calendar] Pushed %d events successfully", created_count)
                        st.session_state["calendar_events_pushed"] = True
                    else:
                        st.warning("⚠️ Connected to Google Calendar, but couldn't extract events from your plan.")

                # ── Google Fit sync ──
                if _sync_target in ("google_fit", "both"):
                    from agent.tools.google_fit_integration import (
                        extract_nutrition_data,
                        extract_activity_sessions,
                        push_nutrition_to_google_fit,
                        push_activities_to_google_fit,
                    )
                    if _domain == "diet":
                        _gcal_logger.info("[GoogleFit] About to extract nutrition: plan_len=%d, profile_keys=%s",
                                          len(_plan_text), sorted(_profile.keys()) if _profile else "empty")
                        nutrition = None
                        for _attempt in range(1, 3):  # 2 attempts
                            with st.spinner(f"💪 Extracting nutrition data (attempt {_attempt}/2)..."):
                                try:
                                    nutrition = extract_nutrition_data(_plan_text, _domain, _profile)
                                except Exception as _ext_err:
                                    _gcal_logger.error("[GoogleFit] Extraction attempt %d failed: %s",
                                                        _attempt, _ext_err, exc_info=True)
                                    nutrition = []
                            if nutrition:
                                break
                            _gcal_logger.warning("[GoogleFit] Extraction attempt %d returned 0 entries", _attempt)

                        _gcal_logger.info("[GoogleFit] Final extraction: %d entries", len(nutrition) if nutrition else 0)
                        if nutrition:
                            st.info(f"Found {len(nutrition)} meals. Pushing to Google Fit...")
                            with st.spinner(f"💪 Pushing {len(nutrition)} meals to Google Fit..."):
                                fit_count, fit_errors = push_nutrition_to_google_fit(nutrition, tokens)
                            st.session_state["_gfit_push_count"] = fit_count
                            if fit_count > 0:
                                st.success(
                                    f"💪 **{fit_count} nutrition entries** synced to Google Fit! "
                                    f"Check the Google Fit app for your meal data."
                                )
                            if fit_errors:
                                st.warning(
                                    f"⚠️ {len(fit_errors)} entries failed:\n"
                                    + "\n".join(f"- {e}" for e in fit_errors[:5])
                                )
                            if fit_count == 0 and not fit_errors:
                                st.warning("⚠️ No entries were pushed. Check the logs for details.")
                            _gcal_logger.info("[GoogleFit] Pushed %d nutrition entries, %d errors", fit_count, len(fit_errors))
                        else:
                            st.warning(
                                f"⚠️ Connected to Google Fit, but couldn't extract nutrition data.\n\n"
                                f"**Debug**: plan_text length = {len(_plan_text)}, domain = {_domain}, "
                                f"profile fields = {len(_profile)} — check Logs panel for details."
                            )
                    elif _domain == "workout":
                        with st.spinner("💪 Extracting workout sessions from your plan..."):
                            sessions = extract_activity_sessions(_plan_text, _domain, _profile)
                        _gcal_logger.info("[GoogleFit] Extracted %d activity sessions", len(sessions) if sessions else 0)
                        if sessions:
                            st.info(f"Found {len(sessions)} workout sessions. Pushing to Google Fit...")
                            with st.spinner(f"💪 Pushing {len(sessions)} sessions to Google Fit..."):
                                fit_count, fit_errors = push_activities_to_google_fit(sessions, tokens)
                            st.session_state["_gfit_push_count"] = fit_count
                            if fit_count > 0:
                                st.success(
                                    f"💪 **{fit_count} workout sessions** synced to Google Fit! "
                                    f"Check the Google Fit app for your activity data."
                                )
                            if fit_errors:
                                st.warning(
                                    f"⚠️ {len(fit_errors)} sessions failed:\n"
                                    + "\n".join(f"- {e}" for e in fit_errors[:5])
                                )
                            _gcal_logger.info("[GoogleFit] Pushed %d sessions, %d errors", fit_count, len(fit_errors))
                        else:
                            st.warning("⚠️ Connected to Google Fit, but couldn't extract activity sessions from your plan.")
                    _gfit_actually_pushed = st.session_state.get("_gfit_push_count", 0) > 0
                    st.session_state["google_fit_data_pushed"] = _gfit_actually_pushed

                # Store tokens and clean up.
                st.session_state["google_calendar_tokens"] = tokens
                clear_oauth_context()
            else:
                st.warning("⚠️ Connected to Google, but no plan text found. Create a plan first, then sync.")

        except Exception as e:
            st.error(f"❌ Sync failed: {e}")
            _gcal_logger.error("[Calendar] OAuth or push failed: %s", e, exc_info=True)

# ── Authentication gate ───────────────────────────────────────────
# Resolve the current user email: either from Google OAuth login or
# the FITGEN_USER_EMAIL env var (dev bypass).

_dev_email = os.getenv("FITGEN_USER_EMAIL", "").strip()

# Check if the user is already authenticated (OAuth login or dev bypass)
_is_authenticated = st.session_state.get("authenticated", False)

if not _is_authenticated:
    # Check if Google OAuth is configured
    from agent.auth.google_auth import GoogleAuthProvider
    _google_auth = GoogleAuthProvider(redirect_uri=_OAUTH_REDIRECT_URI)

    # ── Login page CSS ────────────────────────────────────────────
    # NOTE: Streamlit wraps each st.markdown / st.button in its own
    # container, so we CANNOT nest native widgets inside a custom HTML
    # div.  Instead we style the page-level containers directly.
    st.markdown("""
    <style>
    /* ── Login: hide chrome ──────────────────────────── */
    section[data-testid="stSidebar"]  { display: none !important; }
    header[data-testid="stHeader"]    { display: none !important; }
    #MainMenu, footer                 { display: none !important; }

    /* ── Login: animated background on .stApp ─────────── */
    @keyframes blob1 {
        0%, 100% { transform: translate(0, 0) scale(1); }
        33%      { transform: translate(30px, -50px) scale(1.1); }
        66%      { transform: translate(-20px, 20px) scale(0.9); }
    }
    @keyframes blob2 {
        0%, 100% { transform: translate(0, 0) scale(1); }
        33%      { transform: translate(-40px, 30px) scale(1.15); }
        66%      { transform: translate(25px, -40px) scale(0.85); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes subtlePulse {
        0%, 100% { opacity: 0.85; }
        50%      { opacity: 1; }
    }

    .stApp::before {
        content: '';
        position: fixed; top: -10%; right: -5%;
        width: 600px; height: 600px; border-radius: 50%;
        background: radial-gradient(circle, rgba(255,107,43,0.15) 0%, transparent 70%);
        filter: blur(80px);
        animation: blob1 15s ease-in-out infinite;
        pointer-events: none; z-index: 0;
    }
    .stApp::after {
        content: '';
        position: fixed; bottom: -10%; left: -5%;
        width: 500px; height: 500px; border-radius: 50%;
        background: radial-gradient(circle, rgba(230,57,70,0.10) 0%, transparent 70%);
        filter: blur(80px);
        animation: blob2 18s ease-in-out infinite;
        pointer-events: none; z-index: 0;
    }

    /* ── Login: center column acts as card ────────────── */
    [data-testid="stAppViewContainer"] [data-testid="stVerticalBlock"] {
        max-width: 460px;
        margin: 0 auto;
        padding-top: 6vh;
        animation: fadeInUp 0.8s ease-out;
        position: relative;
        z-index: 1;
    }

    /* ── Login: Google button (white, prominent) ─────── */
    a[data-testid="baseLinkButton-secondary"] {
        background: #ffffff !important;
        color: #1a1a1a !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.78rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 15px rgba(0,0,0,0.2) !important;
        text-decoration: none !important;
    }
    a[data-testid="baseLinkButton-secondary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(255,255,255,0.08) !important;
    }
    a[data-testid="baseLinkButton-secondary"] p {
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }

    /* ── Login: dev button (subtle, dashed) ──────────── */
    button[data-testid="baseButton-secondary"] {
        background: transparent !important;
        color: #666 !important;
        border: 1px dashed #333 !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        padding: 0.55rem 1rem !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.01em !important;
    }
    button[data-testid="baseButton-secondary"]:hover {
        border-color: #555 !important;
        color: #999 !important;
        background: rgba(255,255,255,0.03) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Login page layout ─────────────────────────────────────────
    # Glass-morphism card (entire block via CSS, content via markdown)
    st.markdown(
        '<div style="'
        'background:rgba(22,22,22,0.65);'
        'backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);'
        'border:1px solid rgba(255,107,43,0.10);'
        'border-radius:28px;'
        'padding:2.5rem 2.5rem 1rem 2.5rem;'
        'box-shadow:0 8px 40px rgba(0,0,0,0.35),0 0 80px rgba(255,107,43,0.03);'
        '">'
        # ── Logo icon
        '<div style="'
        'width:68px;height:68px;margin:0 auto 1.2rem auto;'
        'display:flex;align-items:center;justify-content:center;'
        'background:linear-gradient(135deg,rgba(255,107,43,0.14),rgba(230,57,70,0.14));'
        'border-radius:20px;border:1px solid rgba(255,107,43,0.18);'
        'font-size:1.9rem;animation:subtlePulse 3s ease-in-out infinite;'
        '">&#x1F525;</div>'
        # ── Title
        '<div style="'
        'font-family:Inter,sans-serif;font-size:2.2rem;font-weight:900;'
        'text-align:center;letter-spacing:-0.02em;margin-bottom:0.3rem;'
        'background:linear-gradient(135deg,#ff6b2b 0%,#ff8f5e 40%,#e63946 100%);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'background-clip:text;'
        '">FITGEN.AI</div>'
        # ── Subtitle
        '<div style="text-align:center;color:#777;font-size:0.92rem;'
        'margin-bottom:0.2rem;">Your AI-powered personal fitness coach</div>'
        # ── Feature pills
        '<div style="display:flex;flex-wrap:wrap;justify-content:center;'
        'gap:0.4rem;margin:1.2rem 0 1.5rem 0;">'
        '<span style="background:rgba(255,107,43,0.08);color:#cc6a3a;'
        'padding:0.22rem 0.7rem;border-radius:20px;font-size:0.7rem;'
        'font-weight:600;border:1px solid rgba(255,107,43,0.12);">'
        'Workout Plans</span>'
        '<span style="background:rgba(255,107,43,0.08);color:#cc6a3a;'
        'padding:0.22rem 0.7rem;border-radius:20px;font-size:0.7rem;'
        'font-weight:600;border:1px solid rgba(255,107,43,0.12);">'
        'Diet &amp; Nutrition</span>'
        '<span style="background:rgba(255,107,43,0.08);color:#cc6a3a;'
        'padding:0.22rem 0.7rem;border-radius:20px;font-size:0.7rem;'
        'font-weight:600;border:1px solid rgba(255,107,43,0.12);">'
        'Macro Tracking</span>'
        '<span style="background:rgba(255,107,43,0.08);color:#cc6a3a;'
        'padding:0.22rem 0.7rem;border-radius:20px;font-size:0.7rem;'
        'font-weight:600;border:1px solid rgba(255,107,43,0.12);">'
        'Calendar Sync</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Google sign-in button (native Streamlit widget, styled white via CSS)
    if _google_auth.is_configured:
        _login_url = _google_auth.get_login_url()
        st.link_button(
            "\U0001f310  Sign in with Google",
            _login_url,
            use_container_width=True,
        )
    else:
        st.warning(
            "Google OAuth not configured. "
            "Set `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in `.env`."
        )

    # "or" divider + dev button
    if _dev_email:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:1rem;'
            'margin:0.6rem 0 0.4rem 0;">'
            '<div style="flex:1;height:1px;'
            'background:linear-gradient(90deg,transparent,#333,transparent);"></div>'
            '<span style="color:#555;font-size:0.72rem;font-weight:500;'
            'text-transform:uppercase;letter-spacing:0.1em;">or</span>'
            '<div style="flex:1;height:1px;'
            'background:linear-gradient(90deg,transparent,#333,transparent);"></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Sign in as Dev", use_container_width=True):
            st.session_state["authenticated"] = True
            st.session_state["auth_user_email"] = _dev_email
            st.session_state["auth_user_name"] = _dev_email.split("@")[0]
            st.session_state["auth_user_picture"] = ""
            st.rerun()

    # Footer
    st.markdown(
        '<div style="text-align:center;color:#3a3a3a;font-size:0.7rem;'
        'margin-top:1.5rem;letter-spacing:0.01em;">'
        'Powered by OpenAI &middot; LangGraph &middot; MongoDB Atlas'
        '</div>',
        unsafe_allow_html=True,
    )

    st.stop()  # ← block everything below until authenticated

# ── Resolve authenticated user email ─────────────────────────────
_active_email = st.session_state.get("auth_user_email", "") or _dev_email

# ── Session state init ────────────────────────────────────────────

if "graph" not in st.session_state:
    st.session_state.graph = create_graph()

if "agent_state" not in st.session_state:
    user_email = _active_email
    context_id = os.getenv("FITGEN_CONTEXT_ID", str(uuid.uuid4()))
    restored = get_context_state(context_id) or get_latest_context_state_by_email(user_email) or {}
    context_id = restored.get("context_id", context_id)

    # Load existing user profile from MongoDB (do NOT create user doc here —
    # the user doc is only created on plan confirm in the tool handlers).
    user_id = ""
    mongo_profile: dict = {}
    if user_email:
        existing_user = UserRepository.find_by_email(user_email)
        if existing_user:
            user_id = str(existing_user["_id"])
            mongo_profile = UserRepository.get_merged_profile(user_email)

    merged_profile = {**mongo_profile, **restored.get("user_profile", {})}

    st.session_state.agent_state = {
        "messages": [],
        "user_profile": merged_profile,
        "user_email": user_email or restored.get("user_email", ""),
        "user_id": user_id,
        "context_id": context_id,
        "state_id": context_id,
        "workflow": restored.get("workflow", {}),
        "calendar_sync_requested": restored.get("calendar_sync_requested", False),
    }

# chat_history entries: {role, content, tool_used, technique_results, base_results}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "profile_form_pending" not in st.session_state:
    st.session_state.profile_form_pending = False

if "water_glasses" not in st.session_state:
    st.session_state.water_glasses = 0

# ── Sidebar data helpers (cached) ────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def _get_confirmed_diet_plan(user_id: str) -> dict | None:
    """Fetch the latest confirmed diet plan (cached 30 s)."""
    if not user_id:
        return None
    return DietPlanRepository.find_latest_by_user(user_id, status="confirmed")


@st.cache_data(ttl=30, show_spinner=False)
def _get_confirmed_workout_plan(user_id: str) -> dict | None:
    """Fetch the latest confirmed workout plan (cached 30 s)."""
    if not user_id:
        return None
    return WorkoutPlanRepository.find_latest_by_user(user_id, status="confirmed")


def _resolve_user_id() -> str:
    """Get user_id from agent_state, or look it up from email if missing.

    When a plan is confirmed mid-session, the tool creates the user in
    MongoDB but agent_state['user_id'] may still be empty. This resolves
    that by checking the DB.
    """
    uid = st.session_state.agent_state.get("user_id", "")
    if uid:
        return uid
    email = st.session_state.agent_state.get("user_email", "")
    if not email:
        return ""
    existing = UserRepository.find_by_email(email)
    if existing:
        uid = str(existing["_id"])
        st.session_state.agent_state["user_id"] = uid  # persist for rest of session
    return uid


@st.cache_data(ttl=43200, show_spinner=False)
def _generate_daily_tip(goal: str, date_str: str) -> str:
    """Generate a personalized daily fitness tip using the fast model.

    Cached for 12 hours per (goal, date) combo.
    """
    try:
        _tip_llm = ChatOpenAI(model=FAST_MODEL, temperature=0.8, max_tokens=60)
        resp = _tip_llm.invoke(
            f"You are a concise fitness coach. Give ONE short, specific, "
            f"actionable tip (under 20 words) for someone whose goal is: "
            f"{goal or 'general fitness'}. Today is {date_str}. "
            f"Don't start with 'Tip:'. Be motivational but practical."
        )
        return resp.content.strip().strip('"').strip("'")
    except Exception:
        return ""


# ── Helpers ───────────────────────────────────────────────────────────

TOOL_LABELS: dict[str, tuple[str, str, str]] = {
    "workout_tool":    ("💪 Workout Coach",   "tool-badge-workout",  "#2e7d32"),
    "diet_tool":       ("🥗 Diet Coach",      "tool-badge-diet",     "#1565c0"),
    "rag_query_tool":  ("📚 Knowledge Base",  "tool-badge-rag",      "#e65100"),
    "general":         ("🤖 FITGEN.AI",       "tool-badge-general",  "#6a1b9a"),
}


def _badge(tool: str) -> str:
    """Return an HTML pill badge for the given tool name."""
    label, css_class, _fallback = TOOL_LABELS.get(tool, TOOL_LABELS["general"])
    return f'<span class="tool-badge {css_class}">{label}</span>'


def _copy_button(text: str, key: str) -> None:
    """Render a small JS-powered copy-to-clipboard button.

    Uses base64 encoding to safely embed arbitrary text (markdown tables,
    special characters, emojis, newlines) inside the JS without breaking
    the HTML template.
    """
    import base64

    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    st.components.v1.html(
        f"""
        <button onclick="
            var t = atob('{b64}');
            navigator.clipboard.writeText(t).then(function(){{
                this.textContent='Copied!';
                var btn=this;
                setTimeout(function(){{btn.textContent='Copy';}}, 1500);
            }}.bind(this));
        "
        style="background:#374151;color:#e5e7eb;border:none;border-radius:6px;
               padding:4px 12px;font-size:0.75rem;cursor:pointer;margin-top:4px;">
            Copy
        </button>
        """,
        height=38,
    )


def _stream_response(text: str, placeholder) -> None:
    """Simulate streaming by revealing text progressively with a cursor.

    Adapts speed to content length:
      - Short  (<60 words):  word-by-word,  ~16 ms
      - Medium (60-300):      2-word chunks, ~12 ms
      - Long   (>300 words):  4-word chunks, ~10 ms

    Uses a blinking cursor (▌) during streaming for a ChatGPT-like feel.
    """
    import time as _time

    words = text.split(" ")
    total = len(words)
    if total == 0:
        placeholder.markdown(text)
        return

    # Adaptive speed — snappy but readable
    if total < 60:
        chunk_size, delay = 1, 0.016
    elif total < 300:
        chunk_size, delay = 2, 0.012
    else:
        chunk_size, delay = 4, 0.010

    accumulated: list[str] = []

    for i, word in enumerate(words):
        accumulated.append(word)
        # Render on every chunk_size-th word, or on the last word
        if (i + 1) % chunk_size == 0 or i == total - 1:
            display = " ".join(accumulated)
            if i < total - 1:
                placeholder.markdown(display + " ▌")
            else:
                placeholder.markdown(display)
            _time.sleep(delay)


def _render_plan(plan_text: str) -> None:
    """Render a plan — try JSON structured display, fallback to markdown.

    If *plan_text* is valid JSON (from ``generate_plan_as_json``), iterate
    its top-level keys and render each section with appropriate formatting
    (tables for day-based meal/workout data, bullet lists, plain text).
    No hardcoded schema — handles whatever structure the LLM returned.
    """
    try:
        plan = json.loads(plan_text)
        if not isinstance(plan, dict):
            raise TypeError("Plan is not a dict")

        for section_key, section_data in plan.items():
            # Section header: convert snake_case → Title Case
            header = section_key.replace("_", " ").title()
            st.markdown(f"### {header}")

            if isinstance(section_data, dict):
                # Nested dict (e.g. day-by-day meals, macro targets)
                for sub_key, sub_val in section_data.items():
                    sub_header = sub_key.replace("_", " ").title()
                    st.markdown(f"**{sub_header}**")
                    if isinstance(sub_val, list):
                        for item in sub_val:
                            st.markdown(f"- {item}" if isinstance(item, str) else f"- {json.dumps(item)}")
                    elif isinstance(sub_val, dict):
                        for k, v in sub_val.items():
                            st.markdown(f"- **{k}**: {v}")
                    else:
                        st.markdown(str(sub_val))
            elif isinstance(section_data, list):
                for item in section_data:
                    if isinstance(item, dict):
                        parts = [f"**{k}**: {v}" for k, v in item.items()]
                        st.markdown("- " + " | ".join(parts))
                    else:
                        st.markdown(f"- {item}")
            else:
                st.markdown(str(section_data))

            st.markdown("---")

    except (json.JSONDecodeError, TypeError, KeyError):
        # Not JSON or unexpected structure — render as markdown
        st.markdown(plan_text)


# ── Profile intake form config ────────────────────────────────────────

PROFILE_FORM_FIELDS: dict[str, dict] = {
    # ── Stats ────────────────────────────────────────────────────
    "name":            {"type": "text",      "label": "Name"},
    "age":             {"type": "number",    "label": "Age", "min": 10, "max": 100, "step": 1, "default": 25},
    "sex":             {"type": "selectbox", "label": "Biological Sex", "options": ["male", "female"]},
    "height_cm":       {"type": "number",    "label": "Height (cm)", "min": 50.0, "max": 300.0, "step": 0.5, "default": 170.0},
    "weight_kg":       {"type": "number",    "label": "Current Weight (kg)", "min": 20.0, "max": 300.0, "step": 0.5, "default": 70.0},
    "goal":            {"type": "selectbox", "label": "Primary Goal", "options": ["fat loss", "muscle gain", "maintenance", "performance"]},
    "goal_weight":     {"type": "text",      "label": "Goal Weight (kg) or how you want to look/feel"},
    "weight_loss_pace": {"type": "selectbox", "label": "How quickly?", "options": ["steady & sustainable", "moderate", "as fast as safely possible"]},
    # ── Lifestyle ────────────────────────────────────────────────
    "job_type":        {"type": "selectbox", "label": "Job Type", "options": ["desk job", "on my feet (retail/teaching)", "manual labour", "work from home", "other"]},
    "exercise_frequency": {"type": "selectbox", "label": "Exercise per Week", "options": ["0 (none)", "1-2 times", "3-4 times", "5-6 times", "daily"]},
    "exercise_type":   {"type": "text",      "label": "Type of Exercise (weights, running, sports, yoga, etc.)"},
    "sleep_hours":     {"type": "number",    "label": "Sleep (hours/night)", "min": 3, "max": 12, "step": 1, "default": 7},
    "stress_level":    {"type": "selectbox", "label": "Stress Level", "options": ["low", "moderate", "high"]},
    "alcohol_intake":  {"type": "text",      "label": "Alcohol Intake (e.g. '4 beers/week' or 'none')"},
    # ── Food Preferences ─────────────────────────────────────────
    "diet_preference": {"type": "selectbox", "label": "Diet Preference", "options": ["omnivore", "vegetarian", "vegan", "eggetarian", "pescatarian"]},
    "favourite_meals": {"type": "textarea",  "label": "Top 5 Favourite Meals / Dishes (any cuisine)"},
    "foods_to_avoid":  {"type": "text",      "label": "Foods You Hate / Would Never Eat (or 'none')"},
    "allergies":       {"type": "text",      "label": "Allergies / Intolerances (or 'none')"},
    "cooking_style":   {"type": "selectbox", "label": "Cooking Style", "options": ["cook from scratch", "quick meals (under 20 min)", "batch meal prep", "mix of all"]},
    "food_adventurousness": {"type": "number", "label": "Food Adventurousness (1-10)", "min": 1, "max": 10, "step": 1, "default": 5},
    # ── Snack Habits ─────────────────────────────────────────────
    "current_snacks":  {"type": "text",      "label": "Current Snacks (what you reach for during the day)"},
    "snack_reason":    {"type": "selectbox", "label": "Why Do You Snack?", "options": ["hunger", "boredom", "habit", "mix of all"]},
    "snack_preference": {"type": "selectbox", "label": "Snack Preference", "options": ["sweet", "savoury", "both"]},
    "late_night_snacking": {"type": "selectbox", "label": "Late Night Snacking?", "options": ["yes", "sometimes", "no"]},
    # ── Workout-specific ─────────────────────────────────────────
    "experience_level":       {"type": "selectbox", "label": "Experience Level", "options": ["beginner", "intermediate", "advanced"]},
    "training_days_per_week": {"type": "number",    "label": "Training Days Per Week", "min": 1, "max": 7, "step": 1, "default": 4},
    "session_duration":       {"type": "number",    "label": "Session Duration (minutes)", "min": 15, "max": 180, "step": 5, "default": 45},
    "daily_steps":            {"type": "number",    "label": "Daily Steps (approx.)", "min": 0, "max": 50000, "step": 500, "default": 5000},
    "additional_info":        {"type": "textarea",  "label": "Injuries, Physical Limitations, or Other Info (optional)"},
}


def _render_profile_form(domain: str, existing_profile: dict) -> dict | None:
    """Render a Streamlit form for profile intake.

    Only shows fields that are missing from existing_profile.
    Returns the *complete* profile (existing + new) on submit, else None.
    If no fields are missing returns the existing profile immediately (auto-skip).
    """
    required = DOMAIN_REQUIRED_FIELDS.get(domain, list(PROFILE_FORM_FIELDS.keys()))
    tool_name = "diet_tool" if domain == "diet" else "workout_tool"

    # Determine which fields still need to be collected
    missing_fields = [f for f in required if not existing_profile.get(f)]

    # If everything is already filled (e.g. switching from diet → workout
    # where base fields are shared), auto-skip the form entirely.
    if not missing_fields:
        return dict(existing_profile)

    st.markdown(_badge(tool_name), unsafe_allow_html=True)
    if len(missing_fields) < len(required):
        filled = [f for f in required if f not in missing_fields]
        filled_summary = ", ".join(
            f"**{PROFILE_FORM_FIELDS[f]['label']}**: {existing_profile[f]}"
            for f in filled if f in PROFILE_FORM_FIELDS
        )
        st.markdown(f"I already have: {filled_summary}")
        st.markdown("**Please fill in the remaining details:**")
    else:
        st.markdown("**Please fill in your profile details to get started:**")

    # Separate regular fields from full-width fields (textarea)
    _OPTIONAL_FIELDS = {"additional_info", "favourite_meals"}
    _regular_fields = [f for f in missing_fields if PROFILE_FORM_FIELDS.get(f, {}).get("type") != "textarea"]
    _fullwidth_fields = [f for f in missing_fields if PROFILE_FORM_FIELDS.get(f, {}).get("type") == "textarea"]

    _form_key = f"profile_intake_{domain}"
    with st.form(_form_key, clear_on_submit=False):
        form_values: dict = {}
        col1, col2 = st.columns(2)

        for i, field in enumerate(_regular_fields):
            cfg = PROFILE_FORM_FIELDS.get(field)
            if not cfg:
                continue
            target_col = col1 if i % 2 == 0 else col2
            _wkey = f"form_{domain}_{field}"

            with target_col:
                if cfg["type"] == "text":
                    form_values[field] = st.text_input(
                        cfg["label"],
                        value="",
                        key=_wkey,
                    )
                elif cfg["type"] == "number":
                    form_values[field] = st.number_input(
                        cfg["label"],
                        min_value=cfg["min"],
                        max_value=cfg["max"],
                        value=cfg.get("default", cfg["min"]),
                        step=cfg["step"],
                        key=_wkey,
                    )
                elif cfg["type"] == "selectbox":
                    form_values[field] = st.selectbox(
                        cfg["label"],
                        options=cfg["options"],
                        index=0,
                        key=_wkey,
                    )

        # Full-width fields (textarea) rendered below the columns
        for field in _fullwidth_fields:
            cfg = PROFILE_FORM_FIELDS.get(field)
            if not cfg:
                continue
            _wkey = f"form_{domain}_{field}"
            form_values[field] = st.text_area(
                cfg["label"],
                value="",
                height=100,
                key=_wkey,
            )

        submitted = st.form_submit_button("Submit Profile", use_container_width=True, type="primary")

    if submitted:
        # Validate required fields (skip optional ones)
        _required_missing = [
            f for f in missing_fields
            if f not in _OPTIONAL_FIELDS and not form_values.get(f)
        ]
        if _required_missing:
            labels = [PROFILE_FORM_FIELDS[f]["label"] for f in _required_missing if f in PROFILE_FORM_FIELDS]
            st.warning(f"Please fill in: {', '.join(labels)}")
            return None
        # Default optional empty fields to 'none'
        for f in _OPTIONAL_FIELDS:
            if f in form_values and not form_values[f].strip():
                form_values[f] = "none"
        # Merge existing profile with new form values
        merged = dict(existing_profile)
        merged.update(form_values)
        return merged

    return None


def _render_technique_tabs(results: dict[str, str]) -> None:
    """Render a dict of {technique_key: response_text} as labelled st.tabs."""
    keys_present = [k for k in TECHNIQUE_KEYS if k in results]
    tab_labels = [
        f"{TECHNIQUE_META[k]['icon']} {TECHNIQUE_META[k]['label']}"
        for k in keys_present
    ]
    tabs = st.tabs(tab_labels)
    for tab, key in zip(tabs, keys_present):
        meta = TECHNIQUE_META[key]
        with tab:
            st.caption(meta["description"])
            st.divider()
            st.markdown(results[key])
            _copy_button(results[key], key=f"copy_{key}")


def _run_base_comparison(query: str) -> dict[str, str]:
    """Run query through all 5 BASE_PROMPTS variants in parallel."""
    _ui_logger.info("[BaseCompare] Starting comparison across %d techniques", len(TECHNIQUE_KEYS))

    def _call(tech: str) -> tuple[str, str]:
        llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)
        resp = llm.invoke(
            [SystemMessage(content=BASE_PROMPTS[tech]), HumanMessage(content=query)]
        )
        _ui_logger.debug("[BaseCompare] Completed technique=%s, chars=%d", tech, len(resp.content or ""))
        return tech, resp.content

    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(_call, t): t for t in TECHNIQUE_KEYS}
        for f in futures:
            tech, text = f.result()
            results[tech] = text
    _ui_logger.info("[BaseCompare] Completed")
    return {k: results[k] for k in TECHNIQUE_KEYS if k in results}


# ── Hero Section ──────────────────────────────────────────────────

_is_fresh_session = len(st.session_state.chat_history) == 0

if _is_fresh_session:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">FITGEN.AI</div>
        <div class="hero-subtitle">
            Your AI-powered fitness coach. Get personalized workout routines and
            diet plans tailored to your goals, body type, and lifestyle.
        </div>
        <div class="hero-tags">
            <span class="hero-tag">🏋️ Workout Plans</span>
            <span class="hero-tag">🥗 Diet & Nutrition</span>
            <span class="hero-tag">📊 Macro Tracking</span>
            <span class="hero-tag">📅 Calendar Sync</span>
            <span class="hero-tag">🤖 AI Powered</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick-start prompt suggestions
    st.markdown(
        "<p style='color:#666;font-size:0.85rem;margin-bottom:0.5rem;font-weight:600;'>"
        "GET STARTED</p>",
        unsafe_allow_html=True,
    )
    _qcols = st.columns(3)
    with _qcols[0]:
        if st.button("🥗 Create a diet plan", use_container_width=True):
            st.session_state._quick_prompt = "Create a diet plan"
            st.rerun()
    with _qcols[1]:
        if st.button("💪 Create a workout plan", use_container_width=True):
            st.session_state._quick_prompt = "Create a workout plan"
            st.rerun()
    with _qcols[2]:
        if st.button("❓ What can you do?", use_container_width=True):
            st.session_state._quick_prompt = "What can you help me with?"
            st.rerun()
    st.markdown("")
else:
    st.markdown(
        '<div style="padding:0.5rem 0 0.25rem 0;">'
        '<span style="font-size:1.4rem;font-weight:800;'
        'background:linear-gradient(135deg,#ff6b2b,#e63946);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
        '🔥 FITGEN.AI</span></div>',
        unsafe_allow_html=True,
    )

# ── Sidebar ───────────────────────────────────────────────────────

with st.sidebar:
    # ── Branding ───────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;padding:0.5rem 0 1rem 0;">'
        '<span style="font-size:1.6rem;font-weight:900;'
        'background:linear-gradient(135deg,#ff6b2b,#e63946);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
        '🔥 FITGEN.AI</span><br>'
        '<span style="color:#666;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;">'
        'AI-Powered Fitness Coach</span></div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Profile dropdown ─────────────────────────────────────
    _sidebar_name = st.session_state.get("auth_user_name", "")
    _sidebar_email = st.session_state.get("auth_user_email", "")
    if _sidebar_email:
        _display = _sidebar_name or _sidebar_email.split("@")[0]
        _initial = _display[0].upper() if _display else "U"

        # Avatar + name row
        st.markdown(
            f'<div class="profile-row">'
            f'<div class="profile-avatar">{_initial}</div>'
            f'<div><div class="profile-name">{_display}</div>'
            f'<div class="profile-email">{_sidebar_email}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Check if user already has confirmed plans in DB
        _pop_user_id = _resolve_user_id()
        _pop_has_diet = bool(_get_confirmed_diet_plan(_pop_user_id)) if _pop_user_id else False
        _pop_has_workout = bool(_get_confirmed_workout_plan(_pop_user_id)) if _pop_user_id else False

        with st.popover("Menu", use_container_width=True):
            st.caption(f"Signed in as **{_display}**")

            # Diet button — "Get" if plan exists, "Create" if not
            _diet_label = "🥗 Get Diet Plan" if _pop_has_diet else "🥗 Create Diet Plan"
            _diet_prompt = "Get my diet plan" if _pop_has_diet else "Create a diet plan"
            if st.button(_diet_label, key="pop_diet", use_container_width=True):
                st.session_state._quick_prompt = _diet_prompt
                st.rerun()

            # Workout button — "Get" if plan exists, "Create" if not
            _wk_label = "💪 Get Workout Plan" if _pop_has_workout else "💪 Create Workout Plan"
            _wk_prompt = "Get my workout plan" if _pop_has_workout else "Create a workout plan"
            if st.button(_wk_label, key="pop_workout", use_container_width=True):
                st.session_state._quick_prompt = _wk_prompt
                st.rerun()

            st.divider()
            if st.button("🚪 Logout", key="pop_logout", use_container_width=True):
                for _k in ("authenticated", "auth_user_email", "auth_user_name",
                            "auth_user_picture", "agent_state", "chat_history",
                            "profile_form_pending", "graph", "water_glasses"):
                    st.session_state.pop(_k, None)
                st.rerun()
        st.divider()

    # ── Workflow Progress ──────────────────────────────────────
    st.markdown("## 📊 Session")
    _workflow = st.session_state.agent_state.get("workflow", {})
    _step = _workflow.get("step_completed") or _workflow.get("stage", "")
    _domain = _workflow.get("domain", "")
    _completed = _workflow.get("completed_steps", [])

    _STAGES = [
        ("profile_collection_started", "Profile"),
        ("profile_mapped",             "Confirm"),
        ("plan_generated",             "Plan Review"),
        ("calendar_sync_started",      "Calendar"),
    ]
    _is_complete = any(
        s in _completed
        for s in (
            "diet_confirmed", "workout_confirmed",
            "calendar_sync_started", "google_fit_sync_started",
        )
    )

    if _domain:
        st.markdown(
            f'<div class="stat-card"><div class="stat-card-label">Active Domain</div>'
            f'<div class="stat-card-value">{_domain.title()}</div></div>',
            unsafe_allow_html=True,
        )

    for _s_key, _s_label in _STAGES:
        if _is_complete or _s_key in _completed or any(_s_key in c for c in _completed):
            _dot = "progress-dot-done"
            _style = "color:#2ecc71;"
        elif _s_key == _step or _s_key in _step:
            _dot = "progress-dot-active"
            _style = "color:#ff6b2b;font-weight:600;"
        else:
            _dot = "progress-dot-pending"
            _style = "color:#666;"
        st.markdown(
            f'<div class="progress-stage">'
            f'<span class="progress-dot {_dot}"></span>'
            f'<span style="{_style}">{_s_label}</span></div>',
            unsafe_allow_html=True,
        )

    if _is_complete:
        st.success("Plan complete!")

    st.divider()

    # ── Stats ────────────────────────────────────────────────────
    _ctx_id = st.session_state.agent_state.get("context_id", "")
    _avg_rating = get_average_rating(_ctx_id) if _ctx_id else None

    if "response_times" not in st.session_state:
        st.session_state.response_times = []

    _stat_cols = st.columns(2)
    with _stat_cols[0]:
        _rating_val = f"{_avg_rating}/5" if _avg_rating else "—"
        st.markdown(
            f'<div class="stat-card"><div class="stat-card-label">Rating</div>'
            f'<div class="stat-card-value">{_rating_val}</div></div>',
            unsafe_allow_html=True,
        )
    with _stat_cols[1]:
        _avg_t = (
            f"{sum(st.session_state.response_times) / len(st.session_state.response_times):.1f}s"
            if st.session_state.response_times
            else "—"
        )
        st.markdown(
            f'<div class="stat-card"><div class="stat-card-label">Avg Speed</div>'
            f'<div class="stat-card-value">{_avg_t}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Macro Pie Chart ─────────────────────────────────────────
    _sb_user_id = _resolve_user_id()
    _sb_diet = _get_confirmed_diet_plan(_sb_user_id) if _sb_user_id else None

    if _sb_diet:
        try:
            from agent.diet_visuals import extract_macros_from_plan, create_macro_donut_chart

            # Prefer structured_data from DB; fall back to regex parsing
            _sd = _sb_diet.get("structured_data", {})
            _sd_macros = _sd.get("macros", {})

            if _sd_macros:
                _sb_p = _sd_macros.get("protein_g", 0)
                _sb_c = _sd_macros.get("carbs_g", 0)
                _sb_f = _sd_macros.get("fat_g", 0)
                _sb_kcal = _sd_macros.get("calories", 0)
            else:
                # Fallback: regex parse from markdown
                _fb = extract_macros_from_plan(_sb_diet.get("plan_markdown", ""))
                _sb_p = _fb.get("protein_g", 0)
                _sb_c = _fb.get("carbs_g", 0)
                _sb_f = _fb.get("fat_g", 0)
                _sb_kcal = _fb.get("total_kcal", 0)

            if _sb_p > 0 and _sb_c > 0 and _sb_f > 0:
                st.markdown("## 🥧 Macros")
                _fig = create_macro_donut_chart(_sb_p, _sb_c, _sb_f, _sb_kcal, compact=True)
                st.pyplot(_fig, use_container_width=True)
                plt.close(_fig)
                st.divider()
        except Exception:
            pass  # silently skip if parsing fails

    # ── Today's Workout ─────────────────────────────────────────
    _sb_workout = _get_confirmed_workout_plan(_sb_user_id) if _sb_user_id else None

    if _sb_workout:
        try:
            from datetime import datetime as _dt_cls
            _today_abbr = _dt_cls.now().strftime("%a")
            _DAY_NAMES = {
                "Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday",
                "Thu": "Thursday", "Fri": "Friday", "Sat": "Saturday",
                "Sun": "Sunday",
            }

            # Prefer structured_data from DB; fall back to regex parsing
            _sd_w = _sb_workout.get("structured_data", {})
            _schedule = _sd_w.get("schedule", [])

            _todays = None
            if _schedule:
                for _entry in _schedule:
                    if _entry.get("day") == _today_abbr:
                        _todays = {
                            "day_name": _DAY_NAMES.get(_today_abbr, _today_abbr),
                            "session_name": _entry.get("session", "Workout"),
                            "exercises": _entry.get("exercises", []),
                        }
                        break
            else:
                # Fallback: regex parse from markdown
                from agent.workout_visuals import extract_todays_workout
                _todays = extract_todays_workout(_sb_workout.get("plan_markdown", ""))

            st.markdown("## 🏋️ Today")

            if _todays:
                st.markdown(
                    f'<div class="workout-card">'
                    f'<div class="workout-card-day">{_todays["day_name"]}</div>'
                    f'<div class="workout-card-session">{_todays["session_name"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                for _ex in _todays["exercises"][:6]:
                    _ex_name = _ex.get("name", "")
                    _ex_sets = _ex.get("sets_reps", "")
                    st.markdown(
                        f'<div class="workout-exercise">'
                        f'<span>{_ex_name}</span>'
                        f'<span class="workout-exercise-sets">{_ex_sets}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                if len(_todays["exercises"]) > 6:
                    st.caption(f"+{len(_todays['exercises']) - 6} more exercises")
            else:
                st.markdown(
                    '<div class="workout-card">'
                    '<div class="workout-card-day">Rest Day</div>'
                    '<div class="workout-card-session">Recover & Stretch</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Light walking or yoga for active recovery.")
            st.divider()
        except Exception:
            pass

    # ── AI Daily Tip ────────────────────────────────────────────
    _sb_profile = st.session_state.agent_state.get("user_profile", {})
    _sb_goal = _sb_profile.get("goal", "")
    if _sb_goal:
        from datetime import date as _date_cls
        _tip_text = _generate_daily_tip(_sb_goal, _date_cls.today().isoformat())
        if _tip_text:
            st.markdown(
                f'<div class="tip-card">'
                f'<div class="tip-label">💡 Daily Tip</div>'
                f'<div class="tip-text">{_tip_text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.divider()

    # ── Water Intake Tracker ────────────────────────────────────
    _water_target_liters = 3.0  # sensible default
    if _sb_diet:
        try:
            # Prefer structured_data; fall back to regex
            _sd_h = _sb_diet.get("structured_data", {}).get("hydration", {})
            if _sd_h:
                _water_target_liters = _sd_h.get("rest_day_liters", 3.0)
            elif _sb_diet.get("plan_markdown"):
                from agent.diet_visuals import extract_hydration_target
                _hydration = extract_hydration_target(_sb_diet["plan_markdown"])
                _water_target_liters = _hydration.get("rest_day_liters", 3.0)
        except Exception:
            pass

    _total_glasses = max(int(_water_target_liters / 0.25), 8)
    _current_glasses = st.session_state.get("water_glasses", 0)
    _current_ml = _current_glasses * 250
    _target_ml = _total_glasses * 250
    _pct = min((_current_glasses / _total_glasses) * 100, 100) if _total_glasses > 0 else 0

    st.markdown("## 💧 Water")
    st.markdown(
        f'<div class="water-label">'
        f'<span>{_current_ml} ml</span>'
        f'<span>{_target_ml} ml</span>'
        f'</div>'
        f'<div class="water-progress-bar">'
        f'<div class="water-progress-fill" style="width:{_pct:.0f}%"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"{_current_glasses} / {_total_glasses} glasses (250 ml each)")

    _wcols = st.columns(3)
    with _wcols[0]:
        if st.button("➖", key="water_minus", use_container_width=True):
            if st.session_state.water_glasses > 0:
                st.session_state.water_glasses -= 1
                st.rerun()
    with _wcols[1]:
        if st.button("➕", key="water_plus", use_container_width=True):
            st.session_state.water_glasses += 1
            st.rerun()
    with _wcols[2]:
        if st.button("↺", key="water_reset", use_container_width=True, help="Reset to 0"):
            st.session_state.water_glasses = 0
            st.rerun()

    if _pct >= 100:
        st.success("Daily target reached!")

    st.divider()

    # ── Controls ─────────────────────────────────────────────────
    show_base = False  # technique comparison disabled for production

    if st.button("🗑️ Clear conversation", use_container_width=True):
        context_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.agent_state = {
            "messages": [],
            "user_profile": {},
            "user_email": st.session_state.agent_state.get("user_email", ""),
            "user_id": st.session_state.agent_state.get("user_id", ""),
            "context_id": context_id,
            "state_id": context_id,
            "workflow": {},
            "calendar_sync_requested": False,
        }
        st.session_state.profile_form_pending = False
        st.session_state.water_glasses = 0
        st.rerun()

    st.divider()

    # ── Google Calendar ──────────────────────────────────────────
    st.markdown("## 📅 Calendar")

    _has_google_creds = bool(os.getenv("GOOGLE_CLIENT_ID")) and bool(os.getenv("GOOGLE_CLIENT_SECRET"))
    _wf_state = st.session_state.get("agent_state", {}).get("workflow", {})
    _calendar_step = _wf_state.get("step_completed") or _wf_state.get("stage")
    _already_pushed = st.session_state.get("calendar_events_pushed", False)

    _calendar_sync_active = _calendar_step in (
        "diet_plan_synced_to_google_calendar",
        "workout_plan_synced_to_google_calendar",
        "calendar_oauth_pending",  # legacy
    )

    if _already_pushed:
        st.success("✅ Calendar synced!")
    elif _has_google_creds and _calendar_sync_active:
        try:
            from agent.tools.calendar_integration import (
                get_authorization_url, save_oauth_context, load_oauth_context as _cal_load_ctx,
            )
            # Generate auth URL (PKCE disabled — no code_verifier to manage).
            _auth_url, _ = get_authorization_url()

            # Persist plan data to temp file so it survives the OAuth redirect.
            # Only write if we have plan text — don't overwrite a good context
            # file with empty data on Streamlit reruns.
            _oauth_plan_text = _wf_state.get("plan_text", "")
            if not _oauth_plan_text:
                _existing_cal_ctx = _cal_load_ctx()
                _oauth_plan_text = _existing_cal_ctx.get("plan_text", "")

            if _oauth_plan_text:
                _oauth_domain = _wf_state.get("domain", "diet")
                _oauth_profile = st.session_state.get("agent_state", {}).get("user_profile", {})
                save_oauth_context(
                    plan_text=_oauth_plan_text,
                    domain=_oauth_domain,
                    profile=_oauth_profile,
                    sync_target="both",
                )

            st.link_button("📅 Connect Google Calendar", _auth_url, use_container_width=True)
            st.caption("Sign in with Google to sync your plan.")
        except Exception as _e:
            st.caption(f"⚠️ Calendar error: {_e}")
    elif not _has_google_creds:
        st.caption("Add Google credentials to `.env` to enable calendar sync.")
    else:
        st.caption("Generate a plan to enable calendar sync.")

    # ── Google Fit ──────────────────────────────────────────────
    st.markdown("## 💪 Google Fit")

    _gfit_already_pushed = st.session_state.get("google_fit_data_pushed", False)

    _gfit_sync_active = _calendar_step in (
        "diet_plan_synced_to_google_fit",
        "workout_plan_synced_to_google_fit",
        "google_fit_oauth_pending",  # legacy
    )

    if _gfit_already_pushed:
        st.success("✅ Google Fit synced!")
    elif _has_google_creds and _gfit_sync_active:
        try:
            from agent.tools.calendar_integration import (
                get_authorization_url as _gfit_get_auth_url,
                save_oauth_context as _gfit_save_ctx,
                load_oauth_context as _gfit_load_ctx,
            )
            _gfit_auth_url, _ = _gfit_get_auth_url()

            # Only save context if we have plan text — don't overwrite
            # a good context file with empty data on Streamlit reruns.
            _gfit_plan_text = _wf_state.get("plan_text", "")
            if not _gfit_plan_text:
                # Try loading from existing context file
                _existing_ctx = _gfit_load_ctx()
                _gfit_plan_text = _existing_ctx.get("plan_text", "")

            if _gfit_plan_text:
                _gfit_domain = _wf_state.get("domain", "diet")
                _gfit_profile = st.session_state.get("agent_state", {}).get("user_profile", {})
                _gfit_save_ctx(
                    plan_text=_gfit_plan_text,
                    domain=_gfit_domain,
                    profile=_gfit_profile,
                    sync_target="google_fit",
                )

            st.link_button("💪 Sync to Google Fit", _gfit_auth_url, use_container_width=True)
            st.caption("Sign in with Google to sync nutrition/activity data.")
        except Exception as _e:
            st.caption(f"⚠️ Google Fit error: {_e}")
    elif not _has_google_creds:
        st.caption("Add Google credentials to `.env` to enable Google Fit.")
    else:
        st.caption("Generate a plan to enable Google Fit sync.")

    st.divider()

    st.markdown(
        '<div style="text-align:center;padding:1rem 0;color:#444;font-size:0.65rem;">'
        'Built with LangGraph · LangChain · OpenAI · Streamlit</div>',
        unsafe_allow_html=True,
    )

# ── Render existing chat history ──────────────────────────────────

for i, entry in enumerate(st.session_state.chat_history):
    with st.chat_message(entry["role"]):
        if entry["role"] == "assistant":
            if entry.get("tool_used"):
                st.markdown(_badge(entry["tool_used"]), unsafe_allow_html=True)
            if entry.get("technique_results"):
                with st.expander("📊 Specialist Prompt Comparison", expanded=True):
                    _render_technique_tabs(entry["technique_results"])
            if entry.get("base_results"):
                with st.expander("🤖 Base Agent Technique Comparison", expanded=False):
                    _render_technique_tabs(entry["base_results"])
            if entry.get("content"):
                st.markdown(entry["content"])
                _copy_button(entry["content"], key=f"copy_reply_{i}")
        else:
            st.markdown(entry["content"])

# ── Profile intake form (shown when workflow is in collect_profile stage) ──

_current_workflow = st.session_state.agent_state.get("workflow", {})
_current_step = _current_workflow.get("step_completed") or _current_workflow.get("stage")
if _current_step == "prompted_for_user_profile_data" and st.session_state.profile_form_pending:
    _form_domain = _current_workflow.get("domain", "diet")
    _form_existing = st.session_state.agent_state.get("user_profile", {})
    _form_required = DOMAIN_REQUIRED_FIELDS.get(_form_domain, [])
    _form_missing = [f for f in _form_required if not _form_existing.get(f)]

    if not _form_missing:
        # All fields already filled (e.g. switching diet → workout with shared base fields).
        # Auto-confirm and skip the form entirely.
        _form_data = dict(_form_existing)
    else:
        # Show form for the missing fields only.
        with st.chat_message("assistant"):
            _form_data = _render_profile_form(_form_domain, _form_existing)

    if _form_data is not None:
        st.session_state.profile_form_pending = False
        # Populate profile in agent_state so the tool sees it in ctx.profile.
        # Do NOT touch workflow/step_completed — the tool manages its own flow.
        st.session_state.agent_state["user_profile"].update(_form_data)
        # Send the actual form data as the user message — let the tool handle it.
        _form_parts = [f"{k}: {v}" for k, v in _form_data.items() if v not in (None, "")]
        st.session_state._pending_form_message = ", ".join(_form_parts)
        st.rerun()

# ── Chat input ────────────────────────────────────────────────────

# Always render chat_input so it never disappears
chat_input_value = st.chat_input("Ask me anything about fitness, workouts, or nutrition…")
_pending_form_msg = st.session_state.pop("_pending_form_message", None)
_quick_prompt = st.session_state.pop("_quick_prompt", None)
prompt = _pending_form_msg or _quick_prompt or chat_input_value

_is_form_auto_confirm = _pending_form_msg is not None

if prompt:
    turn_id = uuid.uuid4().hex[:8]
    turn_start = perf_counter()
    _ui_logger.info("[Turn %s] Received user prompt: %s", turn_id, prompt)

    # ── Display user message (hide synthetic form confirmation) ───
    if not _is_form_auto_confirm:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

    # ── Append to agent state ─────────────────────────────────────
    st.session_state.agent_state["messages"].append(HumanMessage(content=prompt))

    # ── Stream agent response ─────────────────────────────────────
    with st.chat_message("assistant"):
        badge_placeholder = st.empty()
        tabs_placeholder = st.empty()
        base_tabs_placeholder = st.empty()
        response_placeholder = st.empty()

        response_content = ""
        tool_used = ""              # name of the @tool that fired
        technique_results: dict = {}  # parsed JSON from ToolMessage
        base_results: dict = {}       # comparison across BASE_PROMPTS
        final_event: dict = {}
        tool_direct_reply = False
        _form_will_render = False   # suppress display when form takes over

        # Rotating status phrases per stage — cycled by a background thread
        import threading, time as _time, itertools
        _STATUS_THINKING = itertools.cycle([
            "Thinking...",
            "Understanding your question...",
            "Analyzing your request...",
        ])
        _STATUS_TOOL: dict[str, list[str]] = {
            "workout_tool": [
                "Consulting Workout Coach...",
                "Reviewing exercise science...",
                "Designing your routine...",
                "Selecting optimal exercises...",
                "Building your workout plan...",
                "Personalizing for your goals...",
            ],
            "diet_tool": [
                "Consulting Diet Coach...",
                "Analyzing nutritional needs...",
                "Designing your meal plan...",
                "Balancing your macros...",
                "Building your diet plan...",
                "Personalizing for your goals...",
            ],
        }
        _STATUS_DEFAULT_TOOL = [
            "Consulting specialist...",
            "Working on your request...",
            "Preparing your answer...",
        ]
        _STATUS_GENERATING = [
            "Crafting your personalized response...",
            "Polishing the details...",
            "Putting it all together...",
            "Almost there...",
        ]

        _status_phase = "thinking"   # thinking → tool → generating
        _status_stop = threading.Event()

        def _rotate_status(st_status):
            """Background thread: rotate status label every 1.8 s."""
            _tool_cycle = None
            _gen_cycle = itertools.cycle(_STATUS_GENERATING)
            while not _status_stop.is_set():
                _status_stop.wait(1.8)
                if _status_stop.is_set():
                    break
                try:
                    if _status_phase == "thinking":
                        st_status.update(label=next(_STATUS_THINKING))
                    elif _status_phase == "tool":
                        if _tool_cycle is None:
                            phrases = _STATUS_TOOL.get(tool_used, _STATUS_DEFAULT_TOOL)
                            _tool_cycle = itertools.cycle(phrases)
                        st_status.update(label=next(_tool_cycle))
                    elif _status_phase == "generating":
                        st_status.update(label=next(_gen_cycle))
                except Exception:
                    pass

        try:
          with st.status("Thinking...", expanded=False) as status:
            _rotator = threading.Thread(target=_rotate_status, args=(status,), daemon=True)
            _rotator.start()

            event_count = 0
            for event in st.session_state.graph.stream(
                st.session_state.agent_state, stream_mode="values"
            ):
                final_event = event
                event_count += 1
                _ui_logger.debug(
                    "[Turn %s] Stream event #%d keys=%s",
                    turn_id,
                    event_count,
                    sorted(event.keys()),
                )

                if event.get("messages"):
                    last_msg = event["messages"][-1]

                    # Detect @tool call intent (latest message only)
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls and not tool_used:
                        tool_used = last_msg.tool_calls[0]["name"]
                        tool_label, _, _ = TOOL_LABELS.get(tool_used, ("Specialist", "", "#333"))
                        _status_phase = "tool"
                        status.update(label=f"Consulting {tool_label}...")
                        badge_placeholder.markdown(_badge(tool_used), unsafe_allow_html=True)
                        _ui_logger.info("[Turn %s] Routed to tool=%s", turn_id, tool_used)

                    # Parse ToolMessage JSON → user-facing assistant message
                    from langchain_core.messages import ToolMessage
                    if isinstance(last_msg, ToolMessage) and last_msg.content:
                        _status_phase = "generating"
                        status.update(label="Crafting your personalized response...")
                        try:
                            parsed = json.loads(last_msg.content)
                            # Check if workflow entered profile collection — form will handle display
                            _tool_state = parsed.get("state_updates", {})
                            _tool_wf = _tool_state.get("workflow", {})
                            _tool_step = _tool_wf.get("step_completed") or _tool_wf.get("stage")
                            if _tool_step == "prompted_for_user_profile_data":
                                _form_will_render = True

                            assistant_message = parsed.get("assistant_message")
                            if assistant_message and assistant_message != response_content:
                                response_content = assistant_message
                                tool_direct_reply = True
                                _ui_logger.debug(
                                    "[Turn %s] Tool assistant_message received (chars=%d)",
                                    turn_id,
                                    len(assistant_message),
                                )
                            if isinstance(parsed, dict) and set(parsed.keys()) & set(TECHNIQUE_KEYS):
                                technique_results = parsed
                                _ui_logger.debug("[Turn %s] Technique comparison payload detected", turn_id)
                                with tabs_placeholder.container():
                                    with st.expander("📊 Specialist Prompt Comparison", expanded=True):
                                        _render_technique_tabs(technique_results)
                        except (json.JSONDecodeError, TypeError):
                            _ui_logger.warning("[Turn %s] ToolMessage JSON parse failed", turn_id)

                    # Final AIMessage text (not tool-call, not ToolMessage)
                    if (
                        hasattr(last_msg, "content")
                        and last_msg.content
                        and last_msg.content != response_content
                        and not getattr(last_msg, "tool_calls", None)
                        and not isinstance(last_msg, ToolMessage)
                        and not tool_direct_reply
                        and not _form_will_render
                    ):
                        response_content = last_msg.content
                        _ui_logger.debug(
                            "[Turn %s] Final AI direct reply received (chars=%d)",
                            turn_id,
                            len(response_content),
                        )

            _status_stop.set()
            elapsed = perf_counter() - turn_start
            status.update(label="Here you go!", state="complete")
            _ui_logger.info(
                "[Turn %s] Stream completed in %.2fs with %d events",
                turn_id,
                elapsed,
                event_count,
            )
        except Exception as e:
            _status_stop.set()
            elapsed = perf_counter() - turn_start
            _ui_logger.error("[Turn %s] Error: %s", turn_id, e)
            import openai
            if isinstance(e, openai.RateLimitError):
                st.warning("Rate limit reached. Please wait a moment and try again.")
            elif isinstance(e, openai.APITimeoutError):
                st.error("Request timed out. Please try again.")
            else:
                st.error(f"An error occurred: {e}")
            response_content = ""

        # ── Stream the response with typewriter effect ──────────
        if response_content and not _form_will_render:
            _stream_response(response_content, response_placeholder)

        # ── Optional: base agent technique comparison ─────────────
        if show_base and tool_used:
            with st.spinner("Running base-agent comparison across techniques…"):
                base_results = _run_base_comparison(prompt)
            with base_tabs_placeholder.container():
                with st.expander("🤖 Base Agent Technique Comparison", expanded=False):
                    _render_technique_tabs(base_results)

        # ── Copy button for the final reply ───────────────────────
        if response_content:
            _copy_button(response_content, key="copy_live_reply")

        # ── Response time display ────────────────────────────────
        if response_content:
            st.caption(f"Response time: {elapsed:.1f}s")
            st.session_state.response_times.append(round(elapsed, 1))

        # ── Visual plan outputs ─────────────────────────────────────
        # Auto-render rich visuals for diet plans (no keyword needed).
        # Workout visuals still require explicit user request.
        _viz_keywords = ["chart", "graph", "visuali", "pie chart", "show me a", "plot", "diagram"]
        _user_wants_viz = any(kw in prompt.lower() for kw in _viz_keywords)
        _is_diet_plan = (
            tool_used == "diet_tool"
            and response_content
            and len(response_content) > 800
            and any(kw in response_content.lower() for kw in ["macro", "meal plan", "7-day", "protein"])
        )

        if response_content and tool_used and (_user_wants_viz or _is_diet_plan):
            _profile = st.session_state.agent_state.get("user_profile", {})
            try:
                if tool_used == "workout_tool" and _user_wants_viz:
                    from agent.visualizations import (
                        create_progress_timeline,
                        create_weekly_schedule,
                    )
                    with st.expander("📈 Visual Plan Outputs", expanded=True):
                        _days = _profile.get("workout_days", 4)
                        fig1 = create_weekly_schedule(
                            workout_days=_days if isinstance(_days, int) else 4,
                            plan_text=response_content,
                            profile=_profile,
                        )
                        st.pyplot(fig1)
                        plt.close(fig1)

                        fig2 = create_progress_timeline(
                            weeks=12,
                            profile=_profile,
                        )
                        st.pyplot(fig2)
                        plt.close(fig2)

                elif _is_diet_plan:
                    from agent.diet_visuals import (
                        extract_macros_from_plan,
                        create_macro_donut_chart,
                    )

                    # ── Macro Pie Chart (next to summary) ───────────────
                    _macros = extract_macros_from_plan(response_content)
                    _p_g = _macros.get("protein_g", 0)
                    _c_g = _macros.get("carbs_g", 0)
                    _f_g = _macros.get("fat_g", 0)
                    _total = _macros.get("total_kcal", 0)

                    if _p_g > 0 and _c_g > 0 and _f_g > 0:
                        st.markdown("---")
                        st.markdown(
                            '<p style="font-size:1.1rem;font-weight:800;color:#ff6b2b;'
                            'margin-bottom:4px;">📊 Macro Distribution</p>',
                            unsafe_allow_html=True,
                        )
                        _macro_col1, _macro_col2 = st.columns([3, 2])
                        with _macro_col1:
                            # Summary cards
                            _mc1, _mc2, _mc3 = st.columns(3)
                            with _mc1:
                                st.markdown(
                                    '<div style="background:#1a1a1a;border:1px solid #2ecc71;'
                                    'border-radius:12px;padding:14px;text-align:center;">'
                                    '<div style="color:#888;font-size:0.7rem;font-weight:600;'
                                    'text-transform:uppercase;letter-spacing:0.08em;">Protein</div>'
                                    f'<div style="color:#2ecc71;font-size:1.6rem;font-weight:900;">'
                                    f'{_p_g:.0f}g</div>'
                                    f'<div style="color:#555;font-size:0.75rem;">'
                                    f'{_p_g * 4:.0f} kcal</div></div>',
                                    unsafe_allow_html=True,
                                )
                            with _mc2:
                                st.markdown(
                                    '<div style="background:#1a1a1a;border:1px solid #3498db;'
                                    'border-radius:12px;padding:14px;text-align:center;">'
                                    '<div style="color:#888;font-size:0.7rem;font-weight:600;'
                                    'text-transform:uppercase;letter-spacing:0.08em;">Carbs</div>'
                                    f'<div style="color:#3498db;font-size:1.6rem;font-weight:900;">'
                                    f'{_c_g:.0f}g</div>'
                                    f'<div style="color:#555;font-size:0.75rem;">'
                                    f'{_c_g * 4:.0f} kcal</div></div>',
                                    unsafe_allow_html=True,
                                )
                            with _mc3:
                                st.markdown(
                                    '<div style="background:#1a1a1a;border:1px solid #e74c3c;'
                                    'border-radius:12px;padding:14px;text-align:center;">'
                                    '<div style="color:#888;font-size:0.7rem;font-weight:600;'
                                    'text-transform:uppercase;letter-spacing:0.08em;">Fat</div>'
                                    f'<div style="color:#e74c3c;font-size:1.6rem;font-weight:900;">'
                                    f'{_f_g:.0f}g</div>'
                                    f'<div style="color:#555;font-size:0.75rem;">'
                                    f'{_f_g * 9:.0f} kcal</div></div>',
                                    unsafe_allow_html=True,
                                )

                        with _macro_col2:
                            _donut_fig = create_macro_donut_chart(_p_g, _c_g, _f_g, _total)
                            st.pyplot(_donut_fig, use_container_width=True)
                            plt.close(_donut_fig)

            except Exception as viz_err:
                _ui_logger.warning("Visualization error: %s", viz_err, exc_info=True)

        # ── User feedback widget ─────────────────────────────────
        if response_content:
            feedback_col1, feedback_col2 = st.columns([1, 3])
            with feedback_col1:
                rating = st.feedback("stars", key=f"fb_{turn_id}")
            with feedback_col2:
                if rating is not None:
                    _ctx = st.session_state.agent_state.get("context_id", "")
                    save_feedback(_ctx, turn_id, rating + 1)  # st.feedback is 0-indexed
                    st.caption("Thanks for your feedback!")

    # ── Sync agent state with final graph state ───────────────────
    if final_event and final_event.get("messages"):
        st.session_state.agent_state["messages"] = final_event["messages"]
        if "user_profile" in final_event:
            st.session_state.agent_state["user_profile"] = final_event["user_profile"]
        if "user_email" in final_event:
            st.session_state.agent_state["user_email"] = final_event["user_email"]
        if "workflow" in final_event:
            st.session_state.agent_state["workflow"] = final_event["workflow"]
        if "context_id" in final_event:
            st.session_state.agent_state["context_id"] = final_event["context_id"]
        if "state_id" in final_event:
            st.session_state.agent_state["state_id"] = final_event["state_id"]
        if "calendar_sync_requested" in final_event:
            st.session_state.agent_state["calendar_sync_requested"] = final_event["calendar_sync_requested"]

        _ui_logger.info(
            "[Turn %s] State sync complete: workflow=%s, profile_keys=%s, calendar_sync_requested=%s",
            turn_id,
            st.session_state.agent_state.get("workflow"),
            sorted(st.session_state.agent_state.get("user_profile", {}).keys()),
            st.session_state.agent_state.get("calendar_sync_requested"),
        )

    # ── Trigger profile form when entering profile collection step ──
    _synced_wf = st.session_state.agent_state.get("workflow", {})
    _synced_step = _synced_wf.get("step_completed") or _synced_wf.get("stage")
    if _synced_step == "prompted_for_user_profile_data" and not st.session_state.profile_form_pending:
        st.session_state.profile_form_pending = True
        st.rerun()
    elif _synced_step != "prompted_for_user_profile_data":
        st.session_state.profile_form_pending = False

    # ── Persist to display history ────────────────────────────────
    # Skip persisting when the form will take over (profile questions
    # are handled by the form widget, not the chat history).
    if not _form_will_render:
        assistant_entry = {
            "role": "assistant",
            "content": response_content,
            "tool_used": tool_used or "",
            "technique_results": technique_results,
            "base_results": base_results,
        }
        if not (
            st.session_state.chat_history
            and st.session_state.chat_history[-1].get("role") == "assistant"
            and st.session_state.chat_history[-1].get("content") == response_content
        ):
            st.session_state.chat_history.append(assistant_entry)
    _ui_logger.info("[Turn %s] Assistant response persisted to history", turn_id)
