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
</style>
""", unsafe_allow_html=True)

# ── Load env ──────────────────────────────────────────────────────
load_dotenv()
init_db()

if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    st.error(
        "**API key not found.**  "
        "Copy `.env.example` → `.env` and add your `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`), then restart."
    )
    st.stop()

# ── Google Calendar OAuth callback handler ────────────────────────
# When Google redirects back with ?code=..., intercept it here.

_gcal_params = st.query_params
if "code" in _gcal_params:
    _gcal_code = _gcal_params.get("code", "")
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

# ── Session state init ────────────────────────────────────────────

if "graph" not in st.session_state:
    st.session_state.graph = create_graph()

if "agent_state" not in st.session_state:
    user_email = os.getenv("FITGEN_USER_EMAIL", "").strip()
    context_id = os.getenv("FITGEN_CONTEXT_ID", str(uuid.uuid4()))
    restored = get_context_state(context_id) or get_latest_context_state_by_email(user_email) or {}
    context_id = restored.get("context_id", context_id)
    st.session_state.agent_state = {
        "messages": [],
        "user_profile": restored.get("user_profile", {}),
        "user_email": user_email or restored.get("user_email", ""),
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
    "activity_level":  {"type": "selectbox", "label": "Activity Level", "options": ["sedentary", "light", "moderate", "high", "athlete"]},
    "fitness_level":   {"type": "selectbox", "label": "Fitness Level", "options": ["beginner", "intermediate", "advanced"]},
    "equipment":       {"type": "text",      "label": "Equipment Available"},
    "workout_days":    {"type": "number",    "label": "Workout Days Per Week", "min": 1, "max": 7, "step": 1, "default": 4},
    "additional_info": {"type": "textarea",  "label": "Injuries, Physical Limitations, or Other Info (optional)"},
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

    # ── Controls ─────────────────────────────────────────────────
    show_base = st.checkbox(
        "Compare prompting techniques",
        value=False,
        help="Run all prompting techniques side-by-side on the base agent.",
    )

    if st.button("🗑️ Clear conversation", use_container_width=True):
        context_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.agent_state = {
            "messages": [],
            "user_profile": {},
            "user_email": st.session_state.agent_state.get("user_email", ""),
            "context_id": context_id,
            "state_id": context_id,
            "workflow": {},
            "calendar_sync_requested": False,
        }
        st.session_state.profile_form_pending = False
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

    # ── Live logs panel ──────────────────────────────────────────
    with st.expander("🪵 Logs", expanded=False):
        if _LOG_BUFFER:
            log_text = "\n".join(list(_LOG_BUFFER)[-80:])
            st.code(log_text, language="text")
            if st.button("Clear", key="clear_logs"):
                _LOG_BUFFER.clear()
                st.rerun()
        else:
            st.caption("No logs yet.")

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

        try:
          with st.status("Analyzing your request...", expanded=False) as status:
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
                        status.update(label=f"Routing to {tool_label}...")
                        badge_placeholder.markdown(_badge(tool_used), unsafe_allow_html=True)
                        _ui_logger.info("[Turn %s] Routed to tool=%s", turn_id, tool_used)

                    # Parse ToolMessage JSON → user-facing assistant message
                    from langchain_core.messages import ToolMessage
                    if isinstance(last_msg, ToolMessage) and last_msg.content:
                        status.update(label="Generating your response...")
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
                                if not _form_will_render:
                                    response_placeholder.markdown(response_content)
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
                        response_placeholder.markdown(response_content)
                        _ui_logger.debug(
                            "[Turn %s] Final AI direct reply received (chars=%d)",
                            turn_id,
                            len(response_content),
                        )

            elapsed = perf_counter() - turn_start
            status.update(label="Done", state="complete")
            _ui_logger.info(
                "[Turn %s] Stream completed in %.2fs with %d events",
                turn_id,
                elapsed,
                event_count,
            )
        except Exception as e:
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
