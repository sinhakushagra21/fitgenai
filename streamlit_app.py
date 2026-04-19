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
import re
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from time import perf_counter

import markdown as _md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pdfkit
import streamlit as st
import streamlit.components.v1 as _components  # noqa: F401 — needed for st.components.v1.html
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent import AgentState, create_graph
from agent.feedback import get_average_rating, save_feedback
from agent.persistence import init_db
from agent.config import DEFAULT_MODEL, FAST_MODEL
from agent.logging_config import setup_logging
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
    """Configure console + UI-buffer logging exactly once per process.

    Delegates the console handler + format + noisy-logger muting to
    :func:`agent.logging_config.setup_logging`, then attaches a Streamlit
    log-viewer buffer so the user can read backend logs in the sidebar.
    """
    # Root + fitgen console handler, colour, noisy-logger muting.
    setup_logging()

    fitgen = logging.getLogger("fitgen")
    # Attach the UI buffer handler (idempotent).
    if not any(isinstance(h, _BufferHandler) for h in fitgen.handlers):
        from agent.logging_config import FitgenFormatter
        buffer_handler = _BufferHandler()
        buffer_handler.setFormatter(FitgenFormatter(use_colour=False))
        fitgen.addHandler(buffer_handler)


_configure_logging()
_ui_logger = logging.getLogger("fitgen.streamlit")
_fitgen_logger = logging.getLogger("fitgen")

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="FITGEN.AI — AI Fitness Coach",
    page_icon="🔥",
    layout="wide",
)

# ── Chat avatars (custom emojis, not Streamlit's default SVG icons) ──
AVATAR_USER = "🧑"
AVATAR_ASSISTANT = "🔥"

# ── Custom CSS: original orange-theme + avatar/status upgrades ──
st.markdown("""
<style>
/* ── Import Google Fonts (IBM Plex Mono as Berkeley Mono fallback) ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

/* ── Root variables — OpenCode dark palette ─────────── */
:root {
    --bg-dark: #201d1d;
    --bg-card: #302c2c;
    --bg-card-hover: #3a3636;
    --accent-orange: #007aff;              /* alias kept; now OpenCode accent blue */
    --accent-orange-glow: rgba(0, 122, 255, 0.25);
    --accent-red: #ff3b30;
    --accent-green: #30d158;
    --accent-warning: #ff9f0a;
    --text-primary: #fdfcfc;
    --text-secondary: #9a9898;
    --text-muted: #6e6e73;
    --border-subtle: rgba(15, 0, 0, 0.12);
    --border-strong: #646262;
    --gradient-hero: #201d1d;              /* flat — no gradients in OpenCode */
    --mono-stack: 'Berkeley Mono','IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;
}

/* ── Global overrides ───────────────────────────────── */
.stApp {
    background-color: var(--bg-dark) !important;
    font-family: var(--mono-stack) !important;
}
html, body, [class*="st-"], [data-testid="stAppViewContainer"] * {
    font-family: var(--mono-stack) !important;
}

/* ── Sidebar styling ────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-dark) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

section[data-testid="stSidebar"] h2 {
    color: var(--text-primary) !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}

/* ── Chat messages ──────────────────────────────────── */
.stChatMessage {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
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
    border-radius: 6px !important;
}

.stChatInput textarea {
    color: var(--text-primary) !important;
    font-family: var(--mono-stack) !important;
}

/* ── Buttons — single OpenCode variant ──────────────── */
.stButton > button {
    background: var(--bg-dark) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 4px !important;
    font-family: var(--mono-stack) !important;
    font-weight: 500 !important;
    padding: 4px 20px !important;
    transition: background 0.12s ease, border-color 0.12s ease !important;
}

.stButton > button:hover {
    background: var(--bg-card) !important;
    border-color: var(--text-secondary) !important;
    transform: none !important;
    box-shadow: none !important;
}

.stFormSubmitButton > button {
    background: var(--bg-dark) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 4px !important;
    font-family: var(--mono-stack) !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    padding: 4px 20px !important;
    letter-spacing: 0 !important;
    transition: background 0.12s ease, border-color 0.12s ease !important;
}

.stFormSubmitButton > button:hover {
    background: var(--bg-card) !important;
    border-color: var(--text-secondary) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Forms ──────────────────────────────────────────── */
[data-testid="stForm"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
    padding: 1.5rem !important;
}

/* ── Input fields ───────────────────────────────────── */
.stTextInput > div > div,
.stNumberInput > div > div,
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
}

.stTextInput input,
.stNumberInput input {
    color: var(--text-primary) !important;
    font-family: var(--mono-stack) !important;
}
.stTextInput > div > div:focus-within,
.stNumberInput > div > div:focus-within,
.stSelectbox > div > div:focus-within {
    border-color: #007aff !important;
}

/* ── Progress bar ───────────────────────────────────── */
.stProgress > div > div > div {
    background: #007aff !important;
    border-radius: 4px !important;
}

/* ── Expanders ──────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: 4px !important;
    color: var(--text-primary) !important;
}

/* ── Status widget ──────────────────────────────────── */
[data-testid="stStatusWidget"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
}

/* ── Dividers ───────────────────────────────────────── */
hr {
    border-color: var(--border-subtle) !important;
}

/* ── Custom hero section ────────────────────────────── */
.hero-container {
    background: var(--bg-dark);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.hero-title {
    font-family: var(--mono-stack);
    font-size: 2.38rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
    line-height: 1.5;
}

.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 1.25rem;
    max-width: 600px;
    font-family: var(--mono-stack);
}

.hero-tags {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.hero-tag {
    background: var(--bg-card);
    color: var(--text-primary);
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 0.82rem;
    font-weight: 500;
    border: 1px solid var(--border-subtle);
    font-family: var(--mono-stack);
}

/* ── Tool badges — flat single style ────────────────── */
.tool-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-primary);
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    margin-bottom: 8px;
    font-family: var(--mono-stack);
}

.tool-badge-workout,
.tool-badge-diet,
.tool-badge-general,
.tool-badge-rag { background: var(--bg-card); color: var(--text-primary); }

/* ── Sidebar stats cards ────────────────────────────── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}

.stat-card-label {
    color: var(--text-muted);
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: none;
    letter-spacing: 0;
    font-family: var(--mono-stack);
}

.stat-card-value {
    color: var(--text-primary);
    font-size: 1.25rem;
    font-weight: 700;
    font-family: var(--mono-stack);
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

.progress-dot-done     { background: #30d158; }
.progress-dot-active   { background: #007aff; }
.progress-dot-pending  { background: var(--text-muted); }

/* ── Metrics override ───────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
    padding: 0.75rem 1rem !important;
}

[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-family: var(--mono-stack) !important;
}

/* ── Scrollbar ──────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #555; }

/* ── AI Response Content Enhancement ───────────────── */

/* Section headings inside responses */
.stChatMessage h2, .stChatMessage h3, .stChatMessage h4 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.4rem !important;
    padding-bottom: 0.3rem !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    font-size: 1rem !important;
    letter-spacing: 0 !important;
    font-family: var(--mono-stack) !important;
}
.stChatMessage h2 { font-size: 1.1rem !important; }

/* Bold exercise / food names */
.stChatMessage strong {
    color: var(--text-primary) !important;
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
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 4px !important;
    transition: background 0.12s ease;
}
.stChatMessage ol > li:hover {
    background: var(--bg-card-hover) !important;
}
.stChatMessage ol > li::before {
    content: counter(item);
    position: absolute;
    left: 0.75rem;
    top: 0.7rem;
    width: 1.4rem;
    height: 1.4rem;
    background: var(--bg-dark);
    color: var(--text-primary);
    font-size: 0.75rem;
    font-weight: 700;
    border-radius: 4px;
    border: 1px solid var(--border-strong);
    display: flex;
    align-items: center;
    justify-content: center;
    line-height: 1;
    font-family: var(--mono-stack);
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

/* ── Tables inside chat messages ───────────────────── */
.stChatMessage table {
    width: 100% !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: 4px !important;
    overflow: hidden !important;
    margin: 0.8rem 0 1rem 0 !important;
    border: 1px solid var(--border-subtle) !important;
    font-size: 0.88rem !important;
    font-family: var(--mono-stack) !important;
}
.stChatMessage thead th {
    background: var(--bg-dark) !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    padding: 0.65rem 0.75rem !important;
    border-bottom: 1px solid var(--border-strong) !important;
    text-align: left !important;
    white-space: nowrap !important;
}
.stChatMessage tbody td {
    padding: 0.55rem 0.75rem !important;
    color: var(--text-secondary) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    vertical-align: top !important;
}
.stChatMessage tbody tr:hover td {
    background: var(--bg-card-hover) !important;
    color: var(--text-primary) !important;
}
.stChatMessage tbody td:first-child {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}
.stChatMessage tbody tr:last-child td {
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    border-top: 1px solid var(--border-strong) !important;
    background: var(--bg-dark) !important;
}

/* ── Blockquotes (tips / notes) ────────────────────── */
.stChatMessage blockquote {
    border-left: 2px solid var(--text-secondary) !important;
    background: var(--bg-dark) !important;
    margin: 0.75rem 0 !important;
    padding: 0.6rem 1rem !important;
    border-radius: 0 4px 4px 0 !important;
    color: var(--text-secondary) !important;
    font-size: 0.88rem !important;
}

/* ── Horizontal rules (section separators) ─────────── */
.stChatMessage hr {
    border: none !important;
    height: 1px !important;
    background: var(--border-subtle) !important;
    margin: 1.2rem 0 !important;
}

/* ── Inline code ───────────────────────────────────── */
.stChatMessage code:not(pre code) {
    background: var(--bg-dark) !important;
    color: var(--text-primary) !important;
    padding: 0.15rem 0.45rem !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border: 1px solid var(--border-subtle) !important;
    font-family: var(--mono-stack) !important;
}

/* Streaming cursor */
.streaming-cursor {
    display: inline-block;
    width: 2px;
    height: 1.1em;
    background: var(--text-primary);
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: blink 0.8s step-end infinite;
}
@keyframes blink {
    50% { opacity: 0; }
}

/* ── Profile card ──────────────────────────────────── */
.profile-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 0.85rem 0.8rem;
    margin-bottom: 0.25rem;
}
.profile-row {
    display: flex; align-items: center; gap: 0.7rem;
}
.profile-avatar-ring {
    position: relative;
    width: 44px; height: 44px;
    flex-shrink: 0;
}
.profile-avatar {
    position: relative;
    z-index: 1;
    width: 44px; height: 44px; border-radius: 4px;
    background: var(--bg-dark);
    border: 1px solid var(--border-strong);
    display: flex; align-items: center; justify-content: center;
    color: var(--text-primary); font-weight: 700; font-size: 1.15rem;
    font-family: var(--mono-stack);
}
.profile-status-dot {
    position: absolute;
    bottom: 1px; right: 1px;
    width: 10px; height: 10px;
    background: #30d158;
    border: 2px solid var(--bg-dark);
    border-radius: 50%;
    z-index: 2;
}
.profile-info {
    min-width: 0;
    flex: 1;
}
.profile-name {
    color: var(--text-primary); font-size: 0.9rem; font-weight: 700;
    line-height: 1.5;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    font-family: var(--mono-stack);
}
.profile-email {
    color: var(--text-muted); font-size: 0.75rem; line-height: 1.5;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    margin-top: 1px;
    font-family: var(--mono-stack);
}
.profile-badge {
    display: inline-block;
    margin-top: 4px;
    padding: 2px 8px;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0;
    text-transform: none;
    color: var(--text-primary);
    background: var(--bg-dark);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    font-family: var(--mono-stack);
}

/* ── Workout card ──────────────────────────────────── */
.workout-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.4rem;
}
.workout-card-session {
    color: var(--text-primary);
    font-size: 1rem; font-weight: 700;
    font-family: var(--mono-stack);
}
.workout-card-day {
    color: var(--text-muted);
    font-size: 0.75rem; text-transform: none;
    letter-spacing: 0; font-weight: 500;
    font-family: var(--mono-stack);
}
.workout-exercise {
    color: var(--text-secondary);
    font-size: 0.88rem; padding: 0.15rem 0;
    display: flex; justify-content: space-between;
    font-family: var(--mono-stack);
}
.workout-exercise-sets {
    color: var(--text-muted); font-size: 0.82rem;
}

/* ── Water tracker ─────────────────────────────────── */
.water-progress-bar {
    background: var(--bg-dark);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    height: 18px;
    overflow: hidden;
    margin: 0.4rem 0;
}
.water-progress-fill {
    height: 100%;
    background: #007aff;
    border-radius: 4px;
    transition: width 0.2s ease;
}
.water-label {
    color: var(--text-secondary);
    font-size: 0.82rem;
    display: flex; justify-content: space-between;
    align-items: center;
    font-family: var(--mono-stack);
}

/* ── Daily tip ─────────────────────────────────────── */
.tip-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.4rem;
}
.tip-text {
    color: var(--text-secondary);
    font-size: 0.88rem;
    line-height: 1.5;
    font-style: normal;
    font-family: var(--mono-stack);
}
.tip-label {
    color: var(--text-primary);
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: none;
    letter-spacing: 0;
    margin-bottom: 0.25rem;
    font-family: var(--mono-stack);
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

/* ═══════════════════════════════════════════════════ */
/* Chat input — kill baseweb tint, orange focus ring   */
/* ═══════════════════════════════════════════════════ */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div,
.stChatInput, .stChatInput > div, .stChatInput > div > div,
div[data-baseweb="textarea"],
div[data-baseweb="base-input"] {
    background: var(--bg-card) !important;
    background-color: var(--bg-card) !important;
    border-color: var(--border-subtle) !important;
}
[data-testid="stChatInput"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    padding: 2px !important;
    transition: border-color 0.12s ease !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #007aff !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea,
.stChatInput textarea {
    background: transparent !important;
    background-color: transparent !important;
    color: var(--text-primary) !important;
    caret-color: #007aff !important;
    font-size: 1rem !important;
    line-height: 1.5 !important;
    font-family: var(--mono-stack) !important;
}
[data-testid="stChatInput"] textarea::placeholder,
.stChatInput textarea::placeholder {
    color: var(--text-muted) !important;
    font-style: normal !important;
}
[data-testid="stChatInput"] button {
    background: var(--bg-dark) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 4px !important;
    box-shadow: none !important;
    transition: background 0.12s ease, border-color 0.12s ease !important;
}
[data-testid="stChatInput"] button:hover:not(:disabled) {
    background: var(--bg-card) !important;
    border-color: var(--text-secondary) !important;
    box-shadow: none !important;
    transform: none !important;
}
[data-testid="stChatInput"] button:hover:not(:disabled) svg {
    color: var(--text-primary) !important;
    fill: var(--text-primary) !important;
}
[data-testid="stChatInput"] button svg {
    color: var(--text-primary) !important;
    fill: var(--text-primary) !important;
}

/* ═══════════════════════════════════════════════════ */
/* Chat avatars — warm emoji tokens, animated          */
/* ═══════════════════════════════════════════════════ */
@keyframes avatarBreathe {
    0%, 100% { transform: scale(1);    filter: brightness(1.00) saturate(1.00); }
    50%      { transform: scale(1.10); filter: brightness(1.18) saturate(1.18); }
}
@keyframes avatarAssistantBob {
    0%, 100% { transform: translateY(0)    rotate(0deg); }
    25%      { transform: translateY(-2px) rotate(-4deg); }
    75%      { transform: translateY(-1px) rotate(4deg); }
}
@keyframes avatarUserSway {
    0%, 100% { transform: rotate(0deg); }
    50%      { transform: rotate(3deg); }
}
@keyframes avatarFlameRing {
    0%, 100% { box-shadow: 0 0 0 1px var(--border-strong); }
    50%      { box-shadow: 0 0 0 1px var(--text-secondary); }
}
@keyframes avatarUserRing {
    0%, 100% { box-shadow: 0 0 0 1px var(--border-subtle); }
    50%      { box-shadow: 0 0 0 1px var(--border-strong); }
}

.stChatMessage [data-testid*="hatAvatar"],
.stChatMessage [data-testid*="hatMessageAvatar"],
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    border-radius: 50% !important;
    background: var(--bg-card-hover) !important;
    width: 40px !important;
    height: 40px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 1.25rem !important;
    line-height: 1 !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
    overflow: visible !important;
}

.stChatMessage [data-testid="stChatMessageAvatarAssistant"],
.stChatMessage [data-testid="chatAvatarIcon-assistant"] {
    background: var(--bg-card) !important;
    animation: avatarFlameRing 2.6s ease-in-out infinite !important;
}

.stChatMessage [data-testid="stChatMessageAvatarUser"],
.stChatMessage [data-testid="chatAvatarIcon-user"] {
    background: var(--bg-card-hover) !important;
    animation: avatarUserRing 3.2s ease-in-out infinite !important;
}

.stChatMessage [data-testid*="vatar"] > * {
    display: inline-block !important;
    animation: avatarBreathe 2.8s ease-in-out infinite !important;
    filter: none;
}
.stChatMessage [data-testid="stChatMessageAvatarAssistant"] > *,
.stChatMessage [data-testid="chatAvatarIcon-assistant"] > * {
    animation: avatarBreathe 2.8s ease-in-out infinite,
               avatarAssistantBob 3.4s ease-in-out infinite !important;
    filter: none;
}
.stChatMessage [data-testid="stChatMessageAvatarUser"] > *,
.stChatMessage [data-testid="chatAvatarIcon-user"] > * {
    animation: avatarBreathe 3.0s ease-in-out infinite,
               avatarUserSway 4.0s ease-in-out infinite !important;
    filter: none;
}

.stChatMessage:hover [data-testid*="vatar"] {
    transform: scale(1.12) !important;
}

/* ═══════════════════════════════════════════════════ */
/* Status widget — rotating-word shimmer (orange)      */
/* ═══════════════════════════════════════════════════ */
@keyframes statusShimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
@keyframes statusWordIn {
    0%   { opacity: 0; transform: translateY(4px); }
    40%  { opacity: 1; transform: translateY(0); }
    100% { opacity: 1; transform: translateY(0); }
}

[data-testid="stStatusWidget"] summary > div,
[data-testid="stStatusWidget"] [data-testid="stMarkdownContainer"] p,
[data-testid="stStatusWidget"] label,
[data-testid="stStatusWidget"] span {
    background: linear-gradient(
        90deg,
        var(--text-muted) 0%,
        var(--text-primary) 50%,
        var(--text-muted) 100%
    ) !important;
    background-size: 200% auto !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    color: transparent !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    animation: statusShimmer 4s linear infinite, statusWordIn 0.5s ease-out !important;
}

[data-testid="stStatusWidget"] svg,
[data-testid="stStatusWidget"] [role="progressbar"] {
    color: var(--text-primary) !important;
    fill: var(--text-primary) !important;
}

[data-testid="stSpinner"] > div,
.stSpinner > div {
    background: linear-gradient(
        90deg,
        var(--text-muted) 0%,
        var(--text-primary) 50%,
        var(--text-muted) 100%
    ) !important;
    background-size: 200% auto !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: statusShimmer 4s linear infinite !important;
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

    # ── Calendar OAuth callback ─────────────────────────────────
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

    .stApp::before { content: none !important; }
    .stApp::after  { content: none !important; }

    /* ── Login: center column acts as card ────────────── */
    [data-testid="stAppViewContainer"] [data-testid="stVerticalBlock"] {
        max-width: 720px;
        margin: 0 auto;
        padding-top: 3vh;
        animation: fadeInUp 0.8s ease-out;
        position: relative;
        z-index: 1;
    }

    /* ── Login: Google button (OpenCode dark) ─────────── */
    a[data-testid="baseLinkButton-secondary"] {
        background: #201d1d !important;
        color: #fdfcfc !important;
        border: 1px solid #646262 !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        padding: 4px 20px !important;
        transition: background 0.12s ease, border-color 0.12s ease !important;
        box-shadow: none !important;
        text-decoration: none !important;
        font-family: 'Berkeley Mono','IBM Plex Mono',ui-monospace,monospace !important;
    }
    a[data-testid="baseLinkButton-secondary"]:hover {
        background: #302c2c !important;
        border-color: #9a9898 !important;
        transform: none !important;
        box-shadow: none !important;
    }
    a[data-testid="baseLinkButton-secondary"] p {
        color: #fdfcfc !important;
        font-weight: 500 !important;
    }

    /* ── Login: dev button ──────────────────────────── */
    button[data-testid="baseButton-secondary"] {
        background: #201d1d !important;
        color: #9a9898 !important;
        border: 1px solid rgba(15, 0, 0, 0.12) !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 4px 20px !important;
        transition: background 0.12s ease, border-color 0.12s ease !important;
        letter-spacing: 0 !important;
        font-family: 'Berkeley Mono','IBM Plex Mono',ui-monospace,monospace !important;
    }
    button[data-testid="baseButton-secondary"]:hover {
        border-color: #646262 !important;
        color: #fdfcfc !important;
        background: #302c2c !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Landing page CSS (animations + layout) ─────────────────────
    st.markdown("""
    <style>
    /* ── Staggered entrance keyframes ────────────────── */
    @keyframes landFadeUp {
        from { opacity: 0; transform: translateY(32px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes landScaleIn {
        from { opacity: 0; transform: scale(0.85); }
        to   { opacity: 1; transform: scale(1); }
    }
    @keyframes shimmerGradient {
        0%   { background-position: -200% center; }
        100% { background-position:  200% center; }
    }
    @keyframes floatIcon {
        0%, 100% { transform: translateY(0); }
        50%      { transform: translateY(-6px); }
    }
    @keyframes glowPulse {
        0%, 100% { box-shadow: none; }
        50%      { box-shadow: none; }
    }
    @keyframes drawArrow {
        from { opacity: 0; transform: translateX(-8px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes numPop {
        0%   { opacity: 0; transform: scale(0.5) rotate(-8deg); }
        60%  { transform: scale(1.12) rotate(2deg); }
        100% { opacity: 1; transform: scale(1) rotate(0deg); }
    }

    /* ── Logo ────────────────────────────────────────── */
    .landing-logo {
        width: 76px; height: 76px; margin: 0 auto 1.1rem auto;
        display: flex; align-items: center; justify-content: center;
        font-size: 2.3rem;
        background: transparent;
        border-radius: 4px;
        animation: landScaleIn 0.6s ease-out forwards;
        opacity: 0;
    }

    /* ── Title (flat monospace) ──────────────────────── */
    .landing-title {
        font-family: 'Berkeley Mono','IBM Plex Mono',ui-monospace,monospace;
        font-size: 2.38rem; font-weight: 700;
        text-align: center; letter-spacing: 0;
        margin-bottom: 0.2rem;
        color: #fdfcfc;
        line-height: 1.5;
        animation: landFadeUp 0.5s ease-out 0.1s forwards;
        opacity: 0;
    }

    /* ── Subtitle + tagline ─────────────────────────── */
    .landing-subtitle {
        text-align: center; color: #fdfcfc; font-size: 1rem;
        font-weight: 400; margin-bottom: 0.1rem;
        font-family: 'Berkeley Mono','IBM Plex Mono',ui-monospace,monospace;
        animation: landFadeUp 0.5s ease-out 0.2s forwards;
        opacity: 0;
    }
    .landing-tagline {
        text-align: center; color: #9a9898; font-size: 0.88rem;
        margin-bottom: 2rem;
        font-family: 'Berkeley Mono','IBM Plex Mono',ui-monospace,monospace;
        animation: landFadeUp 0.5s ease-out 0.3s forwards;
        opacity: 0;
    }

    /* ── Feature list (flat text — no cards) ────────── */
    .feature-row {
        display: flex; gap: 1.5rem; margin-bottom: 2rem;
        flex-direction: column;
    }
    .feature-card {
        flex: 1;
        background: transparent;
        border: none;
        border-top: 1px solid rgba(15, 0, 0, 0.12);
        border-radius: 0;
        padding: 1rem 0;
        text-align: left;
        opacity: 0;
        animation: landFadeUp 0.5s ease-out forwards;
        transition: none;
    }
    .feature-card:nth-child(1) { animation-delay: 0.4s; }
    .feature-card:nth-child(2) { animation-delay: 0.5s; }
    .feature-card:nth-child(3) { animation-delay: 0.6s; }
    .feature-card:hover {
        transform: none;
        border-color: rgba(15, 0, 0, 0.12);
        box-shadow: none;
    }
    .feature-icon {
        font-size: 1.2rem; margin-right: 0.6rem;
        display: inline-block;
        animation: none;
    }
    .feature-name {
        color: #fdfcfc; font-size: 1rem; font-weight: 700;
        margin-bottom: 0.3rem;
        font-family: 'Berkeley Mono','IBM Plex Mono',ui-monospace,monospace;
    }
    .feature-desc {
        color: #9a9898; font-size: 0.88rem; line-height: 1.5;
        font-family: 'Berkeley Mono','IBM Plex Mono',ui-monospace,monospace;
    }

    /* ── How it works (flat text) ───────────────────── */
    .how-section {
        background: transparent;
        border: none;
        border-top: 1px solid rgba(15, 0, 0, 0.12);
        border-bottom: 1px solid rgba(15, 0, 0, 0.12);
        border-radius: 0;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        text-align: left;
        opacity: 0;
        animation: landFadeUp 0.5s ease-out 0.7s forwards;
    }
    .how-label {
        color: #9a9898; font-size: 0.82rem; font-weight: 500;
        text-transform: none; letter-spacing: 0;
        margin-bottom: 1rem;
        font-family: 'Berkeley Mono','IBM Plex Mono',ui-monospace,monospace;
    }
    .how-row {
        display: flex; align-items: flex-start;
        justify-content: flex-start; gap: 0.5rem;
        flex-wrap: wrap;
    }
    .how-step { text-align: left; flex: 1; min-width: 160px; }
    .how-num {
        width: 28px; height: 28px; border-radius: 4px;
        background: #201d1d;
        border: 1px solid #646262;
        color: #fdfcfc;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 0.88rem; font-weight: 700;
        margin-bottom: 0.5rem;
        font-family: 'Berkeley Mono','IBM Plex Mono',ui-monospace,monospace;
        animation: landFadeUp 0.4s ease-out forwards;
        opacity: 0;
    }
    .how-step:nth-child(1) .how-num { animation-delay: 0.8s; }
    .how-step:nth-child(3) .how-num { animation-delay: 0.9s; }
    .how-step:nth-child(5) .how-num { animation-delay: 1.0s; }
    .how-step-title {
        color: #fdfcfc; font-size: 0.95rem; font-weight: 700;
        margin-bottom: 0.2rem;
        font-family: 'Berkeley Mono','IBM Plex Mono',ui-monospace,monospace;
    }
    .how-step-desc {
        color: #9a9898; font-size: 0.82rem;
        font-family: 'Berkeley Mono','IBM Plex Mono',ui-monospace,monospace;
    }
    .how-arrow {
        color: #9a9898; font-size: 1rem;
        padding-top: 0.2rem;
        opacity: 1;
        animation: none;
        font-family: 'Berkeley Mono','IBM Plex Mono',ui-monospace,monospace;
    }

    /* ── Stats bar ──────────────────────────────────── */
    .stats-row {
        display: flex; justify-content: space-between;
        margin-bottom: 2rem; padding: 0.3rem 0;
        gap: 1rem;
    }
    .stat-item {
        text-align: left;
        opacity: 0;
        animation: landFadeUp 0.4s ease-out forwards;
    }
    .stat-item:nth-child(1) { animation-delay: 1.1s; }
    .stat-item:nth-child(2) { animation-delay: 1.15s; }
    .stat-item:nth-child(3) { animation-delay: 1.2s; }
    .stat-item:nth-child(4) { animation-delay: 1.25s; }
    .stat-num {
        color: #fdfcfc; font-size: 1.5rem; font-weight: 700;
        font-style: normal; line-height: 1.2;
        font-family: 'Berkeley Mono','IBM Plex Mono',ui-monospace,monospace;
    }
    .stat-label {
        color: #9a9898; font-size: 0.82rem; font-weight: 400;
        text-transform: none; letter-spacing: 0;
        margin-top: 0.2rem;
        font-family: 'Berkeley Mono','IBM Plex Mono',ui-monospace,monospace;
    }

    /* ── CTA card ───────────────────────────────────── */
    .cta-card {
        background: transparent;
        border: 1px solid rgba(15, 0, 0, 0.12);
        border-radius: 4px;
        padding: 1.2rem 1.2rem;
        text-align: left;
        margin-bottom: 1rem;
        opacity: 0;
        animation: landFadeUp 0.5s ease-out 1.3s forwards;
    }
    .cta-title {
        color: #fdfcfc; font-size: 1rem; font-weight: 700;
        margin-bottom: 0.3rem;
        font-family: 'Berkeley Mono','IBM Plex Mono',ui-monospace,monospace;
    }
    .cta-desc {
        color: #9a9898; font-size: 0.88rem;
        font-family: 'Berkeley Mono','IBM Plex Mono',ui-monospace,monospace;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Logo + Title ──
    st.markdown(
        '<div class="landing-logo">&#x1F525;</div>'
        '<div class="landing-title">FITGEN.AI</div>'
        '<div class="landing-subtitle">Your AI-Powered Personal Fitness Coach</div>'
        '<div class="landing-tagline">Train smarter. Eat better. Track everything.</div>',
        unsafe_allow_html=True,
    )

    # ── Feature cards ──
    st.markdown(
        '<div class="feature-row">'
        '  <div class="feature-card">'
        '    <div class="feature-icon">💪</div>'
        '    <div class="feature-name">AI Workout Plans</div>'
        '    <div class="feature-desc">Periodized programs tailored to your goals, equipment &amp; schedule</div>'
        '  </div>'
        '  <div class="feature-card">'
        '    <div class="feature-icon">🥦</div>'
        '    <div class="feature-name">Smart Nutrition</div>'
        '    <div class="feature-desc">7-day meal plans with macro targets, calorie calculations &amp; snack swaps</div>'
        '  </div>'
        '  <div class="feature-card">'
        '    <div class="feature-icon">&#x1F4C5;</div>'
        '    <div class="feature-name">Google Sync</div>'
        '    <div class="feature-desc">Push meals and workouts to Google Calendar automatically</div>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── How it works ──
    st.markdown(
        '<div class="how-section">'
        '  <div class="how-label">HOW IT WORKS</div>'
        '  <div class="how-row">'
        '    <div class="how-step">'
        '      <div class="how-num">1</div>'
        '      <div class="how-step-title">Share Your Goals</div>'
        '      <div class="how-step-desc">Tell us about yourself</div>'
        '    </div>'
        '    <div class="how-arrow">&rarr;</div>'
        '    <div class="how-step">'
        '      <div class="how-num">2</div>'
        '      <div class="how-step-title">AI Builds Your Plan</div>'
        '      <div class="how-step-desc">Personalized in seconds</div>'
        '    </div>'
        '    <div class="how-arrow">&rarr;</div>'
        '    <div class="how-step">'
        '      <div class="how-num">3</div>'
        '      <div class="how-step-title">Sync &amp; Crush It</div>'
        '      <div class="how-step-desc">Google Calendar sync</div>'
        '    </div>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Stats bar ──
    st.markdown(
        '<div class="stats-row">'
        '  <div class="stat-item"><div class="stat-num">6</div><div class="stat-label">AI Techniques</div></div>'
        '  <div class="stat-item"><div class="stat-num">24+</div><div class="stat-label">Profile Fields</div></div>'
        '  <div class="stat-item"><div class="stat-num">7-Day</div><div class="stat-label">Meal &amp; Workout Plans</div></div>'
        '  <div class="stat-item"><div class="stat-num">100%</div><div class="stat-label">Personalized</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── CTA card ──
    st.markdown(
        '<div class="cta-card">'
        '  <div class="cta-title">Ready to transform your fitness?</div>'
        '  <div class="cta-desc">Sign in to get your personalized AI coaching experience</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Google sign-in button ──
    if _google_auth.is_configured:
        _login_url = _google_auth.get_login_url()
        st.link_button(
            "\U0001F680 Get Started with Google",
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
        'margin-top:1.2rem;letter-spacing:0.01em;">'
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
    # Always start a fresh session — each login gets a clean workflow.
    # Only the user profile is carried over from MongoDB (not old sessions).
    context_id = str(uuid.uuid4())

    # Load existing user profile from MongoDB (do NOT create user doc here —
    # the user doc is only created on plan confirm in the tool handlers).
    user_id = ""
    mongo_profile: dict = {}
    if user_email:
        existing_user = UserRepository.find_by_email(user_email)
        if existing_user:
            user_id = str(existing_user["_id"])
            mongo_profile = UserRepository.get_merged_profile(user_email)

    st.session_state.agent_state = {
        "messages": [],
        "user_profile": mongo_profile,          # MongoDB profile only — no stale session data
        "user_email": user_email or "",
        "user_id": user_id,
        "context_id": context_id,
        "state_id": context_id,
        "workflow": {},                          # Always fresh — no stale workflow
        "calendar_sync_requested": False,
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


@st.cache_data(ttl=30, show_spinner=False)
def _get_all_confirmed_diet_plans(user_id: str) -> list[dict]:
    """Fetch all confirmed diet plans for a user (cached 30 s)."""
    if not user_id:
        return []
    all_plans = DietPlanRepository.find_all_by_user(user_id, limit=20)
    confirmed = [p for p in all_plans if p.get("status") == "confirmed"]
    # Convert ObjectId → str so st.cache_data can serialize safely
    for p in confirmed:
        if "_id" in p:
            p["_id"] = str(p["_id"])
        if "user_id" in p:
            p["user_id"] = str(p["user_id"])
    return confirmed


@st.cache_data(ttl=30, show_spinner=False)
def _get_all_confirmed_workout_plans(user_id: str) -> list[dict]:
    """Fetch all confirmed workout plans for a user (cached 30 s)."""
    if not user_id:
        return []
    all_plans = WorkoutPlanRepository.find_all_by_user(user_id, limit=20)
    confirmed = [p for p in all_plans if p.get("status") == "confirmed"]
    for p in confirmed:
        if "_id" in p:
            p["_id"] = str(p["_id"])
        if "user_id" in p:
            p["user_id"] = str(p["user_id"])
    return confirmed


def _plan_label(plan: dict, domain: str) -> str:
    """Build a human-readable label for a plan in a selectbox."""
    emoji = "🥗" if domain == "diet" else "💪"

    # Prefer the stored plan name (LLM-generated on confirm)
    name = plan.get("name", "")
    if name:
        return f"{emoji} {name}"

    # Fallback: date + goal
    created = plan.get("created_at")
    date_str = created.strftime("%b %d, %Y") if created else "Unknown date"
    profile = plan.get("profile_snapshot", {})
    goal = profile.get("primary_goal") or profile.get("goal") or ""
    goal_str = f" — {goal.title()}" if goal else ""
    return f"{emoji} {date_str}{goal_str}"


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


def _strip_latex(text: str) -> str:
    """Remove common LaTeX artifacts that LLMs sometimes produce.

    Converts \\text{FOO} → FOO, \\textbf{FOO} → **FOO**,
    \\frac{a}{b} → a/b, \\times → ×, and strips stray $ delimiters
    wrapping non-math text.
    """
    import re as _re
    # \textbf{...} → **...**
    text = _re.sub(r"\\textbf\{([^}]+)\}", r"**\1**", text)
    # \text{...} → ...
    text = _re.sub(r"\\text\{([^}]+)\}", r"\1", text)
    # \frac{a}{b} → a/b
    text = _re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", text)
    # \times → ×
    text = text.replace("\\times", "×")
    # \approx → ≈
    text = text.replace("\\approx", "≈")
    # Strip display math blocks: \[ ... \] → content
    text = _re.sub(r"\\\[\s*", "", text)
    text = _re.sub(r"\s*\\\]", "", text)
    # Strip inline math $...$ that wraps non-math text (heuristic: if content has letters)
    text = _re.sub(r"\$([^$]{1,80})\$", r"\1", text)
    return text


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


def _render_profile_confirm_form(domain: str, existing_profile: dict) -> dict | None:
    """Render a Streamlit form for profile confirmation/editing.

    Shows ALL fields for the domain, **pre-filled** with existing values.
    The user can edit any field and click 'Confirm & Generate Plan'.
    Returns the merged profile on submit, else None.
    """
    all_fields = DOMAIN_REQUIRED_FIELDS.get(domain, list(PROFILE_FORM_FIELDS.keys()))
    tool_name = "diet_tool" if domain == "diet" else "workout_tool"

    st.markdown(_badge(tool_name), unsafe_allow_html=True)
    st.markdown("**Review & edit your profile, then confirm:**")

    _OPTIONAL_FIELDS = {"additional_info", "favourite_meals"}
    _show_fields = [f for f in all_fields if f in PROFILE_FORM_FIELDS]
    _regular_fields = [f for f in _show_fields if PROFILE_FORM_FIELDS[f].get("type") != "textarea"]
    _fullwidth_fields = [f for f in _show_fields if PROFILE_FORM_FIELDS[f].get("type") == "textarea"]

    _form_key = f"profile_confirm_{domain}"
    with st.form(_form_key, clear_on_submit=False):
        form_values: dict = {}
        col1, col2 = st.columns(2)

        for i, field in enumerate(_regular_fields):
            cfg = PROFILE_FORM_FIELDS.get(field)
            if not cfg:
                continue
            target_col = col1 if i % 2 == 0 else col2
            _wkey = f"confirm_{domain}_{field}"
            _existing_val = existing_profile.get(field)

            with target_col:
                if cfg["type"] == "text":
                    form_values[field] = st.text_input(
                        cfg["label"],
                        value=str(_existing_val) if _existing_val else "",
                        key=_wkey,
                    )
                elif cfg["type"] == "number":
                    _num_val = _existing_val
                    if _num_val is not None:
                        try:
                            _num_val = type(cfg.get("default", cfg["min"]))(_num_val)
                        except (ValueError, TypeError):
                            _num_val = cfg.get("default", cfg["min"])
                    else:
                        _num_val = cfg.get("default", cfg["min"])
                    # Clamp to valid range
                    _num_val = max(cfg["min"], min(cfg["max"], _num_val))
                    form_values[field] = st.number_input(
                        cfg["label"],
                        min_value=cfg["min"],
                        max_value=cfg["max"],
                        value=_num_val,
                        step=cfg["step"],
                        key=_wkey,
                    )
                elif cfg["type"] == "selectbox":
                    _opts = cfg["options"]
                    _idx = 0
                    if _existing_val:
                        _ev_lower = str(_existing_val).strip().lower()
                        for j, opt in enumerate(_opts):
                            if opt.lower() == _ev_lower:
                                _idx = j
                                break
                    form_values[field] = st.selectbox(
                        cfg["label"],
                        options=_opts,
                        index=_idx,
                        key=_wkey,
                    )

        for field in _fullwidth_fields:
            cfg = PROFILE_FORM_FIELDS.get(field)
            if not cfg:
                continue
            _wkey = f"confirm_{domain}_{field}"
            _existing_val = existing_profile.get(field)
            form_values[field] = st.text_area(
                cfg["label"],
                value=str(_existing_val) if _existing_val else "",
                height=100,
                key=_wkey,
            )

        submitted = st.form_submit_button(
            "✅ Confirm & Generate Plan",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        # Validate required fields
        _required_fields = [f for f in _show_fields if f not in _OPTIONAL_FIELDS]
        _required_missing = [
            f for f in _required_fields
            if not form_values.get(f)
        ]
        if _required_missing:
            labels = [PROFILE_FORM_FIELDS[f]["label"] for f in _required_missing if f in PROFILE_FORM_FIELDS]
            st.warning(f"Please fill in: {', '.join(labels)}")
            return None
        for f in _OPTIONAL_FIELDS:
            if f in form_values and isinstance(form_values[f], str) and not form_values[f].strip():
                form_values[f] = "none"
        merged = dict(existing_profile)
        merged.update(form_values)
        return merged

    return None


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
        '<span style="font-size:1.4rem;font-weight:700;color:#fdfcfc;'
        'font-family:\'Berkeley Mono\',\'IBM Plex Mono\',ui-monospace,monospace;'
        'letter-spacing:-0.01em;">'
        'FITGEN.AI</span></div>',
        unsafe_allow_html=True,
    )

# ── Sidebar ───────────────────────────────────────────────────────

with st.sidebar:
    # ── Branding ───────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;padding:0.5rem 0 1rem 0;">'
        '<span style="font-size:1.6rem;font-weight:700;color:#fdfcfc;'
        'font-family:\'Berkeley Mono\',\'IBM Plex Mono\',ui-monospace,monospace;'
        'letter-spacing:-0.01em;">'
        'FITGEN.AI</span><br>'
        '<span style="color:#6e6e73;font-size:0.7rem;letter-spacing:0.1em;text-transform:uppercase;'
        'font-family:\'Berkeley Mono\',\'IBM Plex Mono\',ui-monospace,monospace;">'
        'AI Fitness Coach</span></div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Profile dropdown ─────────────────────────────────────
    _sidebar_name = st.session_state.get("auth_user_name", "")
    _sidebar_email = st.session_state.get("auth_user_email", "")
    if _sidebar_email:
        _display = _sidebar_name or _sidebar_email.split("@")[0]
        _initial = _display[0].upper() if _display else "U"

        # Avatar + name card
        st.markdown(
            f'<div class="profile-card">'
            f'<div class="profile-row">'
            f'<div class="profile-avatar-ring">'
            f'<div class="profile-avatar">{_initial}</div>'
            f'<div class="profile-status-dot"></div>'
            f'</div>'
            f'<div class="profile-info">'
            f'<div class="profile-name">{_display}</div>'
            f'<div class="profile-email">{_sidebar_email}</div>'
            f'<div class="profile-badge">✦ Member</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Fetch user plans for sidebar
        _pop_user_id = _resolve_user_id()

        with st.popover("Menu", use_container_width=True):
            st.caption(f"Signed in as **{_display}**")

            # Always show Create buttons
            if st.button("🥗 Create Diet Plan", key="pop_create_diet", use_container_width=True):
                st.session_state._quick_prompt = "Create a diet plan"
                st.rerun()
            if st.button("💪 Create Workout Plan", key="pop_create_workout", use_container_width=True):
                st.session_state._quick_prompt = "Create a workout plan"
                st.rerun()

            st.divider()
            if st.button("🚪 Logout", key="pop_logout", use_container_width=True):
                for _k in ("authenticated", "auth_user_email", "auth_user_name",
                            "auth_user_picture", "agent_state", "chat_history",
                            "profile_form_pending", "graph", "water_glasses"):
                    st.session_state.pop(_k, None)
                st.rerun()

        # ── My Plans section ─────────────────────────────────────
        _all_diet_plans = _get_all_confirmed_diet_plans(_pop_user_id) if _pop_user_id else []
        _all_workout_plans = _get_all_confirmed_workout_plans(_pop_user_id) if _pop_user_id else []

        if _all_diet_plans or _all_workout_plans:
            st.markdown("## 📋 My Plans")

            # ── Diet plans selector ──
            if _all_diet_plans:
                _diet_labels = [_plan_label(p, "diet") for p in _all_diet_plans]
                _diet_idx = st.selectbox(
                    "Diet Plans",
                    range(len(_diet_labels)),
                    format_func=lambda i: _diet_labels[i],
                    key="diet_plan_selector",
                )
                if st.button("View Diet Plan", key="view_diet_plan", use_container_width=True):
                    # Fetch by _id for guaranteed correctness
                    _sel = _all_diet_plans[_diet_idx]
                    _plan_id = _sel.get("_id")
                    _plan_doc = DietPlanRepository.find_by_id(_plan_id) if _plan_id else _sel
                    _plan_md = (_plan_doc or {}).get("plan_markdown", "")
                    if _plan_md:
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": f"Show me my diet plan: {_diet_labels[_diet_idx]}",
                        })
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": _plan_md,
                            "tool_used": "diet_tool",
                            "show_pdf": True,
                        })
                        st.rerun()

            # ── Workout plans selector ──
            if _all_workout_plans:
                _wk_labels = [_plan_label(p, "workout") for p in _all_workout_plans]
                _wk_idx = st.selectbox(
                    "Workout Plans",
                    range(len(_wk_labels)),
                    format_func=lambda i: _wk_labels[i],
                    key="workout_plan_selector",
                )
                if st.button("View Workout Plan", key="view_workout_plan", use_container_width=True):
                    _sel = _all_workout_plans[_wk_idx]
                    _plan_id = _sel.get("_id")
                    _plan_doc = WorkoutPlanRepository.find_by_id(_plan_id) if _plan_id else _sel
                    _plan_md = (_plan_doc or {}).get("plan_markdown", "")
                    if _plan_md:
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": f"Show me my workout plan: {_wk_labels[_wk_idx]}",
                        })
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": _plan_md,
                            "tool_used": "workout_tool",
                            "show_pdf": True,
                        })
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
            "calendar_sync_started",
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
            _style = "color:#30d158;"
        elif _s_key == _step or _s_key in _step:
            _dot = "progress-dot-active"
            _style = "color:#007aff;font-weight:600;"
        else:
            _dot = "progress-dot-pending"
            _style = "color:#6e6e73;"
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

    st.divider()

    st.markdown(
        '<div style="text-align:center;padding:1rem 0;color:#444;font-size:0.65rem;">'
        'Built with LangGraph · LangChain · OpenAI · Streamlit</div>',
        unsafe_allow_html=True,
    )

# ── Render existing chat history ──────────────────────────────────

# A4 content width in pt: 595pt - 2 × ~45pt margins ≈ 505pt; round down
# to 480pt for padding/border slack.
_PDF_CONTENT_WIDTH_PT = 480

# Percentage column widths for tables of each column count.  Column 2
# (the "description" column — Foods, Exercise, etc.) always gets the
# biggest share so xhtml2pdf never collapses it.
_PDF_COL_WIDTHS = {
    2: [35, 65],
    3: [22, 55, 23],
    4: [18, 44, 19, 19],
    5: [15, 48, 13, 12, 12],
    6: [13, 50, 10, 9, 9, 9],
    7: [11, 47, 9, 9, 8, 8, 8],
}

_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)
_TABLE_OPEN_RE = re.compile(r"<table\b[^>]*>", re.IGNORECASE)
_TH_RE = re.compile(r"<th\b[^>]*>", re.IGNORECASE)
_TD_RE = re.compile(r"<td\b[^>]*>", re.IGNORECASE)
_FIRST_TR_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)


def _widths_for(n_cols: int) -> list[int]:
    if n_cols in _PDF_COL_WIDTHS:
        return _PDF_COL_WIDTHS[n_cols]
    if n_cols <= 1:
        return [100]
    rest = (100 - 13 - 50) // (n_cols - 2)
    w = [13, 50] + [rest] * (n_cols - 2)
    w[-1] += 100 - sum(w)
    return w


def _inject_table_widths(html: str) -> str:
    """For every <table>, inject a <colgroup> with absolute pt widths
    and set `table-layout: fixed` on the table tag.  This is the only
    way xhtml2pdf will reliably size wide tables — without it, the
    reportlab table engine content-guesses and collapses long-text
    columns to ~1 character wide."""
    def repl(m: re.Match) -> str:
        table = m.group(0)
        first_tr = _FIRST_TR_RE.search(table)
        if not first_tr:
            return table
        first_row = first_tr.group(1)
        n_cols = len(_TH_RE.findall(first_row)) or len(_TD_RE.findall(first_row))
        if n_cols < 2:
            return table

        pcts = _widths_for(n_cols)
        cols_html = "".join(
            f'<col style="width:{p}%; width:{int(round(_PDF_CONTENT_WIDTH_PT * p / 100))}pt" '
            f'width="{int(round(_PDF_CONTENT_WIDTH_PT * p / 100))}"/>'
            for p in pcts
        )
        colgroup = f"<colgroup>{cols_html}</colgroup>"

        def open_sub(_om: re.Match) -> str:
            return (
                f'<table width="{_PDF_CONTENT_WIDTH_PT}" '
                f'style="table-layout:fixed;width:{_PDF_CONTENT_WIDTH_PT}pt;'
                f'border-collapse:collapse;">'
                + colgroup
            )

        return _TABLE_OPEN_RE.sub(open_sub, table, count=1)

    return _TABLE_RE.sub(repl, html)


_WKHTMLTOPDF_BIN = "/opt/homebrew/bin/wkhtmltopdf"
_PDFKIT_CONFIG = (
    pdfkit.configuration(wkhtmltopdf=_WKHTMLTOPDF_BIN)
    if os.path.exists(_WKHTMLTOPDF_BIN)
    else None
)


@st.cache_data(show_spinner=False)
def _response_to_pdf(content: str) -> bytes:
    """markdown → HTML → PDF bytes, rendered as shown in the response.
    Uses wkhtmltopdf (WebKit) which renders wide tables across page
    breaks correctly — xhtml2pdf's reportlab engine collapses long-text
    columns on continuation pages."""
    html_body = _md.markdown(
        content or "",
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><style>"
        "body{font-family:Helvetica,Arial,sans-serif;font-size:11pt;color:#222;line-height:1.45;}"
        "h1{color:#d03050;}"
        "h2{color:#ff6a3d;margin-top:18px;}"
        "h3{color:#ff6a3d;}"
        "table{border-collapse:collapse;margin:8px 0;font-size:9.5pt;width:100%;}"
        "th,td{border:1px solid #e0e0e0;padding:6px 8px;text-align:left;vertical-align:top;"
        "word-wrap:break-word;overflow-wrap:break-word;}"
        "th{background:#ff6a3d;color:#fff;font-weight:bold;border-color:#ff6a3d;}"
        "tr:nth-child(even) td{background:#fafafa;}"
        "blockquote{background:#fff4e8;border-left:4px solid #ff6a3d;padding:6px 10px;}"
        "</style></head><body>" + html_body + "</body></html>"
    )
    options = {
        "page-size": "A4",
        "margin-top": "16mm",
        "margin-bottom": "16mm",
        "margin-left": "12mm",
        "margin-right": "12mm",
        "encoding": "UTF-8",
        "quiet": "",
    }
    return pdfkit.from_string(doc, False, options=options, configuration=_PDFKIT_CONFIG)


for i, entry in enumerate(st.session_state.chat_history):
    _entry_avatar = AVATAR_ASSISTANT if entry["role"] == "assistant" else AVATAR_USER
    with st.chat_message(entry["role"], avatar=_entry_avatar):
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
                if entry.get("show_pdf"):
                    _pdf_src = entry.get("pdf_content") or entry["content"]
                    try:
                        st.download_button(
                            "⬇️ Download as PDF",
                            data=_response_to_pdf(_pdf_src),
                            file_name=f"fitgen_plan_{i}.pdf",
                            mime="application/pdf",
                            key=f"dl_pdf_{i}",
                        )
                    except Exception:  # noqa: BLE001
                        pass
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
        with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
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

# ── Profile confirmation form (editable, pre-filled with existing data) ──

if _current_step == "user_profile_mapped" and st.session_state.get("profile_confirm_pending"):
    _confirm_domain = _current_workflow.get("domain", "diet")
    _confirm_existing = st.session_state.agent_state.get("user_profile", {})

    with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
        _confirm_data = _render_profile_confirm_form(_confirm_domain, _confirm_existing)

    if _confirm_data is not None:
        st.session_state.pop("profile_confirm_pending", None)
        # Update profile with confirmed/edited values
        st.session_state.agent_state["user_profile"].update(_confirm_data)
        # Send "yes" (confirmed) as the user message so the tool generates the plan.
        # The tool's Step C will extract any profile tweaks from ctx.profile.
        st.session_state._pending_form_message = "yes"
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
        with st.chat_message("user", avatar=AVATAR_USER):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

    # ── Append to agent state ─────────────────────────────────────
    st.session_state.agent_state["messages"].append(HumanMessage(content=prompt))

    # ── Stream agent response ─────────────────────────────────────
    with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
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
        _tool_structured_data: dict = {}  # macros/hydration from tool response
        # Multi-tool support: when the router dispatches "both" (diet +
        # workout in one turn), we receive multiple ToolMessages. Track
        # each tool_call_id → its parsed assistant_message so we can
        # stitch them into a single combined response at the end.
        _tool_call_map: dict[str, str] = {}   # call_id → tool_name
        _tool_responses: dict[str, str] = {}  # call_id → assistant_message
        _is_multi_tool = False

        # Rotating status words — Claude-Code style: single gerunds,
        # shuffled per-phase, cycled every ~1.1 s by a background thread.
        import threading, time as _time, itertools, random
        _STATUS_THINKING = itertools.cycle([
            "Thinking", "Pondering", "Reading", "Parsing",
            "Weighing", "Considering", "Reasoning", "Listening",
        ])
        _STATUS_TOOL: dict[str, list[str]] = {
            "workout_tool": [
                "Designing", "Sculpting", "Programming", "Periodizing",
                "Balancing", "Sequencing", "Tailoring", "Calibrating",
                "Lifting", "Assembling", "Structuring",
            ],
            "diet_tool": [
                "Composing", "Plating", "Balancing", "Portioning",
                "Tallying", "Seasoning", "Calibrating", "Layering",
                "Macroing", "Nourishing", "Tailoring",
            ],
        }
        _STATUS_DEFAULT_TOOL = [
            "Consulting", "Researching", "Synthesizing",
            "Analyzing", "Deliberating",
        ]
        _STATUS_GENERATING = [
            "Crafting", "Composing", "Polishing", "Refining",
            "Weaving", "Finishing", "Tightening", "Phrasing",
        ]

        _status_phase = "thinking"   # thinking → tool → generating
        _status_stop = threading.Event()

        def _rotate_status(st_status):
            """Background thread: rotate one-word status every ~2.2 s."""
            _tool_cycle = None
            _gen_cycle = itertools.cycle(_STATUS_GENERATING)
            while not _status_stop.is_set():
                _status_stop.wait(2.2)
                if _status_stop.is_set():
                    break
                try:
                    if _status_phase == "thinking":
                        _word = next(_STATUS_THINKING)
                    elif _status_phase == "tool":
                        if _tool_cycle is None:
                            phrases = list(_STATUS_TOOL.get(tool_used, _STATUS_DEFAULT_TOOL))
                            random.shuffle(phrases)
                            _tool_cycle = itertools.cycle(phrases)
                        _word = next(_tool_cycle)
                    elif _status_phase == "generating":
                        _word = next(_gen_cycle)
                    else:
                        continue
                    # Trailing ellipsis — single char for Claude-style cadence
                    st_status.update(label=f"{_word}…")
                except Exception:
                    pass

        try:
          with st.status("Thinking…", expanded=False) as status:
            # CRITICAL: attach ScriptRunContext so status.update() calls from
            # the background thread actually reach the frontend — without this
            # Streamlit silently drops widget mutations made off the main thread.
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            _rotator = threading.Thread(target=_rotate_status, args=(status,), daemon=True)
            add_script_run_ctx(_rotator)
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
                        _calls = last_msg.tool_calls
                        _is_multi_tool = len(_calls) > 1
                        # Record call_id → tool_name for stitching later
                        for _tc in _calls:
                            _cid = _tc.get("id") or ""
                            _cname = _tc.get("name") or ""
                            if _cid and _cname:
                                _tool_call_map[_cid] = _cname
                        tool_used = _calls[0]["name"]
                        if _is_multi_tool:
                            _status_phase = "tool"
                            _tool_names = [c["name"] for c in _calls]
                            _labels = [
                                TOOL_LABELS.get(n, ("Specialist", "", "#333"))[0]
                                for n in _tool_names
                            ]
                            status.update(label=f"Consulting {' + '.join(_labels)}…")
                            # Render both badges side-by-side
                            badge_placeholder.markdown(
                                " ".join(_badge(n) for n in _tool_names),
                                unsafe_allow_html=True,
                            )
                            _ui_logger.info(
                                "[Turn %s] Multi-tool dispatch: %s",
                                turn_id, _tool_names,
                            )
                        else:
                            tool_label, _, _ = TOOL_LABELS.get(tool_used, ("Specialist", "", "#333"))
                            _status_phase = "tool"
                            status.update(label=f"Consulting {tool_label}…")
                            badge_placeholder.markdown(_badge(tool_used), unsafe_allow_html=True)
                            _ui_logger.info("[Turn %s] Routed to tool=%s", turn_id, tool_used)

                    # Parse ToolMessage JSON → user-facing assistant message
                    from langchain_core.messages import ToolMessage
                    if isinstance(last_msg, ToolMessage) and last_msg.content:
                        _status_phase = "generating"
                        status.update(label="Crafting…")
                        try:
                            parsed = json.loads(last_msg.content)
                            # Check if workflow entered profile collection — form will handle display
                            _tool_state = parsed.get("state_updates", {})
                            _tool_wf = _tool_state.get("workflow", {})
                            _tool_step = _tool_wf.get("step_completed") or _tool_wf.get("stage")
                            if _tool_step in ("prompted_for_user_profile_data", "user_profile_mapped"):
                                _form_will_render = True
                            # Capture structured_data for macro chart
                            _sd = _tool_wf.get("structured_data", {})
                            if _sd:
                                _tool_structured_data = _sd

                            assistant_message = parsed.get("assistant_message")
                            if assistant_message:
                                if _is_multi_tool:
                                    # Record by call_id and stitch on every update
                                    _cid = getattr(last_msg, "tool_call_id", "") or ""
                                    if _cid:
                                        _tool_responses[_cid] = assistant_message
                                    # Stitch in dispatch order so sections are
                                    # stable (diet_tool first, workout_tool second)
                                    _sections: list[str] = []
                                    for _cid_k, _tname in _tool_call_map.items():
                                        _resp = _tool_responses.get(_cid_k)
                                        if not _resp:
                                            continue
                                        _tlabel, _, _ = TOOL_LABELS.get(
                                            _tname, ("Specialist", "", "#333"),
                                        )
                                        _sections.append(f"### {_tlabel}\n\n{_resp}")
                                    if _sections:
                                        response_content = "\n\n---\n\n".join(_sections)
                                        tool_direct_reply = True
                                        _ui_logger.debug(
                                            "[Turn %s] Multi-tool stitched response "
                                            "(tools=%d, chars=%d)",
                                            turn_id, len(_tool_responses),
                                            len(response_content),
                                        )
                                elif assistant_message != response_content:
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
            status.update(label="Done!", state="complete")
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
            response_content = _strip_latex(response_content)
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

                # NOTE: Macro donut + macro cards are rendered in the
                # LEFT SIDEBAR (see the "Macro Distribution" block in
                # the sidebar render) after the plan is confirmed.
                # We intentionally do NOT duplicate them inline in the
                # assistant response — the LLM's response should stay
                # textual / tabular only.

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
    _synced_domain = _synced_wf.get("domain", "")
    _prev_form_domain = st.session_state.get("_profile_form_domain", "")

    if _synced_step == "prompted_for_user_profile_data":
        # Rerun if form not yet pending, OR if domain switched (e.g. workout → diet)
        if not st.session_state.profile_form_pending or _synced_domain != _prev_form_domain:
            st.session_state.profile_form_pending = True
            st.session_state._profile_form_domain = _synced_domain
            st.rerun()
    elif _synced_step == "user_profile_mapped":
        # Profile confirmation — show editable form instead of markdown table
        if not st.session_state.get("profile_confirm_pending"):
            st.session_state.profile_confirm_pending = True
            st.session_state._profile_form_domain = _synced_domain
            st.rerun()
    else:
        st.session_state.profile_form_pending = False
        st.session_state.pop("profile_confirm_pending", None)
        st.session_state._profile_form_domain = ""

    # ── Persist to display history ────────────────────────────────
    # Skip persisting when the form will take over (profile questions
    # are handled by the form widget, not the chat history).
    if not _form_will_render:
        # PDF download should appear only when the user has confirmed a plan
        # (or it's already synced) OR when the user explicitly retrieved a
        # plan via get_diet / get_workout. Everything else (profile intake,
        # draft plans awaiting confirmation, follow-up Q&A, updates, sync
        # prompts, refusals) should NOT show the PDF button.
        _wf_for_pdf = st.session_state.agent_state.get("workflow", {}) or {}
        _step_for_pdf = _wf_for_pdf.get("step_completed") or ""
        _intent_for_pdf = _wf_for_pdf.get("intent") or ""
        _domain_for_pdf = _wf_for_pdf.get("domain") or ""
        _show_pdf = (
            _step_for_pdf in {"diet_confirmed", "workout_confirmed"}
            or "synced" in _step_for_pdf
            or _intent_for_pdf in {"get_diet", "get_workout"}
        )
        # The response_content at confirm/sync time is a short ack
        # ("Your plan is confirmed!"), not the plan itself. Resolve the
        # actual plan markdown so the PDF contains the plan.
        _pdf_content = ""
        if _show_pdf:
            _pdf_content = _wf_for_pdf.get("plan_text") or ""
            if not _pdf_content and _domain_for_pdf and st.session_state.agent_state.get("user_email"):
                try:
                    _uid = st.session_state.agent_state.get("user_id") or ""
                    if _domain_for_pdf == "diet":
                        _latest = DietPlanRepository.find_latest_by_user(_uid, status="confirmed") if _uid else None
                    elif _domain_for_pdf == "workout":
                        _latest = WorkoutPlanRepository.find_latest_by_user(_uid, status="confirmed") if _uid else None
                    else:
                        _latest = None
                    if _latest:
                        _pdf_content = _latest.get("plan_markdown", "") or ""
                except Exception:  # noqa: BLE001
                    _pdf_content = ""
            if not _pdf_content:
                _pdf_content = response_content
        assistant_entry = {
            "role": "assistant",
            "content": response_content,
            "tool_used": tool_used or "",
            "technique_results": technique_results,
            "base_results": base_results,
            "show_pdf": _show_pdf,
            "pdf_content": _pdf_content,
        }
        if not (
            st.session_state.chat_history
            and st.session_state.chat_history[-1].get("role") == "assistant"
            and st.session_state.chat_history[-1].get("content") == response_content
        ):
            st.session_state.chat_history.append(assistant_entry)
    _ui_logger.info("[Turn %s] Assistant response persisted to history", turn_id)
