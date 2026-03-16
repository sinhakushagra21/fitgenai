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

import json
import logging
import os
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import streamlit as st
import streamlit.components.v1 as _components  # noqa: F401 — needed for st.components.v1.html
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent import AgentState, create_graph
from agent.persistence import get_context_state, get_latest_context_state_by_email, init_db
from agent.prompts.base_prompts import BASE_PROMPTS
from agent.prompts.techniques import TECHNIQUE_KEYS, TECHNIQUE_META


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
    page_title="FITGEN.AI — Prompt Engineering Demo",
    page_icon="💪",
    layout="wide",
)

# ── Load env ──────────────────────────────────────────────────────
load_dotenv()
init_db()

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "**OPENAI_API_KEY not found.**  "
        "Copy `.env.example` → `.env` and add your key, then restart."
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
            exchange_code_for_tokens,
            extract_calendar_events,
            push_events_to_calendar,
        )

        tokens = exchange_code_for_tokens(_gcal_code)
        _gcal_logger.info("[Calendar] Token exchange successful")

        # Pull plan_text and domain from workflow state.
        _wf = st.session_state.get("agent_state", {}).get("workflow", {})
        _plan_text = _wf.get("plan_text", "")
        _domain = _wf.get("domain", "diet")
        _profile = st.session_state.get("agent_state", {}).get("user_profile", {})

        if _plan_text:
            events = extract_calendar_events(_plan_text, _domain, _profile)
            if events:
                created_count = push_events_to_calendar(events, tokens)
                st.success(
                    f"✅ **{created_count} events** synced to your Google Calendar! "
                    f"Check your calendar for recurring {_domain} reminders starting tomorrow."
                )
                _gcal_logger.info("[Calendar] Pushed %d events successfully", created_count)

                # Store tokens in session for potential future use.
                st.session_state["google_calendar_tokens"] = tokens
                st.session_state["calendar_events_pushed"] = True
            else:
                st.warning("⚠️ Connected to Google, but couldn't extract events from the plan.")
        else:
            st.warning("⚠️ Connected to Google, but no plan text found. Create a plan first, then sync.")

    except Exception as e:
        st.error(f"❌ Calendar sync failed: {e}")
        _gcal_logger.error("[Calendar] OAuth or push failed: %s", e)

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

# ── Helpers ───────────────────────────────────────────────────────────

TOOL_LABELS: dict[str, tuple[str, str]] = {
    "workout_tool": ("💪 Workout Coach", "#2e7d32"),
    "diet_tool":    ("🥗 Diet Coach",    "#1565c0"),
    "general":      ("🤖 FITGEN.AI",     "#6a1b9a"),
}


def _badge(tool: str) -> str:
    """Return an HTML pill badge for the given tool name."""
    label, color = TOOL_LABELS.get(tool, TOOL_LABELS["general"])
    return (
        f'<span style="background:{color};color:white;padding:3px 12px;'
        f'border-radius:12px;font-size:0.75rem;font-weight:600;'
        f'margin-bottom:8px;display:inline-block">{label}</span>'
    )


def _copy_button(text: str, key: str) -> None:
    """Render a small JS-powered copy-to-clipboard button."""
    # Escape backticks and backslashes so the text is safe inside a JS template literal
    safe = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    st.components.v1.html(
        f"""
        <button onclick="navigator.clipboard.writeText(`{safe}`).then(()=>{{
            this.textContent='✅ Copied!';
            setTimeout(()=>this.textContent='📋 Copy', 1500);
        }})"
        style="background:#374151;color:#e5e7eb;border:none;border-radius:6px;
               padding:4px 12px;font-size:0.75rem;cursor:pointer;margin-top:4px;">
            📋 Copy
        </button>
        """,
        height=38,
    )


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
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
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


# ── Header ────────────────────────────────────────────────────────

col_title, col_sub = st.columns([2, 3])
with col_title:
    st.markdown("# 💪 FITGEN.AI")
with col_sub:
    st.markdown(
        "<p style='color:gray;margin-top:1.1rem;'>"
        "Prompt Engineering Demo · 5 Techniques · Live Comparison"
        "</p>",
        unsafe_allow_html=True,
    )
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Prompting Techniques")
    for key in TECHNIQUE_KEYS:
        m = TECHNIQUE_META[key]
        st.markdown(
            f"**{m['icon']} {m['label']}** — <span style='color:{m['color']};font-size:0.85rem;'>{m['description']}</span>",
            unsafe_allow_html=True,
        )
    st.divider()

    st.markdown("## 🔀 Routing")
    st.markdown(
        """
        - 💪 **workout_tool** — training plans & exercise science
        - 🥗 **diet_tool** — nutrition, meals & macros
        - 🤖 **Direct reply** — greetings & mixed topics
        """
    )
    st.divider()

    show_base = st.checkbox(
        "Show base-agent technique comparison",
        value=False,
        help="Run all 5 prompting techniques on the BASE agent too and show side-by-side.",
    )

    st.divider()
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
        st.rerun()

    st.markdown("---")

    # ── Live logs panel ───────────────────────────────────────────
    with st.expander("🪵 Backend Logs", expanded=False):
        if _LOG_BUFFER:
            log_text = "\n".join(list(_LOG_BUFFER)[-120:])  # last 120 lines
            st.code(log_text, language="text")
            if st.button("Clear logs", key="clear_logs"):
                _LOG_BUFFER.clear()
                st.rerun()
        else:
            st.caption("No logs yet — send a message to see backend activity.")

    # ── Google Calendar sidebar ───────────────────────────────────
    st.divider()
    st.markdown("## 📅 Google Calendar")

    _has_google_creds = bool(os.getenv("GOOGLE_CLIENT_ID")) and bool(os.getenv("GOOGLE_CLIENT_SECRET"))
    _wf_state = st.session_state.get("agent_state", {}).get("workflow", {})
    _calendar_stage = _wf_state.get("stage")
    _already_pushed = st.session_state.get("calendar_events_pushed", False)

    if _already_pushed:
        st.success("✅ Calendar synced!")
    elif _has_google_creds and _calendar_stage == "calendar_oauth_pending":
        try:
            from agent.tools.calendar_integration import get_authorization_url
            _auth_url, _ = get_authorization_url()
            st.link_button("📅 Connect Google Calendar", _auth_url, use_container_width=True)
            st.caption("Sign in with Google to push your plan as calendar events.")
        except Exception as _e:
            st.caption(f"⚠️ Calendar setup error: {_e}")
    elif not _has_google_creds:
        st.caption("Add `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` to .env to enable.")
    else:
        st.caption("Complete a plan and say 'yes' to calendar sync to enable.")

    st.caption("Built with LangGraph · LangChain · OpenAI · Streamlit")

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

# ── Chat input ────────────────────────────────────────────────────

prompt = st.chat_input("Ask me anything about fitness, workouts, or nutrition…")

if prompt:
    turn_id = uuid.uuid4().hex[:8]
    turn_start = perf_counter()
    _ui_logger.info("[Turn %s] Received user prompt: %s", turn_id, prompt)

    # ── Display user message ──────────────────────────────────────
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

        with st.spinner("Thinking…"):
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
                        badge_placeholder.markdown(_badge(tool_used), unsafe_allow_html=True)
                        _ui_logger.info("[Turn %s] Routed to tool=%s", turn_id, tool_used)

                    # Parse ToolMessage JSON → user-facing assistant message
                    from langchain_core.messages import ToolMessage
                    if isinstance(last_msg, ToolMessage) and last_msg.content:
                        try:
                            parsed = json.loads(last_msg.content)
                            assistant_message = parsed.get("assistant_message")
                            if assistant_message and assistant_message != response_content:
                                response_content = assistant_message
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
                    ):
                        response_content = last_msg.content
                        response_placeholder.markdown(response_content)
                        _ui_logger.debug(
                            "[Turn %s] Final AI direct reply received (chars=%d)",
                            turn_id,
                            len(response_content),
                        )

            _ui_logger.info(
                "[Turn %s] Stream completed in %.2fs with %d events",
                turn_id,
                perf_counter() - turn_start,
                event_count,
            )

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

    # ── Persist to display history ────────────────────────────────
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
