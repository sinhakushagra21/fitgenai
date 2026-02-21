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
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import streamlit.components.v1 as _components  # noqa: F401 — needed for st.components.v1.html
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent import AgentState, create_graph
from agent.prompts.base_prompts import BASE_PROMPTS
from agent.prompts.techniques import TECHNIQUE_KEYS, TECHNIQUE_META


# ── Log capture ───────────────────────────────────────────────────
class _ListHandler(logging.Handler):
    """Appends formatted log records to st.session_state.run_logs."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            st.session_state.setdefault("run_logs", []).append(self.format(record))
        except Exception:  # noqa: BLE001
            pass

_fitgen_logger = logging.getLogger("fitgen")
_fitgen_logger.setLevel(logging.DEBUG)
if not any(isinstance(h, _ListHandler) for h in _fitgen_logger.handlers):
    _handler = _ListHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s  %(name)s  %(message)s", "%H:%M:%S"))
    _fitgen_logger.addHandler(_handler)

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="FITGEN.AI — Prompt Engineering Demo",
    page_icon="💪",
    layout="wide",
)

# ── Load env ──────────────────────────────────────────────────────
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "**OPENAI_API_KEY not found.**  "
        "Copy `.env.example` → `.env` and add your key, then restart."
    )
    st.stop()

# ── Session state init ────────────────────────────────────────────

if "graph" not in st.session_state:
    st.session_state.graph = create_graph()

if "agent_state" not in st.session_state:
    st.session_state.agent_state: AgentState = {
        "messages": [],
        "user_profile": {},
    }

# chat_history entries: {role, content, tool_used, technique_results, base_results}
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []

if "run_logs" not in st.session_state:
    st.session_state.run_logs: list[str] = []

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

    def _call(tech: str) -> tuple[str, str]:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        resp = llm.invoke(
            [SystemMessage(content=BASE_PROMPTS[tech]), HumanMessage(content=query)]
        )
        return tech, resp.content

    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(_call, t): t for t in TECHNIQUE_KEYS}
        for f in futures:
            tech, text = f.result()
            results[tech] = text
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
        st.session_state.chat_history = []
        st.session_state.agent_state = {
            "messages": [],
            "user_profile": {},
        }
        st.rerun()

    st.markdown("---")

    # ── Live logs panel ───────────────────────────────────────────
    with st.expander("🪵 Backend Logs", expanded=False):
        if st.session_state.get("run_logs"):
            log_text = "\n".join(st.session_state.run_logs[-60:])  # last 60 lines
            st.code(log_text, language="text")
            if st.button("Clear logs", key="clear_logs"):
                st.session_state.run_logs = []
                st.rerun()
        else:
            st.caption("No logs yet — send a message to see backend activity.")

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

        with st.spinner("Thinking…"):
            for event in st.session_state.graph.stream(
                st.session_state.agent_state, stream_mode="values"
            ):
                final_event = event

                if event.get("messages"):
                    for msg in event["messages"]:
                        # Detect @tool call intent
                        if hasattr(msg, "tool_calls") and msg.tool_calls and not tool_used:
                            tool_used = msg.tool_calls[0]["name"]
                            badge_placeholder.markdown(_badge(tool_used), unsafe_allow_html=True)

                        # Parse ToolMessage JSON → technique_results
                        from langchain_core.messages import ToolMessage
                        if isinstance(msg, ToolMessage) and msg.content:
                            try:
                                parsed = json.loads(msg.content)
                                if isinstance(parsed, dict) and set(parsed.keys()) & set(TECHNIQUE_KEYS):
                                    technique_results = parsed
                                    with tabs_placeholder.container():
                                        with st.expander("📊 Specialist Prompt Comparison", expanded=True):
                                            _render_technique_tabs(technique_results)
                            except (json.JSONDecodeError, TypeError):
                                pass

                    # Final AIMessage text (not tool-call, not ToolMessage)
                    last_msg = event["messages"][-1]
                    if (
                        hasattr(last_msg, "content")
                        and last_msg.content
                        and last_msg.content != response_content
                        and not getattr(last_msg, "tool_calls", None)
                        and not isinstance(last_msg, ToolMessage)
                    ):
                        response_content = last_msg.content
                        response_placeholder.markdown(response_content)

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

    # ── Persist to display history ────────────────────────────────
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": response_content,
            "tool_used": tool_used or "",
            "technique_results": technique_results,
            "base_results": base_results,
        }
    )
