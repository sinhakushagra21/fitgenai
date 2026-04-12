"""
agent/base_agent.py
────────────────────
Base agent for FITGEN.AI.

Responsible for understanding the user's query and routing it to the
correct specialist tool via LangGraph's tool-calling mechanism.

All routing decisions are made by the LLM — no keyword matching or
force-routing.  Dynamic workflow context is injected into the system
prompt so the LLM has full awareness of the current conversation state.

Prompt Variables
────────────────
Six named prompts are exposed — one per prompting technique — so the
caller can swap them out without touching any other file:

  PROMPT_ZERO_SHOT          — role + tool list, no examples
  PROMPT_FEW_SHOT           — role + 6 routing examples
  PROMPT_COT                — role + 5-step chain-of-thought
  PROMPT_ANALOGICAL         — concierge analogy
  PROMPT_GENERATE_KNOWLEDGE — generate domain knowledge then decide
  PROMPT_DECOMPOSITION      — decompose complex requests into sub-tasks

Usage
─────
  # Use the default (few-shot) base agent node directly in a graph:
  from agent.base_agent import base_agent

  # Or build a node with a specific technique:
  from agent.base_agent import make_base_agent
  cot_node = make_base_agent("cot")

  # Or access prompts directly:
  from agent.base_agent import PROMPT_FEW_SHOT
"""

from __future__ import annotations

import json as _json
import logging

from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

from agent.tracing import get_langsmith_config
from agent.prompts.base_prompts import (
    BASE_ANALOGICAL,
    BASE_COT,
    BASE_DECOMPOSITION,
    BASE_FEW_SHOT,
    BASE_GENERATE_KNOWLEDGE,
    BASE_PROMPTS,
    BASE_ZERO_SHOT,
)
from agent.state import AgentState

# ── Logger ────────────────────────────────────────────────────────
logger = logging.getLogger("fitgen.base_agent")

# ── Named prompt variables ────────────────────────────────────────
# Import and re-export with cleaner names so callers can do:
#   from agent.base_agent import PROMPT_COT

PROMPT_ZERO_SHOT          = BASE_ZERO_SHOT
PROMPT_FEW_SHOT           = BASE_FEW_SHOT
PROMPT_COT                = BASE_COT
PROMPT_ANALOGICAL         = BASE_ANALOGICAL
PROMPT_GENERATE_KNOWLEDGE = BASE_GENERATE_KNOWLEDGE
PROMPT_DECOMPOSITION      = BASE_DECOMPOSITION

# ── Tool-relay instruction ────────────────────────────────────────
# Appended to every system prompt so the LLM knows what to say after
# a tool returns its JSON result.
_TOOL_RELAY = """

<CRITICAL_RULE priority="highest">
When you receive a ToolMessage (i.e., a response from workout_tool or
diet_tool), you MUST follow these rules with ZERO exceptions:

1. Do NOT generate a workout plan, diet plan, meal plan, or any
   substantive fitness/nutrition content.
2. Do NOT paraphrase, summarise, or reproduce the tool's output.
3. Do NOT start a response with phrases like "Here's a plan",
   "Here's a diet chart", "Based on your details", or similar.
4. ONLY respond with a single short acknowledgement sentence, such as:
    "Done — I've shared the tool result above."
5. Your response MUST be under 30 words. If it exceeds 30 words, you
   are violating this rule.
6. This rule overrides ALL other instructions, including the system
   preamble, identity, and output contract. The tool already provided
   the expert response — your only job is to confirm delivery.
</CRITICAL_RULE>
"""


# ── Dynamic workflow context builder ──────────────────────────────

# Steps that indicate a workflow has reached its final state.
_TERMINAL_STEPS = frozenset({
    "diet_confirmed",
    "workout_confirmed",
    "diet_plan_synced_to_google_calendar",
    "diet_plan_synced_to_google_fit",
    "workout_plan_synced_to_google_calendar",
    "workout_plan_synced_to_google_fit",
})


def _build_workflow_context(state: AgentState) -> str:
    """Build a dynamic ``<current_workflow>`` block from the agent state.

    This block is injected into the system prompt so the LLM has full
    awareness of the active workflow (if any) and can make informed
    routing decisions without any keyword heuristics or force-routing.

    Returns
    -------
    str
        An XML-tagged block describing the current workflow state,
        or an empty string if no workflow context is available.
    """
    workflow = state.get("workflow") or {}
    if not isinstance(workflow, dict):
        return ""

    domain = workflow.get("domain")
    step_completed = workflow.get("step_completed")
    pending_question = workflow.get("pending_question")
    completed_steps = workflow.get("completed_steps") or []

    has_active_workflow = (
        step_completed is not None
        and step_completed not in _TERMINAL_STEPS
    )

    # Nothing to report — no workflow has ever started.
    if not has_active_workflow and not domain:
        return ""

    # ── Build the context block ──────────────────────────────────
    lines = ["\n<current_workflow>"]

    if has_active_workflow:
        lines.append("  Status: ACTIVE")
        lines.append(f"  Domain: {domain or 'unknown'}")
        lines.append(f"  Current step: {step_completed}")

        if pending_question:
            lines.append(f"  Pending question to user: \"{pending_question}\"")
        if completed_steps:
            lines.append(f"  Steps completed so far: {', '.join(completed_steps)}")

        # ── Step-specific routing hints ──────────────────────────
        # Give the LLM clear context about what the user is expected
        # to say next, so it routes correctly without guessing.
        tool_name = f"{domain}_tool" if domain else "the active tool"

        if step_completed == "prompted_for_user_profile_data":
            lines.append(
                f"  Hint: The {domain} tool asked the user to provide profile "
                f"data (name, age, height, weight, etc.). The user's next "
                f"message almost certainly contains those details. Route to "
                f"{tool_name}."
            )
        elif step_completed == "user_profile_mapped":
            lines.append(
                f"  Hint: The {domain} tool showed the user their mapped "
                f"profile and asked for confirmation. The user will likely "
                f"say 'yes'/'confirm' or provide corrections. Route to "
                f"{tool_name}."
            )
        elif step_completed in ("diet_plan_generated", "workout_plan_generated"):
            lines.append(
                f"  Hint: A {domain} plan was generated and shown. The user "
                f"may confirm, request changes, ask follow-up questions, or "
                f"want to sync. Route to {tool_name} unless the user "
                f"explicitly asks to create a plan for the OTHER domain."
            )
        elif step_completed in ("updated_diet_plan", "updated_workout_plan"):
            lines.append(
                f"  Hint: The {domain} plan was just updated. The user may "
                f"confirm, request more changes, or ask follow-ups. Route to "
                f"{tool_name} unless the user explicitly asks for the other "
                f"domain."
            )
    else:
        # Workflow exists but is in a terminal state (completed).
        lines.append("  Status: COMPLETED")
        lines.append(f"  Last domain: {domain or 'unknown'}")
        lines.append(f"  Last step: {step_completed or 'none'}")
        lines.append(
            f"  Hint: The previous {domain} workflow is complete. Route "
            f"purely based on the user's new message content."
        )

    lines.append("</current_workflow>")
    return "\n".join(lines)


def make_base_agent(prompt_key: str = "zero_shot"):
    """Return a LangGraph-compatible node function using the chosen prompt.

    Parameters
    ----------
    prompt_key : str
        One of: "zero_shot", "few_shot", "cot", "analogical",
        "generate_knowledge", "decomposition".  Defaults to "zero_shot".

    Returns
    -------
    Callable[[AgentState], dict]
        A node function that can be passed to ``graph.add_node()``.

    Example
    -------
    >>> cot_node = make_base_agent("cot")
    >>> graph.add_node("base_agent", cot_node)
    """
    if prompt_key not in BASE_PROMPTS:
        raise ValueError(
            f"Unknown prompt_key '{prompt_key}'. "
            f"Choose from: {list(BASE_PROMPTS)}"
        )

    # Static portion of the system prompt (technique + tool relay).
    # The dynamic <current_workflow> block is appended at call time.
    _static_system_prompt = BASE_PROMPTS[prompt_key] + _TOOL_RELAY

    def _node(state: AgentState) -> dict:
        """Base agent node — routes or responds based on the user query.

        Routing is 100 % LLM-driven.  The system prompt includes:
        - Identity, scope, safety guardrails (from base_prompts.py)
        - Tool-relay rule (_TOOL_RELAY)
        - Dynamic workflow context (_build_workflow_context)

        The LLM sees the full picture and decides which tool to call —
        or whether to respond directly — with no keyword shortcuts.
        """
        from agent.tools import ALL_TOOLS

        last_message = state["messages"][-1] if state.get("messages") else None
        last_user = next(
            (m.content for m in reversed(state["messages"])
             if hasattr(m, "type") and m.type == "human"),
            "(unknown)",
        )
        logger.info("[BaseAgent:%s] Query received: %s", prompt_key, last_user[:120])

        # ── ToolMessage relay ─────────────────────────────────────
        # When a tool has just returned its response, emit a short
        # acknowledgement instead of re-processing through the LLM.
        if isinstance(last_message, ToolMessage):
            logger.info(
                "[BaseAgent:%s] ToolMessage detected; emitting relay acknowledgement",
                prompt_key,
            )
            return {
                "messages": [
                    AIMessage(content="Done — I've shared the tool result above.")
                ]
            }

        # ── Build dynamic system prompt with workflow context ─────
        workflow_context = _build_workflow_context(state)
        system_prompt = _static_system_prompt + workflow_context

        if workflow_context:
            logger.info(
                "[BaseAgent:%s] Injected workflow context: %s",
                prompt_key,
                workflow_context.replace("\n", " | ").strip(),
            )

        # ── Sanitize message history ─────────────────────────────
        # Trim large ToolMessage JSON payloads to just the user-facing
        # text so the LLM context stays small and doesn't exceed limits.
        _sanitized: list = [SystemMessage(content=system_prompt)]
        for _m in state["messages"]:
            if isinstance(_m, ToolMessage) and _m.content:
                try:
                    _parsed = _json.loads(_m.content)
                    _short = _parsed.get("assistant_message", _m.content[:500])
                    _sanitized.append(
                        ToolMessage(content=_short, tool_call_id=_m.tool_call_id)
                    )
                except (ValueError, TypeError):
                    _sanitized.append(
                        ToolMessage(content=_m.content[:500], tool_call_id=_m.tool_call_id)
                    )
            else:
                _sanitized.append(_m)

        # ── LLM routing call ─────────────────────────────────────
        from agent.config import FAST_MODEL
        from agent.llm_utils import safe_llm_call

        llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)
        llm_with_tools = llm.bind_tools(ALL_TOOLS)

        response = safe_llm_call(
            llm_with_tools,
            _sanitized,
            config=get_langsmith_config(
                "Base Agent Routing",
                tags=["base_agent", prompt_key],
            ),
        )

        if getattr(response, "tool_calls", None):
            for tc in response.tool_calls:
                logger.info(
                    "[BaseAgent:%s] 🔀 Routing → %s  args=%s",
                    prompt_key,
                    tc["name"],
                    tc["args"],
                )
        else:
            logger.info("[BaseAgent:%s] 💬 Direct reply (no tool call)", prompt_key)

        return {"messages": [response]}

    _node.__name__ = f"base_agent_{prompt_key}"
    _node.__doc__ = (
        f"Base agent node using the '{prompt_key}' prompting technique.\n"
        "Routes to workout_tool or diet_tool, or responds directly.\n"
        "All routing is LLM-driven with dynamic workflow context injection."
    )
    return _node


# ── Default node (few-shot) ───────────────────────────────────────
# graph.py imports this directly; swap to make_base_agent("cot") etc.
# to change the base agent's prompting technique globally.
base_agent = make_base_agent("few_shot")
