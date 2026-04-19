"""
agent/router.py
───────────────
Deterministic router for the FITGEN.AI LangGraph graph.

Replaces the LLM-driven routing in base_agent.py with a programmatic
approach that is faster, cheaper, and more reliable:

  1. Active workflow → LLM "active-turn gate" picks one of:
        stay | side_diet | side_workout | switch | direct
     — using the @tool descriptions (loaded at import) as context.
  2. "stay"              → dispatch to active domain's tool.
  3. "side_diet/workout" → answer out-of-band in the OTHER domain
                           WITHOUT mutating the active workflow.
  4. "switch"            → fall through to the fresh-conversation classifier.
  5. "direct"            → LLM generates a direct response (greetings / OOS).
  6. No active workflow  → lightweight LLM classifier picks a tool.

Why a gate instead of a keyword heuristic:
  - Users ask about "traps", "biceps", "creatine", "RDL" etc. mid-flow.
    No hand-maintained keyword list can keep up; the LLM gate reads the
    tool descriptions and the active-workflow context and decides
    naturally.
  - Side queries are answered out-of-band so an unconfirmed plan draft
    is not destroyed by a stray exercise question.

Graph topology
──────────────
  START → router → tools_condition:
    ├─ tool_calls → ToolNode → state_sync → router (ack) → END
    └─ no tool_calls → END
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agent.config import FAST_MODEL
from agent.error_utils import handle_exception
from agent.llm_utils import safe_llm_call
from agent.state import AgentState
from agent.tracing import get_langsmith_config, log_event

logger = logging.getLogger("fitgen.router")

# ── Tool description cache (loaded once at import) ───────────────────
# We read `.description` off each @tool-decorated BaseTool so the LLM
# gate can reason over live tool docs instead of hand-maintained blurbs.
# This cache is built on app import and never changes at runtime.
try:
    from agent.tools import diet_tool as _diet_tool_obj
    from agent.tools import workout_tool as _workout_tool_obj

    _TOOL_DESCRIPTIONS: dict[str, str] = {
        "diet_tool": (getattr(_diet_tool_obj, "description", "") or "").strip(),
        "workout_tool": (getattr(_workout_tool_obj, "description", "") or "").strip(),
    }
    logger.info(
        "[Router] Loaded tool descriptions: %s",
        {k: len(v) for k, v in _TOOL_DESCRIPTIONS.items()},
    )
except Exception as _exc:  # noqa: BLE001
    logger.warning("[Router] Could not load tool descriptions: %s", _exc)
    _TOOL_DESCRIPTIONS = {
        "diet_tool": "Diet, meal plans, nutrition, foods, macros, calories.",
        "workout_tool": "Workout and exercise plans, training schedules, reps, sets, body parts, lifts.",
    }

# ── Terminal steps (workflow is "done") ──────────────────────────────
# NOTE: `diet_confirmed` / `workout_confirmed` are NOT terminal — the user
# is still being prompted for a sync decision (calendar / skip). Keeping
# the workflow active while `step_completed` is `*_confirmed` lets the
# router deterministically dispatch the follow-up reply ("yes", "skip",
# etc.) to the correct domain tool. Once the user actually completes
# (or declines) sync, the tool transitions to one of the terminal steps
# below — OR wipes the workflow for a graceful skip — so the next message
# goes through the fresh-conversation LLM classifier path.
_TERMINAL_STEPS = frozenset({
    "diet_plan_synced_to_google_calendar",
    "workout_plan_synced_to_google_calendar",
})

# ── Valid route targets ──────────────────────────────────────────────
_VALID_ROUTES = frozenset({
    "diet_tool",
    "workout_tool",
    "both",
    "direct",
})

# ── Active-turn gate (LLM) ───────────────────────────────────────────
# When a workflow is in progress, this gate decides per turn whether to:
#   - stay   : continue the active workflow (profile answers, confirms, syncs)
#   - side_diet / side_workout : answer a one-shot question in the OTHER
#     domain WITHOUT touching the active workflow
#   - switch : user wants to abandon the active flow and start fresh
#   - direct : greeting / chitchat / out-of-scope
_ACTIVE_TURN_VALID = frozenset(
    {"stay", "side_diet", "side_workout", "switch", "direct"}
)

_ACTIVE_TURN_GATE_SYSTEM = """\
You are a routing gate for FITGEN.AI, a multi-turn fitness coaching assistant.

A {domain} workflow is already in progress. The user's last completed step is \
"{step}". Your job is to decide how to handle the NEXT user turn.

Here are the specialist tools available, with their own descriptions:

[diet_tool]
{diet_desc}

[workout_tool]
{workout_desc}

Choose EXACTLY ONE label for the user turn:

- "stay"         — The turn belongs to the active {domain} workflow. Use for \
profile answers (name, age, weight, goals…), confirmations ("yes", "confirm", \
"looks good"), sync replies ("calendar", "skip", "done"), \
edits/updates to the {domain} plan, or on-topic {domain} follow-up questions.
- "side_diet"    — One-off nutrition / food / diet question that is NOT about \
creating or modifying a plan. Answer out-of-band without disturbing the active \
workflow.
- "side_workout" — One-off exercise / body-part / training question that is NOT \
about creating or modifying a plan. Answer out-of-band without disturbing the \
active workflow.
- "switch"       — The user wants to ABANDON the active {domain} workflow and \
start a new plan in the other domain, delete the current plan, or explicitly \
asks to switch ("forget that", "instead make me a workout plan", "create a \
diet plan now").
- "direct"       — Greeting, chitchat, meta-question about the assistant, or \
fully out-of-scope (politics, coding, etc.).

Important rules:
- Short utterances like "yes", "no", "ok", "skip", numbers, names, \
single food words (e.g. "chicken") during profile intake → ALWAYS "stay".
- Questions about body parts, lifts, reps, sets, form, cardio → "side_workout" \
(unless the active domain is already workout, in which case "stay").
- Questions about nutrients, meals, recipes, calories, allergies → "side_diet" \
(unless the active domain is already diet, in which case "stay").
- Only pick "switch" if the user clearly wants to START A NEW PLAN in the \
other domain. A side question is NOT a switch.

Respond with ONE lowercase label and nothing else."""


def _classify_active_turn(query: str, domain: str, step: str | None) -> str:
    """Ask the fast LLM what to do with this turn during an active workflow.

    Returns one of: "stay", "side_diet", "side_workout", "switch", "direct".
    Falls back to "stay" on error (safest: don't disrupt the workflow).
    """
    try:
        llm = ChatOpenAI(model=FAST_MODEL, temperature=0, max_tokens=8)
        system = _ACTIVE_TURN_GATE_SYSTEM.format(
            domain=domain,
            step=step or "(none)",
            diet_desc=_TOOL_DESCRIPTIONS.get("diet_tool", ""),
            workout_desc=_TOOL_DESCRIPTIONS.get("workout_tool", ""),
        )
        resp = safe_llm_call(
            llm,
            [SystemMessage(content=system), HumanMessage(content=query)],
            config=get_langsmith_config(
                "Router Active-Turn Gate",
                tags=["router", "gate", domain],
            ),
        )
        label = resp.content.strip().lower().strip("\"'`.")
        # Tolerate minor formatting noise
        label = label.split()[0] if label else ""
        if label not in _ACTIVE_TURN_VALID:
            logger.warning(
                "Active-turn gate returned '%s' (invalid) — defaulting to 'stay'",
                label,
            )
            return "stay"
        return label
    except Exception as exc:  # noqa: BLE001
        handle_exception(
            exc,
            module="router",
            context="active-turn gate",
            level="WARNING",
            extra={"query_preview": query[:120], "domain": domain, "step": step},
        )
        return "stay"

# ── LLM classifier system prompt ────────────────────────────────────
_CLASSIFIER_SYSTEM = """\
You are an intent classifier for FITGEN.AI, a fitness coaching assistant.
Classify the user's message into exactly ONE category:

- "diet_tool" — User wants to create, modify, view, or delete a \
diet/meal/nutrition plan, is asking about their existing diet plan, \
or has a nutrition-specific question.
- "workout_tool" — User wants to create, modify, view, or delete a \
workout/exercise/training plan, is asking about their existing \
workout plan, or has an exercise-specific question (including general \
knowledge questions about exercise, training, or body parts).
- "both" — User's message asks about BOTH domains in one turn \
(e.g. "show me my diet and workout plan for Tuesday", "create a bulk \
plan with meals and workouts", "what's my full plan today?"). Both \
specialist tools should answer and their results will be combined.
- "direct" — Greetings ("hi", "hello"), out-of-scope questions \
(politics, coding, math), or meta questions about the assistant.

Note: General nutrition knowledge questions (e.g. "is creatine safe?") \
should map to "diet_tool"; general training knowledge questions \
(e.g. "benefits of HIIT?") should map to "workout_tool". If the query \
mixes nutrition AND training in a single request, pick "both".

Respond with ONLY the category name. No explanation, no quotes, \
no punctuation."""

# ── Direct response system prompt ────────────────────────────────────
_DIRECT_RESPONSE_SYSTEM = """\
<identity>
You are **FITGEN.AI** — an AI-powered personal fitness coaching assistant \
built to help users with exercise programming and evidence-based nutrition \
guidance. You are friendly, motivating, and concise.
</identity>

<scope>
You can help with:
- Personalised workout plans and training programmes
- Diet plans, meal planning, and nutrition guidance
- General fitness and nutrition knowledge

You CANNOT help with:
- Non-fitness topics (politics, coding, math, entertainment, etc.)
- Medical diagnoses or prescriptions
- Anabolic steroids or controlled substances
</scope>

<rules>
- If the user greets you, respond warmly and explain your capabilities.
- If the user asks something out-of-scope, politely decline and redirect \
  to fitness topics.
- Keep responses concise and friendly.
- ALWAYS respond in English only, regardless of the user's language.
- NEVER generate workout plans, diet plans, or detailed fitness advice \
  directly — those are handled by specialist tools. Just greet or redirect.
</rules>"""


# =====================================================================
# Helper functions
# =====================================================================

def _last_human_text(messages: list) -> str:
    """Extract the content of the most recent HumanMessage."""
    for m in reversed(messages):
        if hasattr(m, "type") and m.type == "human":
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _classify_intent(query: str) -> str:
    """Use a focused fast-LLM call to classify user intent.

    Returns one of: "diet_tool", "workout_tool", "both", "direct".
    Falls back to "direct" on any error.
    """
    try:
        llm = ChatOpenAI(model=FAST_MODEL, temperature=0, max_tokens=20)
        resp = safe_llm_call(
            llm,
            [
                SystemMessage(content=_CLASSIFIER_SYSTEM),
                HumanMessage(content=query),
            ],
            config=get_langsmith_config(
                "Router Intent Classifier",
                tags=["router", "classifier"],
            ),
        )
        result = resp.content.strip().lower().strip("\"'")
        if result not in _VALID_ROUTES:
            logger.warning(
                "Classifier returned '%s' (not in valid routes) — "
                "defaulting to 'direct'",
                result,
            )
            log_event(
                "router.classifier.invalid_route",
                level="WARNING",
                module="router",
                raw_response=result,
                query_preview=query[:80],
            )
            return "direct"
        return result
    except Exception as exc:  # noqa: BLE001
        handle_exception(
            exc,
            module="router",
            context="intent classification",
            extra={"query_preview": query[:120]},
        )
        return "direct"


def _emit_tool_call(
    tool_name: str,
    query: str,
    *,
    side_query: bool = False,
) -> dict:
    """Create an AIMessage with a programmatic tool_call.

    LangGraph's ``tools_condition`` will see the tool_calls and route
    to the ToolNode, which injects ``InjectedState`` automatically.

    When ``side_query=True``, the tool is instructed to answer the turn
    as ``general_{domain}_query`` WITHOUT mutating the active workflow
    or user_profile — used for out-of-domain probes during an active
    workflow in the OTHER domain.
    """
    call_id = f"call_{tool_name}_{uuid4().hex[:8]}"
    args: dict[str, Any] = {"query": query}
    if side_query:
        args["side_query"] = True
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": call_id,
                        "name": tool_name,
                        "args": args,
                    }
                ],
            )
        ]
    }


def _emit_multi_tool_calls(tool_names: list[str], query: str) -> dict:
    """Emit an AIMessage with multiple tool_calls in a single turn.

    Used when the user's query spans multiple domains (e.g. asks for
    both diet and workout in one message). LangGraph's ToolNode will
    dispatch all tool_calls and produce one ToolMessage per call; the
    downstream state_sync node merges state updates from both before
    the router emits the final acknowledgement.
    """
    calls = [
        {
            "id": f"call_{name}_{uuid4().hex[:8]}",
            "name": name,
            "args": {"query": query},
        }
        for name in tool_names
    ]
    return {
        "messages": [AIMessage(content="", tool_calls=calls)]
    }


def _generate_direct_response(state: AgentState) -> dict:
    """Generate a direct LLM response for greetings / out-of-scope."""
    # Build a sanitised message list with a minimal system prompt.
    sanitized: list = [SystemMessage(content=_DIRECT_RESPONSE_SYSTEM)]
    for m in state.get("messages") or []:
        if isinstance(m, ToolMessage) and m.content:
            try:
                parsed = _json.loads(m.content)
                short = parsed.get("assistant_message", m.content[:300])
            except (ValueError, TypeError):
                short = m.content[:300]
            sanitized.append(
                ToolMessage(content=short, tool_call_id=m.tool_call_id)
            )
        else:
            sanitized.append(m)

    try:
        llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)
        response = safe_llm_call(
            llm,
            sanitized,
            config=get_langsmith_config(
                "Router Direct Response",
                tags=["router", "direct"],
            ),
        )
        return {"messages": [response]}
    except Exception as exc:  # noqa: BLE001
        handle_exception(
            exc,
            module="router",
            context="direct response generation",
        )
        return {
            "messages": [
                AIMessage(
                    content="Hi! I'm FITGEN.AI — I can help you with "
                    "workout plans and nutrition guidance. "
                    "What would you like to work on today?"
                )
            ]
        }


# =====================================================================
# Main router node
# =====================================================================

def router_node(state: AgentState) -> dict:
    """Deterministic router node for the FITGEN.AI LangGraph graph.

    Routing logic (evaluated in order):

    1. **ToolMessage** → emit a short acknowledgement (no LLM call).
    2. **Active workflow** → route to the active domain's tool
       (deterministic).  If a domain switch is detected via keyword
       heuristic, route to the other domain's tool instead.
    3. **No active workflow** → use a focused LLM classifier to decide:
       ``diet_tool`` | ``workout_tool`` | ``both`` | ``direct``.
       When the classifier returns ``both`` (cross-domain request),
       dispatch a single AIMessage containing tool_calls for BOTH
       diet_tool and workout_tool; ToolNode will run them in parallel.
    4. **"direct" route** → generate a direct LLM response for greetings
       and out-of-scope queries.
    """
    messages = state.get("messages") or []
    last_msg = messages[-1] if messages else None

    # ── 1. ToolMessage → short acknowledgement ──────────────────────
    if isinstance(last_msg, ToolMessage):
        logger.info("Tool result received -> emitting acknowledgement")
        log_event("router.ack", module="router")
        return {
            "messages": [
                AIMessage(content="Done — I've shared the tool result above.")
            ]
        }

    # ── Extract user query ──────────────────────────────────────────
    user_query = _last_human_text(messages)
    if not user_query:
        logger.warning("No human message found -> direct response")
        return _generate_direct_response(state)

    logger.info("Routing query: %s", user_query[:120])

    # ── 2. Active workflow → deterministic routing ──────────────────
    workflow = state.get("workflow") or {}
    domain = workflow.get("domain")
    step = workflow.get("step_completed")

    has_active_workflow = (
        domain is not None
        and step is not None
        and step not in _TERMINAL_STEPS
    )

    if has_active_workflow:
        gate = _classify_active_turn(user_query, domain=domain, step=step)
        logger.info(
            "[Router] Active-turn gate: %s (domain=%s, step=%s)",
            gate, domain, step,
        )
        log_event(
            "router.active_turn_gate",
            module="router",
            gate=gate,
            domain=domain,
            step=step,
            query_preview=user_query[:80],
        )

        if gate == "stay":
            return _emit_tool_call(f"{domain}_tool", user_query)

        if gate == "side_diet":
            # If active domain IS diet, "side_diet" is just "stay".
            if domain == "diet":
                return _emit_tool_call("diet_tool", user_query)
            # Otherwise dispatch to diet_tool with side_query=True so it
            # forces general_diet_query intent AND preserves the active
            # workout workflow/profile (no clobber).
            return _emit_tool_call(
                "diet_tool", user_query, side_query=True,
            )

        if gate == "side_workout":
            if domain == "workout":
                return _emit_tool_call("workout_tool", user_query)
            return _emit_tool_call(
                "workout_tool", user_query, side_query=True,
            )

        if gate == "switch":
            # Fall through to the fresh-conversation classifier below so
            # the user can start a new plan / knowledge query / etc.
            # The target tool's own from_state() will reset the stale
            # cross-domain workflow.
            logger.info(
                "[Router] Switch requested — falling through to fresh classifier"
            )

        elif gate == "direct":
            return _generate_direct_response(state)

        # gate == "switch" falls through to the fresh-classifier path

    # ── 3. No active workflow → LLM classifier ─────────────────────
    route = _classify_intent(user_query)
    logger.info("Classified intent: %s", route)
    log_event(
        "router.classified",
        module="router",
        route=route,
        query_preview=user_query[:80],
    )

    if route == "both":
        logger.info("[Router] Cross-domain query — dispatching to both tools")
        return _emit_multi_tool_calls(["diet_tool", "workout_tool"], user_query)

    if route in ("diet_tool", "workout_tool"):
        return _emit_tool_call(route, user_query)

    # ── 4. Direct response ──────────────────────────────────────────
    logger.info("Direct response (greeting / out-of-scope)")
    return _generate_direct_response(state)
