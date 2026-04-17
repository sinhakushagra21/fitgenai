"""
agent/router.py
───────────────
Deterministic router for the FITGEN.AI LangGraph graph.

Replaces the LLM-driven routing in base_agent.py with a programmatic
approach that is faster, cheaper, and more reliable:

  1. Active workflow → route to active domain's tool  (deterministic)
  2. Domain switch detected → route to other tool     (keyword heuristic)
  3. No active workflow → lightweight LLM classifier  (focused call)
  4. "direct" intent → LLM generates a response       (greetings / OOS)

Why this is better than the old base_agent routing:
  - Active workflows are ALWAYS routed to the correct tool — no more
    "MANDATORY" prompt hints that the LLM sometimes ignores.
  - Domain switches are detected by a fast keyword heuristic, not by
    hoping the LLM notices an "EXCEPTION" clause buried in the prompt.
  - The LLM classifier is a FOCUSED classification call (one system
    message + one user message), not a response-generating call that
    also happens to pick a tool.  Much more reliable.
  - Direct responses use a minimal prompt (identity + scope), not the
    massive base_agent prompt with routing rules, workflow context, and
    safety sections that confused the LLM.

Graph topology
──────────────
  START → router → tools_condition:
    ├─ tool_calls → ToolNode → state_sync → router (ack) → END
    └─ no tool_calls → END
"""

from __future__ import annotations

import json as _json
import logging
import re
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agent.config import FAST_MODEL
from agent.error_utils import handle_exception
from agent.llm_utils import safe_llm_call
from agent.state import AgentState
from agent.tracing import get_langsmith_config, log_event

logger = logging.getLogger("fitgen.router")

# ── Terminal steps (workflow is "done") ──────────────────────────────
_TERMINAL_STEPS = frozenset({
    "diet_confirmed",
    "workout_confirmed",
    "diet_plan_synced_to_google_calendar",
    "diet_plan_synced_to_google_fit",
    "workout_plan_synced_to_google_calendar",
    "workout_plan_synced_to_google_fit",
})

# ── Valid route targets ──────────────────────────────────────────────
_VALID_ROUTES = frozenset({
    "diet_tool",
    "workout_tool",
    "rag_query_tool",
    "direct",
})

# ── Domain keyword sets for switch detection ─────────────────────────
_DIET_KEYWORDS = frozenset({
    "diet", "meal", "nutrition", "food", "eat", "eating",
    "calorie", "macro", "macros",
})
_WORKOUT_KEYWORDS = frozenset({
    "workout", "exercise", "training", "gym", "fitness",
    "lifting", "cardio", "strength",
})
_PLAN_KEYWORDS = frozenset({
    "plan", "program", "programme", "routine", "chart", "schedule",
})
_CREATE_KEYWORDS = frozenset({
    "create", "make", "build", "generate", "design", "start", "new",
    "give", "get", "show", "fetch",
})

# ── LLM classifier system prompt ────────────────────────────────────
_CLASSIFIER_SYSTEM = """\
You are an intent classifier for FITGEN.AI, a fitness coaching assistant.
Classify the user's message into exactly ONE category:

- "diet_tool" — User wants to create, modify, view, or delete a \
diet/meal/nutrition plan, is asking about their existing diet plan, \
or has a nutrition-specific question.
- "workout_tool" — User wants to create, modify, view, or delete a \
workout/exercise/training plan, is asking about their existing \
workout plan, or has an exercise-specific question.
- "rag_query_tool" — General fitness/nutrition KNOWLEDGE question \
that needs evidence-based, cited sources (e.g. "Is creatine safe?", \
"What are the benefits of HIIT?"). NOT a plan request.
- "direct" — Greetings ("hi", "hello"), out-of-scope questions \
(politics, coding, math), or meta questions about the assistant.

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


def _wants_domain_switch(query: str, current_domain: str) -> bool:
    """Detect if the user explicitly wants to switch to the other domain.

    Uses a fast keyword heuristic.  A switch is detected when the user's
    message mentions the *other* domain's keywords AND contains either:
      - a plan keyword + a creation keyword, OR
      - a switch phrase ("instead", "rather", "switch to").

    This correctly ignores profile field values like "exercise_frequency"
    or "diet_preference" that mention the other domain's words without
    implying a switch.
    """
    q = query.lower()
    words = set(re.findall(r"\w+", q))

    other = "workout" if current_domain == "diet" else "diet"
    other_kw = _WORKOUT_KEYWORDS if other == "workout" else _DIET_KEYWORDS

    # Must mention the other domain at all
    if not (words & other_kw):
        return False

    has_plan = bool(words & _PLAN_KEYWORDS)
    has_create = bool(words & _CREATE_KEYWORDS)
    has_switch_phrase = any(
        p in q
        for p in ("instead", "rather", "switch to", "switch from")
    )

    # Explicit "no, <action>" prefix — strong switch signal
    has_no_prefix = bool(re.match(r"^\s*no[\s,!.]+", q))

    return (
        (has_plan and has_create)
        or (has_plan and has_switch_phrase)
        or (has_switch_phrase)
        or (has_no_prefix and (has_plan or has_create))
    )


def _classify_intent(query: str) -> str:
    """Use a focused fast-LLM call to classify user intent.

    Returns one of: "diet_tool", "workout_tool", "rag_query_tool", "direct".
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


def _emit_tool_call(tool_name: str, query: str) -> dict:
    """Create an AIMessage with a programmatic tool_call.

    LangGraph's ``tools_condition`` will see the tool_calls and route
    to the ToolNode, which injects ``InjectedState`` automatically.
    """
    call_id = f"call_{tool_name}_{uuid4().hex[:8]}"
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": call_id,
                        "name": tool_name,
                        "args": {"query": query},
                    }
                ],
            )
        ]
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
       ``diet_tool`` | ``workout_tool`` | ``rag_query_tool`` | ``direct``.
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
        if _wants_domain_switch(user_query, current_domain=domain):
            other = "workout" if domain == "diet" else "diet"
            target = f"{other}_tool"
            logger.info(
                "Domain switch detected: %s -> %s", domain, other,
            )
            log_event(
                "router.domain_switch",
                module="router",
                from_domain=domain,
                to_domain=other,
                query_preview=user_query[:80],
            )
        else:
            target = f"{domain}_tool"
            logger.info(
                "Active workflow -> %s (step: %s)", target, step,
            )
            log_event(
                "router.active_workflow",
                module="router",
                target=target,
                step=step,
            )
        return _emit_tool_call(target, user_query)

    # ── 3. No active workflow → LLM classifier ─────────────────────
    route = _classify_intent(user_query)
    logger.info("Classified intent: %s", route)
    log_event(
        "router.classified",
        module="router",
        route=route,
        query_preview=user_query[:80],
    )

    if route in ("diet_tool", "workout_tool", "rag_query_tool"):
        return _emit_tool_call(route, user_query)

    # ── 4. Direct response ──────────────────────────────────────────
    logger.info("Direct response (greeting / out-of-scope)")
    return _generate_direct_response(state)
