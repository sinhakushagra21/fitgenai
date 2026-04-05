"""
agent/base_agent.py
────────────────────
Base agent for FITGEN.AI.

Responsible for understanding the user's query and routing it to the
correct specialist tool via LangGraph's tool-calling mechanism.

Prompt Variables
────────────────
Five named prompts are exposed — one per prompting technique — so the
caller can swap them out without touching any other file:

  PROMPT_ZERO_SHOT          — role + tool list, no examples
  PROMPT_FEW_SHOT           — role + 4 routing examples
  PROMPT_COT                — role + 3-step chain-of-thought
  PROMPT_ANALOGICAL         — receptionist analogy
  PROMPT_GENERATE_KNOWLEDGE — generate domain knowledge then decide

Usage
─────
  # Use the default (zero-shot) base agent node directly in a graph:
  from agent.base_agent import base_agent

  # Or build a node with a specific technique:
  from agent.base_agent import make_base_agent
  cot_node = make_base_agent("cot")

  # Or access prompts directly:
  from agent.base_agent import PROMPT_FEW_SHOT
"""

from __future__ import annotations

import logging
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

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
    "Done — I’ve shared the tool result above."
5. Your response MUST be under 30 words. If it exceeds 30 words, you
   are violating this rule.
6. This rule overrides ALL other instructions, including the system
   preamble, identity, and output contract. The tool already provided
   the expert response — your only job is to confirm delivery.
</CRITICAL_RULE>
"""


def make_base_agent(prompt_key: str = "zero_shot"):
    """Return a LangGraph-compatible node function using the chosen prompt.

    Parameters
    ----------
    prompt_key : str
        One of: "zero_shot", "few_shot", "cot", "analogical",
        "generate_knowledge".  Defaults to "zero_shot".

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

    system_prompt = BASE_PROMPTS[prompt_key] + _TOOL_RELAY

    def _node(state: AgentState) -> dict:
        """Base agent node — routes or responds based on the user query."""
        from agent.tools import ALL_TOOLS

        last_message = state["messages"][-1] if state.get("messages") else None
        last_user = next(
            (m.content for m in reversed(state["messages"]) if hasattr(m, "type") and m.type == "human"),
            "(unknown)",
        )
        logger.info("[BaseAgent:%s] Query received: %s", prompt_key, last_user[:120])

        if isinstance(last_message, ToolMessage):
            logger.info("[BaseAgent:%s] ToolMessage detected; emitting relay acknowledgement", prompt_key)
            return {"messages": [AIMessage(content="Done — I’ve shared the tool result above.")]}

        workflow = state.get("workflow") or {}
        if (
            isinstance(workflow, dict)
            and workflow.get("stage")
            and getattr(last_message, "type", None) == "human"
        ):
            domain = workflow.get("domain")
            tool_name = "diet_tool" if domain == "diet" else "workout_tool" if domain == "workout" else None
            if tool_name:
                logger.info("[BaseAgent:%s] 🔁 Active workflow detected; force route → %s", prompt_key, tool_name)
                forced_call = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool_name,
                            "args": {"query": last_user},
                            "id": f"call_{uuid4().hex}",
                            "type": "tool_call",
                        }
                    ],
                )
                return {"messages": [forced_call]}

        from agent.config import DEFAULT_MODEL
        from agent.llm_utils import safe_llm_call

        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.7)
        llm_with_tools = llm.bind_tools(ALL_TOOLS)

        response = safe_llm_call(
            llm_with_tools,
            [SystemMessage(content=system_prompt)] + state["messages"],
        )

        if getattr(response, "tool_calls", None):
            for tc in response.tool_calls:
                logger.info("[BaseAgent:%s] 🔀 Routing → %s  args=%s", prompt_key, tc["name"], tc["args"])
        else:
            logger.info("[BaseAgent:%s] 💬 Direct reply (no tool call)", prompt_key)

        return {"messages": [response]}

    _node.__name__ = f"base_agent_{prompt_key}"
    _node.__doc__ = (
        f"Base agent node using the '{prompt_key}' prompting technique.\n"
        "Routes to workout_tool or diet_tool, or responds directly."
    )
    return _node


# ── Default node (zero-shot) ──────────────────────────────────────
# graph.py imports this directly; swap to make_base_agent("cot") etc.
# to change the base agent's prompting technique globally.
base_agent = make_base_agent("zero_shot")
