"""
agent/graph.py
──────────────
LangGraph ReAct agent graph for FITGEN.AI.

Architecture
────────────

         START
           │
           ▼
    ┌─────────────┐   has tool_calls?   ┌────────────┐
    │ base_agent  │ ──────────────────▶ │   tools    │
    │  (LLM +     │                     │ (ToolNode: │
    │   tools)    │ ◀────────────────── │  @tool fn) │
    └──────┬──────┘   tool result back  └────────────┘
           │ no tool_calls
           ▼
          END

How it works
────────────
1. base_agent  — a GPT-4o-mini LLM with workout_tool and diet_tool bound
                 to it.  Given the user message, it either responds
                 directly (general chat / intake) or emits a tool_call.
2. tool_node   — LangGraph's built-in ToolNode.  Executes whichever
                 @tool function the LLM requested and appends a
                 ToolMessage to state["messages"].
3. tools_condition — LangGraph's built-in conditional edge.  Checks
                 whether the last AIMessage contains tool_calls:
                   • yes  → "tool_node"
                   • no   → END

Extensibility
─────────────
To add a new specialist tool:
  1. Create agent/tools/my_tool.py with a @tool-decorated function.
  2. Add it to ALL_TOOLS in agent/tools/__init__.py.
  graph.py never needs to change.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.base_agent import base_agent  # default zero-shot node; swap via make_base_agent()
from agent.state import AgentState
from agent.tools import ALL_TOOLS


# ── Graph factory ─────────────────────────────────────────────────

def create_graph() -> StateGraph:
    """Build and compile the FITGEN.AI ReAct LangGraph.

    Topology
    --------
    START -> base_agent -> (tools_condition) -> tool_node -> base_agent
                                             -> END

    Returns
    -------
    Compiled LangGraph graph ready for ``.invoke()`` or ``.stream()``.
    """
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("base_agent", base_agent)
    graph.add_node("tools", ToolNode(ALL_TOOLS))  # must be "tools" for tools_condition

    # Edges
    graph.add_edge(START, "base_agent")

    # tools_condition: routes to "tools" if tool_calls present, else END
    graph.add_conditional_edges("base_agent", tools_condition)

    # After tool executes, always go back to base_agent to synthesise
    graph.add_edge("tools", "base_agent")

    return graph.compile()
