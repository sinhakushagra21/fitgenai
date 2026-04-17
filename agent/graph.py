"""
agent/graph.py
──────────────
LangGraph ReAct agent graph for FITGEN.AI.

Architecture (deterministic router)
────────────────────────────────────

         START
           |
           v
    +-------------+   tool_calls?   +------------+
    |   router    | --------------> |   tools    |
    | (determi-   |                 | (ToolNode: |
    |  nistic +   |                 |  @tool fn) |
    |  classifier)|                 +-----+------+
    +------+------+                       |
           | no tool_calls          +-----v------+
           v                        | state_sync |
          END                       +-----+------+
                                          |
                                          v
                                    +-----+------+
                                    |   router   | (ack)
                                    +------+-----+
                                           | no tool_calls
                                           v
                                          END

How it works
────────────
1. router      — Deterministic routing node (agent/router.py).
                 If there's an active workflow, routes to that domain's
                 tool without any LLM call.  If no workflow, uses a
                 focused fast-model classifier to decide the route.
                 For greetings/out-of-scope, responds directly.
                 Emits an AIMessage with tool_calls (for tool routes)
                 or content (for direct responses).

2. tool_node   — LangGraph's built-in ToolNode.  Executes whichever
                 @tool function was requested and appends a ToolMessage
                 to state["messages"].

3. state_sync  — Extracts state_updates from tool JSON responses and
                 persists them to MongoDB.

4. router(ack) — After state_sync, the router sees the ToolMessage
                 and emits a short acknowledgement ("Done — ...").
                 tools_condition sees no tool_calls → END.

Extensibility
─────────────
To add a new specialist tool:
  1. Create agent/tools/my_tool.py with a @tool-decorated function.
  2. Add it to ALL_TOOLS in agent/tools/__init__.py.
  3. Add the tool name to the router's classifier prompt in router.py.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.router import router_node
from agent.state_sync import apply_tool_state_updates
from agent.state import AgentState
from agent.tools import ALL_TOOLS


# ── Graph factory ─────────────────────────────────────────────────

def create_graph() -> StateGraph:
    """Build and compile the FITGEN.AI LangGraph.

    Topology
    --------
    START -> router -> (tools_condition) -> tool_node -> state_sync
                                         -> END           |
                                                          v
                                                        router -> END

    Returns
    -------
    Compiled LangGraph graph ready for ``.invoke()`` or ``.stream()``.
    """
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("router", router_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("state_sync", apply_tool_state_updates)

    # Edges
    graph.add_edge(START, "router")

    # tools_condition: routes to "tools" if tool_calls present, else END
    graph.add_conditional_edges("router", tools_condition)

    # After tool executes, sync state updates then loop back for ack
    graph.add_edge("tools", "state_sync")
    graph.add_edge("state_sync", "router")

    return graph.compile()
