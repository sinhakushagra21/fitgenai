"""
agent/__init__.py
─────────────────
Public API for the FITGEN.AI agent package.
"""

from agent.base_agent import (
    base_agent,
    make_base_agent,
    PROMPT_ZERO_SHOT,
    PROMPT_FEW_SHOT,
    PROMPT_COT,
    PROMPT_ANALOGICAL,
    PROMPT_GENERATE_KNOWLEDGE,
    PROMPT_DECOMPOSITION,
)
from agent.graph import create_graph
from agent.router import router_node
from agent.state import AgentState
from agent.tools import ALL_TOOLS

__all__ = [
    "create_graph",
    "AgentState",
    "ALL_TOOLS",
    "router_node",
    "base_agent",
    "make_base_agent",
    "PROMPT_ZERO_SHOT",
    "PROMPT_FEW_SHOT",
    "PROMPT_COT",
    "PROMPT_ANALOGICAL",
    "PROMPT_GENERATE_KNOWLEDGE",
    "PROMPT_DECOMPOSITION",
]
