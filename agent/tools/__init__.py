"""
agent/tools/__init__.py
───────────────────────
Public registry of all FITGEN.AI tools.

To add a new tool:
  1. Create agent/tools/your_tool.py with a @tool-decorated function.
  2. Import it here and add it to ALL_TOOLS.
  The base agent in graph.py automatically binds everything in ALL_TOOLS.
"""

from agent.tools.diet_tool import diet_tool
from agent.tools.workout_tool import workout_tool

# ── Tool registry ─────────────────────────────────────────────────
# Add new tools to this list — no other file needs to change.
# NOTE: rag_query_tool (knowledge-base retrieval) is deactivated — RAG
# now lives exclusively inside diet_tool and workout_tool as Personal RAG
# over the user's own saved plan chunks + memory. The module stays on
# disk at agent/rag/rag_tool.py in case it needs to be revived later.
ALL_TOOLS = [
    workout_tool,
    diet_tool,
]

__all__ = ["ALL_TOOLS", "workout_tool", "diet_tool"]
