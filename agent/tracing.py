"""
agent/tracing.py
────────────────
LangSmith tracing configuration for FITGEN.AI.

Provides structured, human-readable traces of the entire pipeline:
  • Base Agent (routing)
  • Intent Classification
  • Profile Extraction
  • Plan Generation (Step 1: JSON, Step 2: Markdown)
  • Plan Validation
  • YouTube Enrichment
  • RAG Queries

Usage:
  from agent.tracing import trace, get_tracer_callbacks

  @trace(name="my_function", run_type="chain")
  def my_function(...):
      ...

  # Or pass callbacks to LLM calls:
  llm.invoke(messages, config={"callbacks": get_tracer_callbacks()})
"""

from __future__ import annotations

import logging
import os
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger("fitgen.tracing")

# ── LangSmith availability check ────────────────────────────────
_LANGSMITH_AVAILABLE = False
_traceable = None

try:
    from langsmith import traceable as _ls_traceable
    from langsmith.run_helpers import get_current_run_tree

    _api_key = os.getenv("LANGCHAIN_API_KEY", "")
    _tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"

    if _api_key and _api_key != "your-langchain-api-key-here" and _tracing_enabled:
        _LANGSMITH_AVAILABLE = True
        _traceable = _ls_traceable
        logger.info(
            "✅ LangSmith tracing ENABLED (project: %s)",
            os.getenv("LANGCHAIN_PROJECT", "default"),
        )
    else:
        logger.info(
            "⚠️ LangSmith tracing DISABLED (API key: %s, tracing_v2: %s)",
            "set" if _api_key and _api_key != "your-langchain-api-key-here" else "missing",
            _tracing_enabled,
        )
except ImportError:
    logger.info("⚠️ LangSmith not installed — tracing disabled")


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is active."""
    return _LANGSMITH_AVAILABLE


def trace(
    *,
    name: str,
    run_type: str = "chain",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """Decorator to trace a function in LangSmith.

    If LangSmith is not available, returns the function unchanged.

    Parameters
    ----------
    name : str
        Human-readable name shown in LangSmith UI.
    run_type : str
        One of "chain", "llm", "tool", "retriever".
    tags : list[str], optional
        Tags for filtering in LangSmith.
    metadata : dict, optional
        Extra metadata attached to the run.
    """
    def decorator(fn: Callable) -> Callable:
        if _LANGSMITH_AVAILABLE and _traceable is not None:
            return _traceable(
                name=name,
                run_type=run_type,
                tags=tags or [],
                metadata=metadata or {},
            )(fn)
        return fn
    return decorator


def get_langsmith_config(run_name: str, tags: list[str] | None = None) -> dict:
    """Return a config dict to pass to LangChain .invoke() calls.

    This attaches a human-readable run_name to the LLM call so it
    appears clearly in the LangSmith trace tree.

    Usage:
        llm.invoke(messages, config=get_langsmith_config("Intent Classification"))
    """
    config: dict[str, Any] = {
        "run_name": run_name,
    }
    if tags:
        config["tags"] = tags
    return config
