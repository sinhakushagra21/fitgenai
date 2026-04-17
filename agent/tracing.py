"""
agent/tracing.py
────────────────
LangSmith tracing configuration for FITGEN.AI.

Provides structured, human-readable traces of the entire pipeline:
  • Router (intent classification, direct response)
  • Diet / Workout tool flows
  • Profile Extraction
  • Plan Generation (JSON + Markdown)
  • Plan Validation
  • YouTube Enrichment
  • RAG Queries

Exports
───────
  - ``is_tracing_enabled()``       — check if LangSmith is active.
  - ``@trace(name=..., run_type=..., tags=..., metadata=...)``
                                   — decorate a function for tracing.
  - ``get_langsmith_config(run_name, tags=...)``
                                   — config dict for LangChain ``.invoke()``.
  - ``log_event(name, **kwargs)``  — push a named event to the current
                                   trace span + local log in one call.
  - ``log_exception(exc, *, module, context=None, run_name=None)``
                                   — push an exception event to LangSmith,
                                   log the traceback locally.
"""

from __future__ import annotations

import logging
import os
import traceback
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger("fitgen.tracing")

# ── LangSmith availability check ────────────────────────────────
_LANGSMITH_AVAILABLE = False
_traceable = None
_get_current_run_tree = None
_Client = None

try:
    from langsmith import traceable as _ls_traceable
    from langsmith.run_helpers import get_current_run_tree as _ls_get_current_run_tree

    try:
        from langsmith import Client as _LSClient  # type: ignore
    except ImportError:
        _LSClient = None

    _api_key = os.getenv("LANGCHAIN_API_KEY", "")
    _tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"

    if _api_key and _api_key != "your-langchain-api-key-here" and _tracing_enabled:
        _LANGSMITH_AVAILABLE = True
        _traceable = _ls_traceable
        _get_current_run_tree = _ls_get_current_run_tree
        _Client = _LSClient
        logger.info(
            "LangSmith tracing ENABLED (project: %s)",
            os.getenv("LANGCHAIN_PROJECT", "default"),
        )
    else:
        logger.info(
            "LangSmith tracing DISABLED (API key: %s, tracing_v2: %s)",
            "set" if _api_key and _api_key != "your-langchain-api-key-here" else "missing",
            _tracing_enabled,
        )
except ImportError:
    logger.info("LangSmith not installed — tracing disabled")


# ── Client (lazy-initialised singleton) ──────────────────────────────
_ls_client_singleton: Any = None


def _get_ls_client() -> Any | None:
    """Return a cached ``langsmith.Client`` or ``None`` if unavailable."""
    global _ls_client_singleton
    if not _LANGSMITH_AVAILABLE or _Client is None:
        return None
    if _ls_client_singleton is None:
        try:
            _ls_client_singleton = _Client()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not initialise LangSmith Client: %s", exc)
            _ls_client_singleton = None
    return _ls_client_singleton


# =====================================================================
# Public API
# =====================================================================

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
    name
        Human-readable name shown in the LangSmith UI.
    run_type
        One of ``"chain"``, ``"llm"``, ``"tool"``, ``"retriever"``.
    tags
        Tags for filtering in LangSmith.
    metadata
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


def get_langsmith_config(
    run_name: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Return a config dict to pass to LangChain ``.invoke()`` calls.

    Attaches a human-readable run_name to the LLM call so it appears
    clearly in the LangSmith trace tree.

    Usage::

        llm.invoke(messages, config=get_langsmith_config("Intent Classification"))
    """
    config: dict[str, Any] = {"run_name": run_name}
    if tags:
        config["tags"] = tags
    return config


# ─────────────────────────────────────────────────────────────────────
# Event push helpers  (NEW)
# ─────────────────────────────────────────────────────────────────────

def log_event(
    name: str,
    *,
    level: str = "INFO",
    module: str = "fitgen",
    **fields: Any,
) -> None:
    """Emit a structured event to both the local log and LangSmith.

    The event is logged via the standard ``logging`` module using the
    ``fitgen.<module>`` logger, and — if tracing is enabled — also
    pushed onto the current LangSmith run tree as metadata so that it
    appears inline with the trace.

    Parameters
    ----------
    name
        Short event name (``"plan_generated"``, ``"sync_failed"``, etc.).
    level
        Log level string: DEBUG / INFO / WARNING / ERROR / CRITICAL.
    module
        Logger sub-name (prepended with ``"fitgen."``).
    **fields
        Arbitrary key-value pairs attached to the event.

    Notes
    -----
    This helper never raises — if LangSmith push fails, it's silently
    dropped after a debug log.  Local logging is always performed.
    """
    # ── 1. Local log ─────────────────────────────────────────────
    log = logging.getLogger(
        module if module.startswith("fitgen.") else f"fitgen.{module}"
    )
    log_fn = getattr(log, level.lower(), log.info)
    if fields:
        rendered = " ".join(f"{k}={_render_field(v)}" for k, v in fields.items())
        log_fn("[event:%s] %s", name, rendered)
    else:
        log_fn("[event:%s]", name)

    # ── 2. LangSmith push ────────────────────────────────────────
    if not _LANGSMITH_AVAILABLE or _get_current_run_tree is None:
        return

    try:
        run_tree = _get_current_run_tree()
        if run_tree is None:
            return
        # Attach as metadata so it shows up in the LangSmith UI.
        existing = dict(getattr(run_tree, "extra", {}) or {})
        events = list(existing.get("events") or [])
        events.append({
            "name": name,
            "level": level,
            "module": module,
            **{k: _serialise(v) for k, v in fields.items()},
        })
        existing["events"] = events
        run_tree.extra = existing
    except Exception as exc:  # noqa: BLE001
        logger.debug("log_event LangSmith push failed (non-fatal): %s", exc)


def log_exception(
    exc: BaseException,
    *,
    module: str,
    context: str | None = None,
    run_name: str | None = None,
    extra: dict[str, Any] | None = None,
    level: str = "ERROR",
) -> None:
    """Log an exception locally (with traceback) and push to LangSmith.

    Use this at the error boundary of every tool/handler/LLM call so
    that failures are visible in both logs and LangSmith traces.

    Parameters
    ----------
    exc
        The caught exception instance.
    module
        Logger sub-name (e.g. ``"router"``, ``"diet_tool"``,
        ``"rag_tool"``).  Prepended with ``"fitgen."``.
    context
        Short human-readable description of what was being attempted
        (e.g. ``"intent classification"``, ``"plan generation"``).
    run_name
        Optional LangSmith run name for the failed span.
    extra
        Arbitrary key-value pairs to attach to the event.
    level
        Log level (default ``"ERROR"``).
    """
    log = logging.getLogger(
        module if module.startswith("fitgen.") else f"fitgen.{module}"
    )
    log_fn = getattr(log, level.lower(), log.error)

    ctx_str = context or "operation"
    log_fn(
        "%s failed: %s: %s",
        ctx_str,
        type(exc).__name__,
        exc,
        exc_info=True,
    )

    # Push to LangSmith as an event.
    fields: dict[str, Any] = {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "context": ctx_str,
    }
    if run_name:
        fields["run_name"] = run_name
    if extra:
        fields.update(extra)

    # Use log_event for LangSmith push — but DON'T log locally again.
    if _LANGSMITH_AVAILABLE and _get_current_run_tree is not None:
        try:
            run_tree = _get_current_run_tree()
            if run_tree is not None:
                existing = dict(getattr(run_tree, "extra", {}) or {})
                events = list(existing.get("events") or [])
                events.append({
                    "name": "exception",
                    "level": level,
                    "module": module,
                    "traceback": traceback.format_exc(),
                    **{k: _serialise(v) for k, v in fields.items()},
                })
                existing["events"] = events
                run_tree.extra = existing
        except Exception as push_exc:  # noqa: BLE001
            logger.debug("log_exception LangSmith push failed: %s", push_exc)


# =====================================================================
# Internal helpers
# =====================================================================

def _render_field(value: Any) -> str:
    """Render a field value for human-readable log output."""
    if isinstance(value, str):
        return value if len(value) < 80 else value[:77] + "..."
    if isinstance(value, (int, float, bool)):
        return str(value)
    if value is None:
        return "None"
    # Lists/dicts: show length + type
    try:
        return f"{type(value).__name__}(len={len(value)})"  # type: ignore[arg-type]
    except TypeError:
        return repr(value)[:80]


def _serialise(value: Any) -> Any:
    """Best-effort JSON-compatible serialisation for LangSmith metadata."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        import json
        return json.loads(json.dumps(value, default=str))
    except Exception:  # noqa: BLE001
        return repr(value)[:500]
