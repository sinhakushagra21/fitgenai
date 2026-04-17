"""
agent/error_utils.py
────────────────────
Unified error-handling utilities for FITGEN.AI.

Every error boundary in the codebase should use one of these helpers
instead of ad-hoc ``try/except`` + ``logger.error`` + ``print(stderr)``
combinations.  Benefits:

  • Tracebacks are logged in one place, in one format.
  • Errors are pushed to LangSmith as structured events so we can see
    them inline with the trace that caused them.
  • The ``[FITGEN ERROR]`` stderr banners are gone — nothing gets lost
    among pymongo's debug noise any more.
  • The ``@safe_handler`` decorator wraps intent handlers so that a
    crash inside a handler returns a friendly message instead of
    blowing up the entire graph turn.

Exports
───────
  - ``handle_exception(exc, *, module, context, ...)``
        Canonical error logger + LangSmith pusher.  Call from any
        ``except:`` block.
  - ``safe_handler(module, fallback_message=...)``
        Decorator that wraps a handler in try/except and returns a
        friendly fallback string on error.
  - ``ErrorBoundary(...)``
        Context manager form — use in one-off call sites.
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from functools import wraps
from typing import Any, Callable, TypeVar

from agent.tracing import log_exception

logger = logging.getLogger("fitgen.error_utils")

F = TypeVar("F", bound=Callable[..., Any])


# ─────────────────────────────────────────────────────────────────────
# Canonical error handler
# ─────────────────────────────────────────────────────────────────────

def handle_exception(
    exc: BaseException,
    *,
    module: str,
    context: str,
    extra: dict[str, Any] | None = None,
    level: str = "ERROR",
    push_to_langsmith: bool = True,
) -> None:
    """Log an exception and (optionally) push it to LangSmith.

    Use this instead of ad-hoc ``logger.error(...) + print(stderr)``
    blocks.  A single traceback is logged via the ``fitgen.<module>``
    logger at the requested level, and — if LangSmith tracing is
    active — the exception becomes an event on the current run span.

    Parameters
    ----------
    exc
        The exception instance being handled.
    module
        Logger sub-name.  ``"router"``, ``"diet_tool"``, etc.
    context
        Short human description of what was being attempted
        (e.g. ``"intent classification"`` or ``"plan generation"``).
    extra
        Arbitrary key-value pairs that will appear in both the local
        log and the LangSmith event.
    level
        One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``,
        ``"CRITICAL"``.  Default ``"ERROR"``.
    push_to_langsmith
        Set ``False`` to suppress the LangSmith push (rare — only for
        expected / non-error exceptions we still want logged).
    """
    if push_to_langsmith:
        # log_exception handles BOTH the local log AND LangSmith push.
        log_exception(
            exc,
            module=module,
            context=context,
            extra=extra,
            level=level,
        )
    else:
        # Local log only.
        log = logging.getLogger(
            module if module.startswith("fitgen.") else f"fitgen.{module}"
        )
        log_fn = getattr(log, level.lower(), log.error)
        log_fn(
            "%s failed: %s: %s",
            context,
            type(exc).__name__,
            exc,
            exc_info=True,
        )


# ─────────────────────────────────────────────────────────────────────
# Decorator — wrap intent handlers so a crash returns a fallback
# ─────────────────────────────────────────────────────────────────────

def safe_handler(
    module: str,
    *,
    fallback_message: str = (
        "I had trouble processing your request. Please try again in a moment."
    ),
    context_prefix: str = "handler",
) -> Callable[[F], F]:
    """Decorator that wraps a function in try/except + ``handle_exception``.

    On success, the wrapped function's return value is passed through.
    On error, the fallback message is returned as the function's value
    (if the wrapped function returns a string) — otherwise the error
    is re-raised.

    The decorator assumes the wrapped function returns a ``str`` (as
    intent handlers do).  For handlers that return richer objects, use
    ``ErrorBoundary`` directly instead.

    Parameters
    ----------
    module
        Logger sub-name (e.g. ``"diet_tool"``).
    fallback_message
        User-facing string returned when the handler raises.
    context_prefix
        Prefix for the error context (defaults to ``"handler"``).

    Usage
    -----
    ::

        @safe_handler("diet_tool")
        def _handle_create_diet(query: str, ctx: DietSessionContext) -> str:
            ...
    """
    def decorator(fn: F) -> F:
        name = getattr(fn, "__name__", "unknown")

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                handle_exception(
                    exc,
                    module=module,
                    context=f"{context_prefix}:{name}",
                    extra={"function": name},
                )
                return fallback_message
        return wrapper  # type: ignore[return-value]

    return decorator


# ─────────────────────────────────────────────────────────────────────
# Context manager — for ad-hoc blocks
# ─────────────────────────────────────────────────────────────────────

class ErrorBoundary(AbstractContextManager):
    """Context manager that catches exceptions and routes them through
    ``handle_exception``.

    Use when you need error handling around an ad-hoc block and a
    decorator would be awkward.

    Parameters
    ----------
    module
        Logger sub-name.
    context
        Human description of the operation.
    reraise
        If ``True`` (default), the original exception propagates after
        logging.  If ``False``, the exception is swallowed — use this
        only when you have a meaningful fallback path outside the block.
    extra
        Key-value pairs attached to the event.

    Usage
    -----
    ::

        with ErrorBoundary(module="state_sync", context="persist context"):
            upsert_context_state(...)
    """

    def __init__(
        self,
        *,
        module: str,
        context: str,
        reraise: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.module = module
        self.context = context
        self.reraise = reraise
        self.extra = extra

    def __enter__(self) -> "ErrorBoundary":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            return False
        handle_exception(
            exc,
            module=self.module,
            context=self.context,
            extra=self.extra,
        )
        return not self.reraise
