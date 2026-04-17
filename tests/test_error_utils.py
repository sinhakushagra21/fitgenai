"""
tests/test_error_utils.py
─────────────────────────
Unit tests for the unified error-handling layer.

Covers:
  • ``handle_exception`` — local log path (LangSmith push is a no-op when
    tracing is disabled in tests, so we only assert local behaviour).
  • ``@safe_handler`` — returns fallback on failure, passes through on success.
  • ``ErrorBoundary`` — catches and optionally swallows exceptions.
"""

from __future__ import annotations

import logging

import pytest

from agent.error_utils import ErrorBoundary, handle_exception, safe_handler


# ─────────────────────────────────────────────────────────────────────
# handle_exception
# ─────────────────────────────────────────────────────────────────────

def test_handle_exception_logs_locally(caplog):
    """handle_exception must log a traceback via the fitgen.<module> logger."""
    with caplog.at_level(logging.ERROR, logger="fitgen.unit_test"):
        try:
            raise ValueError("boom")
        except ValueError as exc:
            handle_exception(
                exc,
                module="unit_test",
                context="test case",
                push_to_langsmith=False,
            )

    assert any("test case failed" in r.getMessage() for r in caplog.records)
    assert any(r.levelno == logging.ERROR for r in caplog.records)
    # Traceback must be attached.
    assert any(r.exc_info is not None for r in caplog.records)


def test_handle_exception_respects_level(caplog):
    """Level override must be honoured (WARNING instead of ERROR)."""
    with caplog.at_level(logging.WARNING, logger="fitgen.unit_test"):
        try:
            raise RuntimeError("warn-me")
        except RuntimeError as exc:
            handle_exception(
                exc,
                module="unit_test",
                context="warn ctx",
                level="WARNING",
                push_to_langsmith=False,
            )

    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert not any(r.levelno == logging.ERROR for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────
# @safe_handler
# ─────────────────────────────────────────────────────────────────────

def test_safe_handler_passes_through_on_success():
    """A healthy handler must return its real result untouched."""
    @safe_handler("unit_test")
    def happy(x: int) -> str:
        return f"ok:{x}"

    assert happy(42) == "ok:42"


def test_safe_handler_returns_fallback_on_error(caplog):
    """A crashing handler must return the fallback string."""
    @safe_handler("unit_test", fallback_message="friendly fallback")
    def crashy() -> str:
        raise RuntimeError("kaboom")

    with caplog.at_level(logging.ERROR, logger="fitgen.unit_test"):
        result = crashy()

    assert result == "friendly fallback"
    assert any("handler:crashy" in r.getMessage() for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────
# ErrorBoundary
# ─────────────────────────────────────────────────────────────────────

def test_error_boundary_reraises_by_default(caplog):
    """Default behaviour: log + re-raise so caller sees the failure."""
    with caplog.at_level(logging.ERROR, logger="fitgen.unit_test"):
        with pytest.raises(ValueError):
            with ErrorBoundary(module="unit_test", context="boundary-ctx"):
                raise ValueError("propagate")


def test_error_boundary_swallows_when_reraise_false(caplog):
    """``reraise=False`` must swallow the exception and log it once."""
    with caplog.at_level(logging.ERROR, logger="fitgen.unit_test"):
        # Must NOT raise.
        with ErrorBoundary(
            module="unit_test",
            context="swallow-ctx",
            reraise=False,
        ):
            raise KeyError("silent")

    assert any("swallow-ctx" in r.getMessage() for r in caplog.records)


def test_error_boundary_no_exception_is_noop():
    """Normal flow through the boundary must be a straight pass-through."""
    with ErrorBoundary(module="unit_test", context="happy path"):
        value = 2 + 2
    assert value == 4
