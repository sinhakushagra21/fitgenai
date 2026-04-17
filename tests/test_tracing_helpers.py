"""
tests/test_tracing_helpers.py
─────────────────────────────
Tests for the ``log_event`` / ``log_exception`` helpers in agent.tracing.

LangSmith is expected to be disabled during tests (no API key), so we
only validate local logging behaviour — the LangSmith push path is
silently skipped, which is the documented safe default.
"""

from __future__ import annotations

import logging

from agent.tracing import log_event, log_exception


def test_log_event_logs_locally_with_fields(caplog):
    """log_event must render field=value pairs in the log message."""
    with caplog.at_level(logging.INFO, logger="fitgen.unit_test"):
        log_event(
            "thing_happened",
            module="unit_test",
            foo=1,
            bar="baz",
        )

    msgs = [r.getMessage() for r in caplog.records]
    joined = " ".join(msgs)
    assert "[event:thing_happened]" in joined
    assert "foo=1" in joined
    assert "bar=baz" in joined


def test_log_event_without_fields(caplog):
    """log_event with no fields must still emit the event header."""
    with caplog.at_level(logging.INFO, logger="fitgen.unit_test"):
        log_event("pulse", module="unit_test")

    assert any("[event:pulse]" in r.getMessage() for r in caplog.records)


def test_log_event_respects_level(caplog):
    """Level argument must route to the correct logger method."""
    with caplog.at_level(logging.WARNING, logger="fitgen.unit_test"):
        log_event("warned", level="WARNING", module="unit_test", detail="x")

    assert any(
        r.levelno == logging.WARNING and "[event:warned]" in r.getMessage()
        for r in caplog.records
    )


def test_log_exception_logs_traceback(caplog):
    """log_exception must log the exception with traceback attached."""
    with caplog.at_level(logging.ERROR, logger="fitgen.unit_test"):
        try:
            raise ValueError("bad input")
        except ValueError as exc:
            log_exception(exc, module="unit_test", context="parse")

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert error_records, "Expected at least one ERROR record"
    assert any("parse failed" in r.getMessage() for r in error_records)
    assert any(r.exc_info is not None for r in error_records)
