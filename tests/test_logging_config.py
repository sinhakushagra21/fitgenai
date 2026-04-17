"""
tests/test_logging_config.py
────────────────────────────
Tests for the central logging configuration.

These tests verify that:
  • ``setup_logging`` produces the expected human-readable format.
  • Repeated invocations are idempotent (no duplicate handlers).
  • Noisy third-party loggers are muted to WARNING.
  • ``get_logger`` always returns a logger under the ``fitgen.`` namespace.
"""

from __future__ import annotations

import io
import logging
import re

from agent.logging_config import get_logger, setup_logging


# ─────────────────────────────────────────────────────────────────────
# Format
# ─────────────────────────────────────────────────────────────────────

def test_setup_logging_produces_expected_format():
    """Log output should have the aligned HH:MM:SS LEVEL [module] message layout."""
    buf = io.StringIO()
    setup_logging(level="INFO", use_colour=False, stream=buf)

    log = get_logger("unit_test_format")
    log.info("hello world")

    output = buf.getvalue()
    # Format: "12:34:56  INFO   [fitgen.unit_test_format]  hello world"
    pattern = re.compile(
        r"\d{2}:\d{2}:\d{2}\s+INFO\s+\[fitgen\.unit_test_format\]\s+hello world"
    )
    assert pattern.search(output), f"Output did not match expected format:\n{output}"


# ─────────────────────────────────────────────────────────────────────
# Idempotency
# ─────────────────────────────────────────────────────────────────────

def test_setup_logging_is_idempotent():
    """Calling setup_logging twice must not duplicate handlers."""
    buf1 = io.StringIO()
    setup_logging(level="INFO", use_colour=False, stream=buf1)
    handler_count_1 = len(logging.getLogger().handlers)

    buf2 = io.StringIO()
    setup_logging(level="INFO", use_colour=False, stream=buf2)
    handler_count_2 = len(logging.getLogger().handlers)

    assert handler_count_1 == handler_count_2 == 1


# ─────────────────────────────────────────────────────────────────────
# Noisy-logger muting
# ─────────────────────────────────────────────────────────────────────

def test_noisy_loggers_are_muted():
    """pymongo / httpx / openai loggers must be capped at WARNING."""
    setup_logging(level="DEBUG", use_colour=False, stream=io.StringIO())

    for name in ("pymongo", "httpx", "openai", "urllib3", "langsmith"):
        assert logging.getLogger(name).level == logging.WARNING, (
            f"Expected {name} logger at WARNING, got "
            f"{logging.getLevelName(logging.getLogger(name).level)}"
        )


# ─────────────────────────────────────────────────────────────────────
# get_logger
# ─────────────────────────────────────────────────────────────────────

def test_get_logger_prefixes_namespace():
    assert get_logger("router").name == "fitgen.router"
    assert get_logger("tools.diet").name == "fitgen.tools.diet"
    # Already-prefixed names must pass through unchanged.
    assert get_logger("fitgen.already").name == "fitgen.already"
