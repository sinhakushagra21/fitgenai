"""
agent/logging_config.py
───────────────────────
Central logging configuration for FITGEN.AI.

Produces a compact, human-readable log format with optional ANSI colours
for TTY output.  Call :func:`setup_logging` once at application startup
(e.g. at the top of ``streamlit_app.py`` or ``app.py``).

Format
──────
    HH:MM:SS  LEVEL  [module]  message

Example::

    14:32:17  INFO   [fitgen.router]  Query: create a diet plan
    14:32:18  WARN   [fitgen.diet_tool]  No handler for intent=foo
    14:32:18  ERROR  [fitgen.llm_helpers]  generate_plan failed (domain=diet)

Why a separate module?
  - Single source of truth for log format, level, and noisy-logger muting.
  - Easy to adjust in one place without touching every file.
  - Makes testing simpler — just call ``setup_logging()`` from a fixture.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

# ── Default log levels ───────────────────────────────────────────────
_DEFAULT_ROOT_LEVEL = os.getenv("FITGEN_LOG_LEVEL", "INFO").upper()
_DEFAULT_FITGEN_LEVEL = os.getenv("FITGEN_APP_LOG_LEVEL", _DEFAULT_ROOT_LEVEL).upper()

# Libraries that spam the console at INFO level — silence them to WARNING
# so our own logs don't drown.  Add more here as needed.
_NOISY_LOGGERS: Final[tuple[str, ...]] = (
    "pymongo",
    "pymongo.topology",
    "pymongo.serverSelection",
    "pymongo.command",
    "httpx",
    "httpcore",
    "openai",
    "openai._base_client",
    "urllib3",
    "faiss",
    "watchdog",
    "langsmith",
    "langsmith.client",
)

# ── ANSI colour codes (TTY only) ─────────────────────────────────────
_COLOURS: Final[dict[str, str]] = {
    "DEBUG":    "\033[37m",      # grey
    "INFO":     "\033[36m",      # cyan
    "WARNING":  "\033[33m",      # yellow
    "ERROR":    "\033[31m",      # red
    "CRITICAL": "\033[1;31m",    # bold red
}
_RESET = "\033[0m"

# Level-name → 5-char label so columns line up.
_LABEL: Final[dict[str, str]] = {
    "DEBUG":    "DEBUG",
    "INFO":     "INFO ",
    "WARNING":  "WARN ",
    "ERROR":    "ERROR",
    "CRITICAL": "CRIT ",
}


class FitgenFormatter(logging.Formatter):
    """Custom formatter with aligned columns and optional ANSI colour.

    Output form (colour escape codes omitted here for clarity)::

        14:32:17  INFO   [fitgen.router]  Query received: create a diet plan
    """

    def __init__(self, *, use_colour: bool) -> None:
        super().__init__()
        self._use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        label = _LABEL.get(record.levelname, record.levelname[:5].ljust(5))
        name = record.name
        msg = record.getMessage()

        if self._use_colour:
            colour = _COLOURS.get(record.levelname, "")
            label = f"{colour}{label}{_RESET}"
            # Subtle dim for the module name
            name = f"\033[2m{name}\033[22m"

        line = f"{ts}  {label}  [{name}]  {msg}"

        if record.exc_info:
            # Traceback on following lines, slightly indented
            tb = self.formatException(record.exc_info)
            line = f"{line}\n{tb}"
        elif record.stack_info:
            line = f"{line}\n{record.stack_info}"

        return line


def setup_logging(
    *,
    level: str | int | None = None,
    fitgen_level: str | int | None = None,
    use_colour: bool | None = None,
    stream=None,
) -> None:
    """Configure the root logger with FITGEN.AI's human-readable format.

    Idempotent — safe to call multiple times.  Previous handlers are
    cleared so that a re-invocation (e.g. during Streamlit rerun) doesn't
    duplicate log lines.

    Parameters
    ----------
    level
        Root logger level.  Defaults to ``$FITGEN_LOG_LEVEL`` or ``"INFO"``.
    fitgen_level
        Level for the ``fitgen.*`` logger tree.  Defaults to the root level.
    use_colour
        Whether to emit ANSI colour codes.  ``None`` (default) auto-detects
        based on whether the stream is a TTY.
    stream
        Output stream.  Defaults to ``sys.stderr``.
    """
    stream = stream or sys.stderr
    root_level = level if level is not None else _DEFAULT_ROOT_LEVEL
    app_level = fitgen_level if fitgen_level is not None else _DEFAULT_FITGEN_LEVEL

    if use_colour is None:
        use_colour = hasattr(stream, "isatty") and stream.isatty()

    formatter = FitgenFormatter(use_colour=use_colour)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Remove any existing handlers so re-invocations don't duplicate output.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(root_level)

    # Dedicated level for our own namespace.
    fitgen_logger = logging.getLogger("fitgen")
    fitgen_logger.setLevel(app_level)
    fitgen_logger.propagate = True

    # Mute noisy third-party libraries.
    for name in _NOISY_LOGGERS:
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        lg.propagate = True

    logging.getLogger("fitgen.logging_config").info(
        "Logging initialised (root=%s, fitgen=%s, colour=%s)",
        root_level, app_level, use_colour,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``fitgen.<name>``.

    Convenience so module files don't need to remember the prefix.
    Example::

        log = get_logger("router")     # ← "fitgen.router"
        log = get_logger("tools.diet") # ← "fitgen.tools.diet"
    """
    if not name.startswith("fitgen."):
        name = f"fitgen.{name}"
    return logging.getLogger(name)
