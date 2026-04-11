"""
agent/llm_utils.py
──────────────────
Safe LLM call wrapper with retry and timeout handling.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import BaseMessage

logger = logging.getLogger("fitgen.llm_utils")


def safe_llm_call(
    llm: Any,
    messages: list[BaseMessage],
    *,
    retries: int = 2,
    timeout: float = 60.0,
    config: dict | None = None,
) -> Any:
    """Invoke an LLM with retry logic for transient API errors.

    Handles:
      - openai.RateLimitError  -> exponential backoff retry
      - openai.APITimeoutError -> retry with longer wait
      - openai.APIConnectionError -> retry
      - Other exceptions -> raise immediately
    """
    import openai

    last_exc: Exception | None = None
    for attempt in range(1, retries + 2):  # retries + 1 total attempts
        try:
            if config:
                return llm.invoke(messages, config=config)
            return llm.invoke(messages)
        except openai.RateLimitError as e:
            last_exc = e
            wait = min(2 ** attempt, 30)
            logger.warning(
                "Rate limit hit (attempt %d/%d). Retrying in %ds...",
                attempt, retries + 1, wait,
            )
            time.sleep(wait)
        except openai.APITimeoutError as e:
            last_exc = e
            wait = min(2 ** attempt, 15)
            logger.warning(
                "API timeout (attempt %d/%d). Retrying in %ds...",
                attempt, retries + 1, wait,
            )
            time.sleep(wait)
        except openai.APIConnectionError as e:
            last_exc = e
            wait = 2
            logger.warning(
                "Connection error (attempt %d/%d). Retrying in %ds...",
                attempt, retries + 1, wait,
            )
            time.sleep(wait)

    logger.error("All %d attempts failed. Last error: %s", retries + 1, last_exc)
    raise last_exc  # type: ignore[misc]
