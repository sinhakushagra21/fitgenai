"""
tests/test_config.py
────────────────────
Tests for centralized configuration.
"""

import os
from unittest.mock import patch

import pytest


class TestConfig:
    def test_default_model_fallback(self):
        """DEFAULT_MODEL falls back to gpt-4o-mini when env var unset."""
        with patch.dict(os.environ, {}, clear=True):
            # Re-import to pick up patched env
            import importlib
            import agent.config
            importlib.reload(agent.config)
            assert agent.config.DEFAULT_MODEL == "gpt-4o-mini"

    def test_default_model_from_env(self):
        """DEFAULT_MODEL reads from FITGEN_LLM_MODEL env var."""
        with patch.dict(os.environ, {"FITGEN_LLM_MODEL": "gpt-4o"}):
            import importlib
            import agent.config
            importlib.reload(agent.config)
            assert agent.config.DEFAULT_MODEL == "gpt-4o"
