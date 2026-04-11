"""
tests/test_config.py
────────────────────
Tests for centralized configuration — multi-model architecture.
"""

import os
from unittest.mock import patch

import pytest


class TestConfig:
    def test_plan_model_default(self):
        """PLAN_MODEL falls back to gpt-5.1 when env var unset."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            import agent.config
            importlib.reload(agent.config)
            assert agent.config.PLAN_MODEL == "gpt-5.1"
            assert agent.config.DEFAULT_MODEL == "gpt-5.1"

    def test_fast_model_default(self):
        """FAST_MODEL falls back to gpt-4.1-mini when env var unset."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            import agent.config
            importlib.reload(agent.config)
            assert agent.config.FAST_MODEL == "gpt-4.1-mini"

    def test_plan_model_from_env(self):
        """PLAN_MODEL reads from FITGEN_PLAN_MODEL env var."""
        with patch.dict(os.environ, {"FITGEN_PLAN_MODEL": "gpt-4o"}):
            import importlib
            import agent.config
            importlib.reload(agent.config)
            assert agent.config.PLAN_MODEL == "gpt-4o"
            assert agent.config.DEFAULT_MODEL == "gpt-4o"

    def test_fast_model_from_env(self):
        """FAST_MODEL reads from FITGEN_FAST_MODEL env var."""
        with patch.dict(os.environ, {"FITGEN_FAST_MODEL": "gpt-4.1-nano"}):
            import importlib
            import agent.config
            importlib.reload(agent.config)
            assert agent.config.FAST_MODEL == "gpt-4.1-nano"
