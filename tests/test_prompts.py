"""
tests/test_prompts.py
─────────────────────
Regression tests for prompt templates.
"""

import pytest

from agent.prompts.base_prompts import BASE_PROMPTS
from agent.prompts.techniques import TECHNIQUE_KEYS, TECHNIQUE_META


class TestBasePrompts:
    def test_all_techniques_present(self):
        """BASE_PROMPTS has one entry per technique key."""
        for key in TECHNIQUE_KEYS:
            assert key in BASE_PROMPTS, f"Missing prompt for technique: {key}"

    def test_no_extra_keys(self):
        """No unexpected keys in BASE_PROMPTS."""
        assert set(BASE_PROMPTS.keys()) == set(TECHNIQUE_KEYS)

    def test_identity_section_present(self):
        """Every prompt contains the <identity> preamble."""
        for key, prompt in BASE_PROMPTS.items():
            assert "<identity>" in prompt, f"Missing <identity> in {key}"

    def test_safety_section_present(self):
        """Every prompt contains the safety guardrails."""
        for key, prompt in BASE_PROMPTS.items():
            assert "<safety_and_guardrails>" in prompt, f"Missing safety section in {key}"

    def test_no_gpt5_references(self):
        """Regression: no prompt should reference the nonexistent gpt-5 model."""
        for key, prompt in BASE_PROMPTS.items():
            assert "gpt-5" not in prompt.lower(), f"Found 'gpt-5' reference in {key}"


class TestTechniqueMeta:
    def test_all_techniques_have_meta(self):
        """Every TECHNIQUE_KEY has a corresponding TECHNIQUE_META entry."""
        for key in TECHNIQUE_KEYS:
            assert key in TECHNIQUE_META, f"Missing TECHNIQUE_META for: {key}"

    def test_meta_has_required_keys(self):
        """Each TECHNIQUE_META entry has label, icon, color, description."""
        required = {"label", "icon", "color", "description"}
        for key, meta in TECHNIQUE_META.items():
            missing = required - set(meta.keys())
            assert not missing, f"TECHNIQUE_META[{key}] missing: {missing}"
