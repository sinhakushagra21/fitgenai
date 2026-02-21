"""
agent/prompts/techniques.py
────────────────────────────
Metadata for the 5 prompt engineering techniques used in FITGEN.AI.
Used by tools (for labelling JSON keys) and by the Streamlit UI
(for tab labels, descriptions, and badge colours).
"""

from __future__ import annotations

# Ordered list of technique keys — order determines tab order in UI
TECHNIQUE_KEYS: list[str] = [
    "zero_shot",
    "few_shot",
    "cot",
    "analogical",
    "generate_knowledge",
]

# Human-readable display info per technique
TECHNIQUE_META: dict[str, dict] = {
    "zero_shot": {
        "label":       "Zero-Shot",
        "icon":        "🎯",
        "color":       "#546e7a",
        "description": (
            "No examples, no explicit reasoning instructions. "
            "The model relies purely on its pre-trained knowledge and the "
            "role definition to generate a response."
        ),
    },
    "few_shot": {
        "label":       "Few-Shot",
        "icon":        "📖",
        "color":       "#1565c0",
        "description": (
            "2–3 ideal example exchanges are embedded in the prompt. "
            "The model mirrors their structure, depth, and tone "
            "without being told *how* to think."
        ),
    },
    "cot": {
        "label":       "Chain-of-Thought",
        "icon":        "🔗",
        "color":       "#6a1b9a",
        "description": (
            "The model is explicitly instructed to reason step-by-step "
            "before presenting its answer, surfacing its internal logic."
        ),
    },
    "analogical": {
        "label":       "Analogical",
        "icon":        "🔍",
        "color":       "#e65100",
        "description": (
            "The model is instructed to explain concepts using real-world "
            "analogies, making complex ideas immediately relatable."
        ),
    },
    "generate_knowledge": {
        "label":       "Generate-Knowledge",
        "icon":        "🧠",
        "color":       "#2e7d32",
        "description": (
            "The model first generates 2–3 relevant scientific facts, "
            "then uses them as the factual basis for its answer."
        ),
    },
}
