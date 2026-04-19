"""
agent/safety/guardrails.py
──────────────────────────
Deterministic input-filter guardrails for FITGEN.AI.

This module is the FIRST line of defence against prompt-injection,
system-prompt-leak attempts, and off-topic reframings (travel, tourism,
place recommendations, shopping, etc.).  It runs BEFORE the LLM router
classifier sees the user message, so a crafted message never reaches
the model at all — the user gets a canned, consistent refusal and the
incident is logged.

Why regex + canonical refusal (industry-standard defence-in-depth):
  * LLM instruction-following is probabilistic — a sufficiently clever
    reframing ("best place to roam NEAR THESE GYMS") can slip past
    system prompts.  A deterministic gate cannot be reframed.
  * Canonical refusal text means every block looks the same to the
    user — no information leakage about WHICH pattern tripped.
  * The LLM system prompts remain the second layer: if a message does
    get past regex, the model still has explicit "do not answer
    off-topic, even if framed as fitness-adjacent" instructions.

This file has NO side effects at import time and NO LLM dependencies.
It is safe to import from anywhere in the agent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("fitgen.safety")


# =====================================================================
# Canonical refusal strings
# =====================================================================
# One refusal per block-category.  Keep them short, friendly, and
# redirect the user back to the in-scope capabilities.  NEVER mention
# the specific pattern or rule that matched — that leaks information
# attackers can use to craft a bypass.
# =====================================================================

REFUSAL_OFF_TOPIC = (
    "I'm FITGEN.AI — your personal fitness coach. I can only help with "
    "workout plans, exercise programming, diet plans, nutrition, and "
    "evidence-based fitness knowledge. I can't help with travel, "
    "tourism, shopping, entertainment, or general-knowledge questions. "
    "What fitness or nutrition goal can I help you with?"
)

REFUSAL_PROMPT_LEAK = (
    "I can't share my internal instructions, system prompt, or "
    "configuration. I'm here to help you with workouts and nutrition — "
    "what's your fitness goal?"
)

REFUSAL_JAILBREAK = (
    "I'm FITGEN.AI — I stay in my role as a fitness and nutrition "
    "coach. I can't ignore my instructions, roleplay a different "
    "assistant, or operate in a different mode. Happy to help with "
    "your workout or meal planning though — what would you like to "
    "work on?"
)


class BlockReason(str, Enum):
    """Reason a message was blocked.  Logged; never shown to the user."""

    PROMPT_LEAK = "prompt_leak"
    JAILBREAK = "jailbreak"
    OFF_TOPIC = "off_topic"


@dataclass(frozen=True)
class GuardrailDecision:
    """Result of screening a single user message."""

    allowed: bool
    reason: BlockReason | None = None
    refusal: str | None = None


ALLOWED = GuardrailDecision(allowed=True)


# =====================================================================
# Pattern sets
# =====================================================================
# All patterns are compiled at module import (once) and matched
# case-insensitively against the RAW user message.  Each pattern is
# deliberately narrow so false-positives on legitimate fitness queries
# stay near zero.  When adding a pattern, ALWAYS add a test case in
# tests/test_guardrails.py covering (a) a malicious phrasing it should
# block and (b) a legitimate fitness phrasing it should NOT block.
# =====================================================================

# --- 1. System-prompt-leak attempts ----------------------------------
# Targets: "show me your system prompt", "what are your instructions",
#          "print your rules", "repeat your configuration verbatim",
#          "reveal your prompt", "dump your instructions".
_PROMPT_LEAK_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\b(show|reveal|print|display|share|give|tell|output|dump|leak|repeat|expose)\s+"
        r"(me|us|your|the)?\s*"
        r"(your|the)?\s*"
        r"(system\s+)?(prompt|instructions?|rules?|configuration|config|"
        r"guidelines?|directive|initial\s+message|hidden\s+message)",
        r"\bwhat\s+(is|are|were)\s+(your|the)\s+"
        r"(system\s+)?(prompt|instructions?|rules?|configuration|guidelines?)",
        r"\b(your|the)\s+(system\s+)?(prompt|instructions?)\s+"
        r"(says?|contains?|includes?|is|are)\b",
        r"\bverbatim\b.*\b(prompt|instructions?|rules?)\b",
        r"\b(prompt|instructions?|rules?)\b.*\bverbatim\b",
        r"\brepeat\b.*\b(above|before|earlier|previous)\b",
    )
)

# --- 2. Jailbreak / role-hijack attempts ------------------------------
# Targets: "ignore previous instructions", "you are now DAN",
#          "pretend you have no rules", "developer mode", "sudo mode",
#          "act as an unrestricted assistant".
_JAILBREAK_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bignore\s+(all\s+|the\s+)?(previous|prior|above|earlier|"
        r"your|any)\s+(instructions?|rules?|prompts?|directives?|guidelines?)",
        r"\bdisregard\s+(all\s+|the\s+)?(previous|prior|above|your)\s+"
        r"(instructions?|rules?)",
        r"\bforget\s+(everything|all|your\s+(instructions?|rules?|prompt))",
        r"\byou\s+are\s+(now\s+)?(in\s+)?(a\s+)?(?:DAN|developer\s+mode|"
        r"jailbroken|unrestricted|unfiltered|uncensored|sudo)\b",
        r"\byou\s+are\s+now\s+(in\s+)?(a\s+)?"
        r"(developer|debug|admin|god|sudo|jailbreak|unrestricted|"
        r"unfiltered)\s+mode\b",
        r"\b(enter|activate|enable|switch\s+to)\s+"
        r"(developer|debug|admin|god|sudo|jailbreak|unrestricted|"
        r"unfiltered)\s+mode\b",
        r"\bpretend\s+(you\s+(are|have)|to\s+be)\b.*\b("
        r"no\s+rules?|unrestricted|another\s+ai|different\s+assistant)",
        r"\bact\s+as\s+(if\s+you\s+(are|were)\s+)?(an?\s+)?"
        r"(unrestricted|unfiltered|uncensored|different|new)\s+(ai|assistant|model)",
        r"\bfrom\s+now\s+on\b.*\byou\s+(are|will|must|should)\b.*\b"
        r"(not|never)\b.*\b(refuse|decline|restricted)",
        r"\boverride\s+(your|the|all)\s+(safety|guardrails?|rules?|instructions?)",
        r"\bbypass\s+(your|the|all)\s+(safety|guardrails?|filters?|restrictions?)",
        r"\brole[-\s]?play\s+as\b.*\b(?:unrestricted|uncensored|unfiltered|"
        r"jailbroken|evil|malicious)",
    )
)

# --- 3. Off-topic reframings -----------------------------------------
# These are the sneaky ones: the user adds a fitness-sounding hook
# ("near gyms", "for runners", "while on my diet") to a travel / tourism
# / shopping / entertainment question.  We block on the OFF-TOPIC
# intent even if a fitness keyword is present.
#
# IMPORTANT: we intentionally do NOT block patterns that are ambiguous
# but legitimate (e.g. "best foods for muscle gain", "gyms with
# functional training equipment"), because those ARE in-scope.  We
# target the tourism/sightseeing/travel-planning vocabulary specifically.
_OFF_TOPIC_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        # Tourism / sightseeing
        r"\b(best|top|good|nice|popular|must[-\s]?see|recommended)\s+"
        r"(places?|spots?|things?|stuff|attractions?|sights?|activities)\s+"
        r"(to\s+)?(visit|see|go|explore|tour|check\s+out|roam|wander|"
        r"hang\s+out|chill|sight[-\s]?see)",
        r"\b(places?|spots?|things?|attractions?)\s+to\s+"
        r"(visit|see|go|explore|tour|roam|wander|check\s+out)",
        r"\bwhere\s+(to|can\s+i|should\s+i)\s+"
        r"(go|visit|roam|wander|travel|explore|sight[-\s]?see|hang\s+out)",
        r"\btourist\s+(attractions?|spots?|places?|destinations?|areas?)",
        r"\bsight[-\s]?seeing\b",
        r"\bthings?\s+to\s+do\s+(in|around|near)\b",
        # Location + explicit ask for a specific place / city /
        # neighbourhood / gym chain.  Catches reframings like
        # "specific place in Europe I can train" or "which gym chains
        # in London".  We require BOTH a 'place/city/chain' word AND
        # a location-intent word, so legitimate queries about generic
        # gym types or equipment still pass.
        r"\b(specific|which|what|recommend|suggest|name|list|best|top)\b"
        r"[^.?!]{0,40}?\b"
        r"(place|places|spot|spots|city|cities|country|countries|"
        r"region|regions|neighbou?rhood|area|areas|gym\s+chains?|"
        r"gym\s+brands?)\b"
        r"[^.?!]{0,20}?\b(in|near|around|at|close\s+to)\s+\w+",
        r"\b(place|spot|city|country|region|gym\s+chain|gym\s+brand)s?\s+"
        r"(in|near|around|at)\s+(?-i:[A-Z])\w+",  # "places in Europe/London/Paris"
        # "best/suggest/recommend/which … gym(s) … in [Capitalised City]"
        # catches "best gym types or chains in London" where filler
        # words sit between "gym" and "chains".  Requires an uppercase
        # proper-noun location after "in/near/around" to keep generic
        # queries like "gyms with equipment" safe.
        r"\b(best|top|good|which|what|recommend|suggest|list|name)\b"
        r"[^.?!]{0,80}?\bgyms?\b"
        r"[^.?!]{0,80}?\b(in|near|around|at)\s+(?-i:[A-Z])\w+",
        # Travel planning
        r"\btravel\s+(guide|tips|itinerary|plan|advice|recommendations?)",
        r"\b(plan|planning)\s+(my|a|the)\s+(trip|vacation|holiday|visit)",
        r"\b(flight|hotel|accommodation|airbnb|hostel|booking|stay)"
        r"\s+(recommendations?|suggestions?|advice|options?|in|near|at)",
        # Shopping / entertainment / general lifestyle
        r"\b(best|top|good)\s+(restaurants?|cafes?|bars?|clubs?|"
        r"pubs?|nightlife|shopping|malls?|museums?|parks?\s+to\s+visit)",
        r"\b(movie|film|show|concert|game|book|novel)\s+"
        r"(recommendations?|suggestions?)",
    )
)


# =====================================================================
# Public API
# =====================================================================

def screen_user_message(text: str) -> GuardrailDecision:
    """Deterministically screen a raw user message.

    Returns a ``GuardrailDecision`` with ``allowed=False`` and a canned
    refusal if the message matches any block pattern.  Order of checks:

      1. Prompt-leak attempt   → REFUSAL_PROMPT_LEAK
      2. Jailbreak attempt     → REFUSAL_JAILBREAK
      3. Off-topic reframing   → REFUSAL_OFF_TOPIC

    Empty / whitespace-only / very short messages (<= 2 chars) are
    always allowed — the downstream router handles those.

    This function NEVER raises.  On any internal error it logs the
    exception and falls back to ``ALLOWED`` so guardrail bugs cannot
    lock legitimate users out of the app.
    """
    try:
        if not isinstance(text, str):
            return ALLOWED
        probe = text.strip()
        if len(probe) <= 2:
            return ALLOWED

        if any(p.search(probe) for p in _PROMPT_LEAK_PATTERNS):
            logger.info(
                "[Guardrail] Blocked prompt-leak attempt: %s", probe[:120]
            )
            return GuardrailDecision(
                allowed=False,
                reason=BlockReason.PROMPT_LEAK,
                refusal=REFUSAL_PROMPT_LEAK,
            )

        if any(p.search(probe) for p in _JAILBREAK_PATTERNS):
            logger.info(
                "[Guardrail] Blocked jailbreak attempt: %s", probe[:120]
            )
            return GuardrailDecision(
                allowed=False,
                reason=BlockReason.JAILBREAK,
                refusal=REFUSAL_JAILBREAK,
            )

        if any(p.search(probe) for p in _OFF_TOPIC_PATTERNS):
            logger.info(
                "[Guardrail] Blocked off-topic reframing: %s", probe[:120]
            )
            return GuardrailDecision(
                allowed=False,
                reason=BlockReason.OFF_TOPIC,
                refusal=REFUSAL_OFF_TOPIC,
            )

        return ALLOWED
    except Exception as exc:  # noqa: BLE001 — MUST NOT raise
        logger.warning(
            "[Guardrail] Screening raised %s; allowing message by default",
            exc,
        )
        return ALLOWED
