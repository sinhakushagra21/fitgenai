"""
agent/shared/llm_helpers.py
───────────────────────────
LLM interaction utilities shared across FITGEN.AI domain tools.

Contains:
  - _llm_json          : lightweight LLM call expecting JSON output
  - classify_intent    : domain-aware intent classifier (returns user_intent + reason)
  - extract_profile_updates / _with_fallback : LLM + rule-based profile extraction
  - answer_followup_question : contextual Q&A without regenerating the plan
  - generate_plan      : raw plan generation (markdown output)
  - generate_plan_as_json : plan generation with JSON output instruction appended
  - validate_plan_json : flexible validation of LLM JSON plan output
  - plan_json_to_markdown : convert JSON plan to readable markdown
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.config import DEFAULT_MODEL, FAST_MODEL
from agent.shared.types import SEX_MAP

logger = logging.getLogger("fitgen.llm_helpers")

# ── Model aliases ────────────────────────────────────────────────────
_PLAN_MODEL = DEFAULT_MODEL   # gpt-5.1 — expensive, high quality
_FAST_MODEL = FAST_MODEL      # gpt-4.1-mini — fast, cheap

__all__ = [
    "classify_intent",
    "extract_profile_updates",
    "extract_profile_updates_with_fallback",
    "answer_followup_question",
    "answer_plan_question",
    "generate_plan",
    "generate_plan_as_json",
    "validate_plan_json",
    "plan_json_to_markdown",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core LLM call
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _llm_json(system: str, user: str, *, retries: int = 2) -> dict[str, Any]:
    """Invoke LLM and parse response as JSON, with retry + regex fallback.

    Args:
        system: System prompt text.
        user: User message text.
        retries: Number of retry attempts on parse/API failure.

    Returns:
        Parsed JSON dict, or empty dict on total failure.
    """
    llm = ChatOpenAI(model=_FAST_MODEL, temperature=0)
    for attempt in range(retries + 1):
        try:
            resp = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            text = resp.content.strip()

            # Direct JSON parse
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Regex fallback: extract first {...} block
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

            logger.warning(
                "_llm_json attempt %d: failed to parse JSON from response",
                attempt,
            )
        except Exception as exc:
            logger.warning("_llm_json attempt %d error: %s", attempt, exc)

    logger.error("_llm_json: all %d attempts failed", retries + 1)
    return {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Intent Classification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def classify_intent(
    query: str,
    *,
    domain: str,
    valid_intents: list[str],
    step_completed: str | None,
    user_profile: dict[str, Any],
    pending_question: str | None = None,
    has_plan: bool = False,
) -> dict[str, str]:
    """Classify user intent via LLM with full workflow context.

    The LLM receives the complete workflow state — current step, profile
    completeness, whether a plan exists — so it can make accurate routing
    decisions without the handlers needing step-based overrides.

    The LLM returns exactly two keys:
      - user_intent: one of the valid_intents provided
      - reason: one-sentence explanation

    Args:
        query: The raw user message.
        domain: "diet" or "workout".
        valid_intents: List of valid intent strings for this domain
                       (e.g. DietIntent values).
        step_completed: Current step in the workflow (provides context).
        user_profile: Current user profile dict (may be partial).
        pending_question: The last question the system asked the user.
        has_plan: Whether a generated plan already exists.

    Returns:
        Dict with keys ``user_intent`` (str) and ``reason`` (str).
        Falls back to ``general_{domain}_query`` if classification fails.
    """
    profile_summary = (
        json.dumps(user_profile, indent=2) if user_profile else "(empty)"
    )
    # Count how many required fields are filled
    from agent.shared.types import DIET_REQUIRED_FIELDS, WORKOUT_REQUIRED_FIELDS
    _req_fields = DIET_REQUIRED_FIELDS if domain == "diet" else WORKOUT_REQUIRED_FIELDS
    _filled = sum(
        1 for f in _req_fields
        if f in user_profile and user_profile[f] not in (None, "")
    )
    _total = len(_req_fields)

    intents_str = ", ".join(f'"{i}"' for i in valid_intents)
    fallback_intent = f"general_{domain}_query"

    system = f"""\
You are an intent classifier for FITGEN.AI, a fitness coaching chatbot.
Your ONLY job is to classify the user's message into exactly one intent
based on the message content AND the full workflow context provided below.

Return ONLY valid JSON with exactly these two keys:
  "user_intent": exactly one of [{intents_str}]
  "reason": one-sentence explanation of your decision

═══════════════════════════════════════════════════════════
CURRENT WORKFLOW STATE
═══════════════════════════════════════════════════════════
  Domain: {domain}
  Current step completed: {step_completed or "(none — fresh conversation)"}
  Has existing plan: {"YES" if has_plan else "NO"}
  Profile completeness: {_filled}/{_total} required fields filled
  Pending question: {pending_question or "(none)"}
  User profile: {profile_summary}

═══════════════════════════════════════════════════════════
WORKFLOW STEPS & WHAT "YES" MEANS AT EACH STEP
═══════════════════════════════════════════════════════════
The {domain} workflow progresses through these steps in order:

Step 1: (none) — Fresh conversation, no workflow started.
  → "yes" / affirmation has no special meaning.
  → If user wants a new plan → "create_{domain}"

Step 2: "prompted_for_user_profile_data" — The system asked the user to
  provide profile details (name, age, height, weight, etc.).
  → User's next message should contain profile data fields.
  → "yes" / affirmation here → "create_{domain}" (continuing the create flow).
  → Profile data → "create_{domain}"
  → NOT "confirm_{domain}" — there is nothing to confirm yet.

Step 3: "user_profile_mapped" — The system mapped the user's data and
  showed them a summary. Asked "Reply yes to confirm, or share corrections."
  → "yes" / "confirm" / "looks good" → "create_{domain}"
     (The user is confirming their PROFILE is correct, so the system can
      proceed to GENERATE the plan. A plan does NOT exist yet.)
  → Corrections ("change age to 30", "I'm actually female") → "create_{domain}"
  → NOT "confirm_{domain}" — no plan has been generated to confirm!

Step 4: "{domain}_plan_generated" — A plan has been generated and shown.
  The system asked "Reply yes to confirm, or tell me what to change."
  → "yes" / "confirm" / "looks good" → "confirm_{domain}" ✓ (NOW it's correct!)
  → "change X" / "make it Y" / update requests → "update_{domain}"
  → Questions about the plan content ("what should I eat tomorrow",
    "show Monday's workout", "whats my routine") → "get_{domain}"
  → Generic knowledge questions ("is creatine safe") → "general_{domain}_query"

Step 5: "updated_{domain}_plan" — The plan was regenerated after changes.
  → Same as Step 4. "yes" → "confirm_{domain}". Changes → "update_{domain}".

Step 6: "{domain}_confirmed" — Plan is confirmed. Workflow is essentially done.
  → Sync requests → "sync_{domain}_to_google_calendar" / "sync_{domain}_to_google_fit"
  → "done" / decline sync → "general_{domain}_query"
  → "create a new plan" → "create_{domain}" (start over)
  → Questions about the plan ("what should I eat tomorrow", "show my
    plan", "whats my Monday workout") → "get_{domain}"

═══════════════════════════════════════════════════════════
CRITICAL RULES (override everything else)
═══════════════════════════════════════════════════════════

RULE 1 — "yes" does NOT always mean "confirm_{domain}":
  • At "user_profile_mapped" (no plan yet) → "create_{domain}"
  • At "prompted_for_user_profile_data" → "create_{domain}"
  • At "{domain}_plan_generated" or "updated_{domain}_plan" → "confirm_{domain}"
  • Use the step context to decide. NEVER default "yes" to "confirm_{domain}"
    without checking the current step.

RULE 2 — Profile data is always "create_{domain}" during create flow:
  Messages containing profile fields (name, age, height, weight, sex, goal,
  diet_preference, exercise_frequency, etc.) when step is "prompted_for_user_profile_data"
  or "user_profile_mapped" → "create_{domain}".

RULE 3 — Distinguish "get_{domain}" from "general_{domain}_query":
  If the user is asking about THEIR OWN existing plan — e.g. "what should
  I eat tomorrow", "what's my workout today", "show my Monday meals",
  "get my diet plan for tomorrow", "whats my routine" — that is "get_{domain}"
  (retrieving info from the stored plan).
  Only use "general_{domain}_query" for generic knowledge questions NOT about
  the user's plan — e.g. "how much protein do I need", "is creatine safe",
  "what exercises target glutes".

RULE 4 — Sync requests require a confirmed plan:
  "sync to calendar" / "sync to google fit" → only valid AFTER plan is confirmed.
  If has_plan is NO, classify as "general_{domain}_query" instead.

RULE 5 — "confirm_{domain}" requires a generated plan:
  If has_plan is NO, do NOT return "confirm_{domain}". The user might be
  saying "yes" to confirm profile data → return "create_{domain}" instead.

═══════════════════════════════════════════════════════════
GENERAL CLASSIFICATION (when no critical rule applies)
═══════════════════════════════════════════════════════════

• Wants a new plan → "create_{domain}"
• Wants to modify/change/update plan → "update_{domain}"
• Wants to delete plan → "delete_{domain}"
• Wants to see/retrieve plan OR asks about their own plan (e.g. "what should
  I eat tomorrow", "what's my workout today", "show me Monday's meals",
  "get my diet", "whats my routine") → "get_{domain}"
• Sync to Google Calendar → "sync_{domain}_to_google_calendar"
• Sync to Google Fit → "sync_{domain}_to_google_fit"
• General knowledge question NOT about user's plan, small talk → "general_{domain}_query"
"""

    data = _llm_json(system, query)

    # Extract and validate intent
    raw_intent = str(data.get("user_intent", fallback_intent)).strip()
    if raw_intent not in valid_intents:
        logger.warning(
            "[classify_intent] LLM returned invalid intent %r, falling back to %s",
            raw_intent,
            fallback_intent,
        )
        raw_intent = fallback_intent

    reason = str(data.get("reason", "")).strip()

    logger.info(
        "[classify_intent] domain=%s query=%r → user_intent=%s reason=%s",
        domain,
        query[:80],
        raw_intent,
        reason[:80] if reason else "(none)",
    )

    return {"user_intent": raw_intent, "reason": reason}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Profile Extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_profile_updates(
    query: str,
    allowed_fields: list[str],
) -> dict[str, Any]:
    """Use LLM to extract profile field values from a user message.

    Args:
        query: The raw user message.
        allowed_fields: Only return fields in this list.

    Returns:
        Dict of extracted field:value pairs (only non-empty values).
    """
    system = """\
Extract ONLY real user profile values from the message.
Return JSON only with keys among:
name, age, sex, height_cm, weight_kg, goal, activity_level,
diet_preference, foods_to_avoid, allergies, fitness_level, equipment,
workout_days, goal_weight, weight_loss_pace, job_type, exercise_frequency,
exercise_type, sleep_hours, stress_level, alcohol_intake, favourite_meals,
cooking_style, food_adventurousness, current_snacks, snack_reason,
snack_preference, late_night_snacking, experience_level,
training_days_per_week, session_duration, daily_steps.

CRITICAL RULES:
- If the message is a command or request (e.g. "create a diet plan",
  "make me a workout", "show my plan", "yes", "no", "confirm"),
  return an EMPTY JSON object {}.
- "name" must be an actual person's name (e.g. "Kushagra", "John").
  Do NOT map commands, sentences, or descriptions to "name".
- "goal" must be one of: fat loss, muscle gain, maintenance, performance.
  Do NOT confuse "create a diet plan" as a goal.
- If a value is absent or ambiguous, OMIT the key entirely.
- Normalize age to int, height_cm and weight_kg to float when possible."""

    data = _llm_json(system, query)
    allowed_set = set(allowed_fields)
    filtered = {
        k: v
        for k, v in data.items()
        if k in allowed_set and v not in (None, "")
    }

    # Normalise sex values
    if "sex" in filtered:
        mapped = SEX_MAP.get(str(filtered["sex"]).strip().lower())
        if mapped:
            filtered["sex"] = mapped

    return filtered


def extract_profile_updates_with_fallback(
    query: str,
    expected_fields: list[str],
    allowed_fields: list[str],
) -> dict[str, Any]:
    """Extract profile fields with LLM, then rule-based fallback.

    First tries LLM extraction. If LLM returns nothing, falls back to
    rule-based single-field parsing for expected fields.

    Args:
        query: The raw user message.
        expected_fields: Fields we specifically expect in this message.
        allowed_fields: Superset of allowed field names.

    Returns:
        Dict of extracted field:value pairs.
    """
    # Lazy import to avoid circular dependency
    from agent.shared.profile_utils import _parse_single_field

    updates = extract_profile_updates(query, allowed_fields)
    allowed_set = set(allowed_fields)
    normalized_expected = [f for f in expected_fields if f in allowed_set]

    # Fallback: if LLM extracted nothing, try parsing the first expected field
    if normalized_expected and len(updates) == 0:
        first_expected = normalized_expected[0]
        parsed = _parse_single_field(first_expected, query)
        if parsed is not None:
            updates[first_expected] = parsed

    # Additional direct-parse attempt for well-defined fields
    direct_parse_fields = {
        "name", "age", "sex", "height_cm", "weight_kg", "goal",
        "activity_level", "fitness_level", "workout_days",
        "experience_level", "training_days_per_week", "session_duration",
        "daily_steps", "stress_level", "job_type",
    }
    for field in normalized_expected:
        if field in updates:
            continue
        if field not in direct_parse_fields:
            continue
        parsed = _parse_single_field(field, query)
        if parsed is not None:
            updates[field] = parsed

    return updates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Follow-up Questions (general_diet_query / general_workout_query)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def answer_followup_question(
    domain: str,
    query: str,
    profile: dict[str, Any],
    plan_text: str,
    system_prompt: str,
) -> str:
    """Answer a follow-up question about the plan without regenerating it.

    Used by the ``general_diet_query`` and ``general_workout_query`` intents.

    Args:
        domain: "diet" or "workout".
        query: The user's question.
        profile: Current user profile dict.
        plan_text: The current plan (JSON string or markdown).
        system_prompt: The domain system prompt (e.g. DIET_PROMPTS["few_shot"]).

    Returns:
        The LLM's answer as a string.
    """
    llm = ChatOpenAI(model=_PLAN_MODEL, temperature=0.4)
    prompt = (
        f"The user has an active {domain} plan. Here is their profile:\n"
        f"{json.dumps(profile, indent=2)}\n\n"
        f"Here is their current plan:\n{plan_text}\n\n"
        f"The user is asking a follow-up question about this plan:\n"
        f'"{query}"\n\n'
        "Answer their question directly and concisely. Stay within the "
        "context of their plan and profile. Do NOT regenerate the full plan. "
        "Just answer the specific question."
    )
    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ])
    return resp.content


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Plan Q&A — answer specific questions from an existing plan
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_PLAN_QA_SYSTEM = """\
You are a helpful fitness and nutrition assistant. The user has an \
existing {domain} plan (shown below). Respond to their request using ONLY \
information from this plan.

Today is {today} ({today_weekday}). Tomorrow is {tomorrow} ({tomorrow_weekday}).

Rules:
- If the user asks about a SPECIFIC day, meal, exercise, or topic — extract \
  ONLY that portion and present it clearly. Do NOT reproduce the entire plan.
- If the user asks for a specific day like "tomorrow" or "Monday", map it to \
  the correct day of the week using the dates above, then return ONLY that day.
- Use markdown tables when the answer involves meals, exercises, or schedules.
- If the user is asking to see their full/entire/complete plan (e.g. "show my \
  plan", "get my diet plan", "display my workout"), return the FULL plan as-is.
- Be concise and directly helpful.
- If the plan doesn't contain the answer, say so honestly.
"""


def answer_plan_question(
    domain: str,
    plan_text: str,
    question: str,
) -> str:
    """Use FAST_MODEL to intelligently answer any query about a stored plan.

    The LLM decides whether to return the full plan or just the relevant
    section based on the user's question — no keyword matching needed.
    """
    from datetime import datetime, timedelta

    today = datetime.now()
    tomorrow = today + timedelta(days=1)

    system = _PLAN_QA_SYSTEM.format(
        domain=domain,
        today=today.strftime("%A, %B %d"),
        today_weekday=today.strftime("%A"),
        tomorrow=tomorrow.strftime("%A, %B %d"),
        tomorrow_weekday=tomorrow.strftime("%A"),
    )

    user_msg = (
        f"## User's {domain} plan:\n\n"
        f"{plan_text}\n\n"
        f"---\n\n"
        f"## User's question:\n{question}"
    )

    llm = ChatOpenAI(model=_FAST_MODEL, temperature=0.3, max_tokens=4096)
    resp = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ])
    return resp.content


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Plan Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_plan(
    domain: str,
    profile: dict[str, Any],
    query: str,
    system_prompt: str,
    *,
    existing_plan: str | None = None,
) -> str:
    """Generate or update a plan as raw text (markdown).

    Args:
        domain: "diet" or "workout".
        profile: Complete user profile dict.
        query: The user's original request or modification request.
        system_prompt: Domain system prompt (any technique: few_shot, cot, etc.).
        existing_plan: If provided, LLM makes incremental changes to this plan.

    Returns:
        The generated plan as a string (markdown format).
    """
    llm = ChatOpenAI(model=_PLAN_MODEL, temperature=0.5)

    if existing_plan:
        prompt = (
            f"The user has an existing {domain} plan. Here is their profile:\n"
            f"{json.dumps(profile, indent=2)}\n\n"
            f"Here is their CURRENT plan:\n"
            f"{existing_plan}\n\n"
            f"The user wants the following changes:\n"
            f'"{query}"\n\n'
            "IMPORTANT INSTRUCTIONS:\n"
            "1. Apply ONLY the requested changes to the existing plan.\n"
            "2. Keep everything else from the current plan EXACTLY as-is.\n"
            "3. You MUST follow the Output Contract from your system prompt — "
            "output ALL required sections in the correct format.\n"
            "4. Output the COMPLETE updated plan (all sections), not just "
            "the changed parts.\n"
        )
    else:
        prompt = (
            f"Create a personalized {domain} plan using this profile:\n"
            f"{json.dumps(profile, indent=2)}\n\n"
            f'User\'s original request: "{query}"\n\n'
            "IMPORTANT: Follow the Output Contract from your system prompt "
            "EXACTLY. Include ALL required sections in order."
        )

    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ])
    return resp.content


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JSON Plan Generation (structured output)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_JSON_OUTPUT_INSTRUCTION = """\

CRITICAL OUTPUT FORMAT REQUIREMENT:
Return your COMPLETE response as a single JSON object. Each section from your
Output Contract should be a key in the JSON. Structure the data naturally —
use nested objects, arrays, strings, and numbers as appropriate for each section.

The JSON must be valid and parseable. Return ONLY the JSON object.
Do NOT wrap it in markdown code fences. Do NOT add any text before or after.

Example top-level structure (adapt sections to your Output Contract):
{
  "calorie_calculation": { ... },
  "macro_targets": { ... },
  "meal_plan": [ ... ],
  "snack_swaps": [ ... ],
  "personal_rules": [ ... ],
  "timeline": "...",
  "hydration": { ... },
  "supplements": [ ... ],
  "disclaimer": "..."
}
"""


def generate_plan_as_json(
    domain: str,
    profile: dict[str, Any],
    query: str,
    system_prompt: str,
    *,
    existing_plan: str | None = None,
    retries: int = 2,
) -> dict[str, Any]:
    """Generate a plan and return it as structured JSON.

    Uses the given system_prompt (any technique: few_shot, cot, analogical, etc.)
    untouched. Appends a JSON output instruction to the user message only.

    The JSON structure is flexible and driven by the prompt — not hardcoded.
    If the user wants fasting on Thursday, the LLM omits meals for that day.

    Args:
        domain: "diet" or "workout".
        profile: Complete user profile dict.
        query: User's request or modification request.
        system_prompt: Domain system prompt (e.g. DIET_PROMPTS["few_shot"]).
        existing_plan: If provided, passed to LLM for incremental update.
        retries: Number of retries on JSON parse failure.

    Returns:
        Parsed JSON dict representing the plan.

    Raises:
        ValueError: If all attempts fail to produce valid JSON.
    """
    llm = ChatOpenAI(model=_PLAN_MODEL, temperature=0.5)

    if existing_plan:
        # Serialise existing plan for the LLM
        existing_str = (
            json.dumps(existing_plan, indent=2)
            if isinstance(existing_plan, dict)
            else str(existing_plan)
        )
        base_prompt = (
            f"The user has an existing {domain} plan. Here is their profile:\n"
            f"{json.dumps(profile, indent=2)}\n\n"
            f"Here is their CURRENT plan:\n"
            f"{existing_str}\n\n"
            f"The user wants the following changes:\n"
            f'"{query}"\n\n'
            "IMPORTANT INSTRUCTIONS:\n"
            "1. Apply ONLY the requested changes to the existing plan.\n"
            "2. Keep everything else from the current plan EXACTLY as-is.\n"
            "3. Output the COMPLETE updated plan (all sections).\n"
        )
    else:
        base_prompt = (
            f"Create a personalized {domain} plan using this profile:\n"
            f"{json.dumps(profile, indent=2)}\n\n"
            f'User\'s original request: "{query}"\n\n'
            "Follow the Output Contract from your system prompt EXACTLY. "
            "Include ALL required sections in order.\n"
        )

    # Append JSON output instruction
    full_prompt = base_prompt + _JSON_OUTPUT_INSTRUCTION

    for attempt in range(retries + 1):
        try:
            resp = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=full_prompt),
            ])
            raw = resp.content.strip()
            parsed = _extract_json_from_text(raw)
            if parsed is not None:
                logger.info(
                    "[generate_plan_as_json] Success on attempt %d, keys=%s",
                    attempt,
                    list(parsed.keys()),
                )
                return parsed

            logger.warning(
                "[generate_plan_as_json] Attempt %d: could not parse JSON "
                "(response length=%d)",
                attempt,
                len(raw),
            )
        except Exception as exc:
            logger.warning(
                "[generate_plan_as_json] Attempt %d error: %s", attempt, exc
            )

    raise ValueError(
        f"Failed to generate valid JSON plan after {retries + 1} attempts"
    )


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Try multiple strategies to extract a JSON object from LLM output.

    Strategies (in order):
      1. Direct JSON parse
      2. Strip markdown code fences and parse
      3. Regex extract first {...} block
    """
    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip ```json ... ``` fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: regex first {...} block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Plan JSON Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_plan_json(
    plan: dict[str, Any],
    profile: dict[str, Any],
) -> list[str]:
    """Validate a JSON plan against profile constraints.

    Performs flexible validation — checks what's present, doesn't enforce
    a rigid schema. The structure is driven by the prompt.

    Args:
        plan: The parsed JSON plan dict.
        profile: The user profile used for generation.

    Returns:
        List of warning/error strings. Empty list means valid.
    """
    issues: list[str] = []

    # Check: plan is not empty
    if not plan:
        issues.append("Plan is empty")
        return issues

    # Check: calorie floor
    sex = str(profile.get("sex", "")).lower()
    calorie_floor = 1200.0 if sex == "female" else 1500.0

    # Look for calorie target in various possible keys
    calorie_keys = ["calorie_target", "calorie_calculation", "calories", "daily_calories"]
    calorie_value = None
    for key in calorie_keys:
        val = plan.get(key)
        if isinstance(val, (int, float)):
            calorie_value = float(val)
            break
        if isinstance(val, dict):
            # Nested: look for target/total/daily
            for sub_key in ["target", "daily_target", "total", "tdee", "calorie_target"]:
                sub_val = val.get(sub_key)
                if isinstance(sub_val, (int, float)):
                    calorie_value = float(sub_val)
                    break
            if calorie_value is not None:
                break

    if calorie_value is not None and calorie_value < calorie_floor:
        issues.append(
            f"Calorie target ({calorie_value:.0f}) is below the safety floor "
            f"({calorie_floor:.0f} for {sex or 'unknown sex'})"
        )

    # Check: no allergens in meal names (best-effort string match)
    allergies_raw = str(profile.get("allergies", "")).lower().strip()
    foods_to_avoid_raw = str(profile.get("foods_to_avoid", "")).lower().strip()

    if allergies_raw and allergies_raw != "none":
        allergens = [a.strip() for a in re.split(r"[,;/]", allergies_raw) if a.strip()]
        plan_text_lower = json.dumps(plan).lower()
        for allergen in allergens:
            if allergen in plan_text_lower:
                issues.append(f"Allergen '{allergen}' found in plan")

    if foods_to_avoid_raw and foods_to_avoid_raw != "none":
        avoid_items = [a.strip() for a in re.split(r"[,;/]", foods_to_avoid_raw) if a.strip()]
        plan_text_lower = json.dumps(plan).lower()
        for item in avoid_items:
            if item in plan_text_lower:
                issues.append(f"Avoided food '{item}' found in plan")

    # Check: calorie/macro values are positive (where present)
    for key in ["protein_g", "carbs_g", "fat_g", "calorie_target"]:
        val = plan.get(key)
        if isinstance(val, (int, float)) and val < 0:
            issues.append(f"Negative value for {key}: {val}")
    macro_targets = plan.get("macro_targets")
    if isinstance(macro_targets, dict):
        for key, val in macro_targets.items():
            if isinstance(val, (int, float)) and val < 0:
                issues.append(f"Negative macro value: {key}={val}")

    return issues


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JSON Plan → Markdown Converter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plan_json_to_markdown(plan: dict[str, Any]) -> str:
    """Convert a JSON plan dict to readable markdown.

    Handles whatever structure the LLM returned — iterates keys and
    renders tables, lists, and text as appropriate. No hardcoded schema.

    Args:
        plan: The parsed JSON plan dict.

    Returns:
        Markdown string for display in Streamlit.
    """
    if not plan:
        return "_No plan data available._"

    sections: list[str] = []

    for key, value in plan.items():
        # Section header from key name
        title = key.replace("_", " ").title()
        sections.append(f"## {title}\n")

        if isinstance(value, str):
            sections.append(value + "\n")

        elif isinstance(value, (int, float)):
            sections.append(f"**{title}:** {value}\n")

        elif isinstance(value, list):
            sections.append(_render_list(value))

        elif isinstance(value, dict):
            sections.append(_render_dict(value))

        else:
            sections.append(str(value) + "\n")

    return "\n".join(sections)


def _render_list(items: list) -> str:
    """Render a list of items as markdown."""
    if not items:
        return "_None_\n"

    # If items are simple strings/numbers
    if all(isinstance(item, (str, int, float)) for item in items):
        lines = [f"- {item}" for item in items]
        return "\n".join(lines) + "\n"

    # If items are dicts (e.g., meals, supplements)
    if all(isinstance(item, dict) for item in items):
        return _render_table(items)

    # Mixed: stringify each
    lines = [f"- {item}" for item in items]
    return "\n".join(lines) + "\n"


def _render_dict(data: dict) -> str:
    """Render a dict as a key-value list or nested sections."""
    lines: list[str] = []
    for key, value in data.items():
        label = key.replace("_", " ").title()
        if isinstance(value, (str, int, float)):
            lines.append(f"- **{label}:** {value}")
        elif isinstance(value, list):
            lines.append(f"\n### {label}\n")
            lines.append(_render_list(value))
        elif isinstance(value, dict):
            lines.append(f"\n### {label}\n")
            lines.append(_render_dict(value))
        else:
            lines.append(f"- **{label}:** {value}")
    return "\n".join(lines) + "\n"


def _render_table(rows: list[dict]) -> str:
    """Render a list of dicts as a markdown table."""
    if not rows:
        return "_None_\n"

    # Collect all keys across all rows
    all_keys: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in all_keys:
                all_keys.append(k)

    # Header
    headers = [k.replace("_", " ").title() for k in all_keys]
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"

    # Data rows
    data_rows: list[str] = []
    for row in rows:
        cells: list[str] = []
        for k in all_keys:
            val = row.get(k, "")
            if isinstance(val, (list, dict)):
                val = json.dumps(val, ensure_ascii=False)
            cells.append(str(val))
        data_rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header_row, separator, *data_rows]) + "\n"
