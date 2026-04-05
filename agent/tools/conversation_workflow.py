"""
agent/tools/conversation_workflow.py
────────────────────────────────────
Shared multi-turn CRUD + intake workflow logic for diet/workout tools.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state_manager import StateManager
from agent.persistence import (
    delete_record,
    get_record,
    update_calendar_sync,
    upsert_record,
)

logger = logging.getLogger("fitgen.workflow")
from agent.config import DEFAULT_MODEL

_LLM_MODEL = DEFAULT_MODEL

BASE_PROFILE_FIELDS = [
    "name",
    "age",
    "sex",
    "height_cm",
    "weight_kg",
    "goal",
    "activity_level",
]

DOMAIN_REQUIRED_FIELDS = {
    "diet": BASE_PROFILE_FIELDS + ["diet_preference", "foods_to_avoid", "allergies"],
    "workout": BASE_PROFILE_FIELDS + ["fitness_level", "equipment", "workout_days"],
}

FIELD_QUESTION = {
    "name": "What name should I use for your plan?",
    "age": "What is your age?",
    "sex": "What is your sex (male/female/other)?",
    "height_cm": "What is your height in cm?",
    "weight_kg": "What is your current weight in kg?",
    "goal": "What is your primary goal (fat loss, muscle gain, maintenance, performance)?",
    "activity_level": "What is your activity level (sedentary, light, moderate, high, athlete)?",
    "diet_preference": "What diet preference do you follow (omnivore, vegetarian, vegan, eggetarian, etc.)?",
    "foods_to_avoid": "Any foods you want to avoid?",
    "allergies": "Any allergies or intolerances?",
    "fitness_level": "What is your current fitness level (beginner, intermediate, advanced)?",
    "equipment": "What equipment do you have access to?",
    "workout_days": "How many workout days per week can you commit to?",
}

SEX_MAP = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female",
    "other": "other",
    "non-binary": "other",
    "nonbinary": "other",
}

GOAL_KEYWORDS = {
    "fat loss": "fat loss",
    "lose fat": "fat loss",
    "weight loss": "fat loss",
    "muscle gain": "muscle gain",
    "gain muscle": "muscle gain",
    "bulk": "muscle gain",
    "maintenance": "maintenance",
    "maintain": "maintenance",
    "performance": "performance",
}

ACTIVITY_KEYWORDS = {
    "sedentary": "sedentary",
    "light": "light",
    "moderate": "moderate",
    "high": "high",
    "athlete": "athlete",
}


def _llm_json(system: str, user: str, retries: int = 2) -> dict[str, Any]:
    """Lightweight LLM JSON call with retry."""
    llm = ChatOpenAI(model=_LLM_MODEL, temperature=0)
    for attempt in range(retries + 1):
        try:
            resp = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            text = resp.content.strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", text, flags=re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except json.JSONDecodeError:
                        pass
            logger.warning("_llm_json attempt %d: failed to parse JSON from response", attempt)
        except Exception as e:
            logger.warning("_llm_json attempt %d error: %s", attempt, e)
    logger.error("_llm_json: all %d attempts failed", retries + 1)
    return {}


def classify_user_intent(
    query: str,
    *,
    domain: str,
    stage: str | None,
    active_intent: str | None,
    user_profile: dict[str, Any],
    pending_question: str | None = None,
) -> dict[str, Any]:
    """Centralised LLM-based intent classifier.

    Replaces the previous fragmented approach (detect_intent, parse_yes_no,
    fallback_yes_no_from_text, looks_like_modify_request,
    is_generic_modify_request, resolve_intent) with a single context-aware
    LLM call.

    Parameters
    ----------
    query : str
        The raw user message for this turn.
    domain : str
        Current domain ("diet" or "workout").
    stage : str | None
        Current workflow stage (e.g., "confirm_profile", "calendar_sync",
        "plan_feedback", "confirm_delete", "collect_profile", etc.).
    active_intent : str | None
        The intent already set in the workflow (e.g., "create", "update").
    user_profile : dict
        The user profile collected so far.
    pending_question : str | None
        The last question the system asked the user (gives context for
        interpreting short replies like "yes", "83", "male").

    Returns
    -------
    dict with keys:
        intent : str
            One of: create, update, delete, get, confirm, reject, other.
        is_confirmation : bool
            True when the user is affirming / agreeing.
        has_modification_details : bool
            True when the user supplies concrete changes (not just "modify it").
        is_generic_request : bool
            True for vague commands like "modify it" with no specifics.
        reason : str
            Short explanation of the classification.
    """
    profile_summary = json.dumps(user_profile, indent=2) if user_profile else "(empty)"

    system = f"""\
You are an intent classifier for FITGEN.AI, a fitness coaching chatbot.
Your ONLY job is to decide what the user wants to do based on their message,
the current conversation context, and the last question the system asked.

Return ONLY valid JSON with these keys:
  intent: exactly one of "create", "update", "delete", "get", "confirm", "reject", "other"
  is_confirmation: true if the user is saying yes / agreeing / affirming ("yes", "sure", "looks good", "perfect", "correct")
  has_modification_details: true if the user provides specific changes (e.g., "change protein to 150g", "remove dairy", "goal muscle gain")
  is_generic_request: true if the user asks for a change but gives NO specifics (e.g., "modify it", "update", "change it")
  reason: one-sentence explanation of your decision

Classification rules:
1. If the user says yes / agrees / affirms / gives positive feedback ("looks good",
   "perfect", "awesome", "works for me", "correct", "confirmed") → intent="confirm", is_confirmation=true.
2. If the user says no / declines / disagrees ("no", "nope", "nah", "not now",
   "cancel", "incorrect") → intent="reject".
3. If the user wants a new plan created ("create a plan", "make me a plan",
   "build a workout") → intent="create".
4. If the user wants to modify / change / update something with specifics
   ("change my goal to muscle gain", "remove dairy", "add more protein") →
   intent="update", has_modification_details=true.
5. If the user wants to modify but gives NO specifics ("modify it", "update",
   "change it", "please modify") → intent="update", is_generic_request=true,
   has_modification_details=false.
6. If the user wants to delete their plan → intent="delete".
7. If the user wants to see / retrieve their plan → intent="get".
8. If the user's message is a greeting, off-topic, or you can't determine
   the intent → intent="other".

Context-aware guidance:
- Current domain: {domain}
- Current workflow stage: {stage or '(none — fresh conversation)'}
- Active intent in workflow: {active_intent or '(none)'}
- Pending question asked by system: {pending_question or '(none)'}
- User profile so far: {profile_summary}

Use the pending question to understand what the user is responding to.
For example, if the pending question is "Do you want Google Calendar sync?"
and the user says "nah", that is intent="reject".
If the pending question asks for profile details and the user provides them,
that is NOT a confirmation — let the caller handle profile extraction.
In that case, if the user is providing data mid-create, use intent="create".
"""

    data = _llm_json(system, query)

    # Normalise and validate
    intent = str(data.get("intent", "other")).strip().lower()
    if intent == "modify":
        intent = "update"
    if intent == "yes":
        intent = "confirm"
    if intent == "no":
        intent = "reject"
    valid_intents = {"create", "update", "delete", "get", "confirm", "reject", "other"}
    if intent not in valid_intents:
        intent = "other"

    result = {
        "intent": intent,
        "is_confirmation": bool(data.get("is_confirmation", intent == "confirm")),
        "has_modification_details": bool(data.get("has_modification_details", False)),
        "is_generic_request": bool(data.get("is_generic_request", False)),
        "reason": str(data.get("reason", "")).strip(),
    }

    logger.info(
        "[classify_user_intent] query=%r → intent=%s confirm=%s mod_details=%s generic=%s reason=%s",
        query[:80],
        result["intent"],
        result["is_confirmation"],
        result["has_modification_details"],
        result["is_generic_request"],
        result["reason"][:80] if result["reason"] else "(none)",
    )
    return result


def extract_profile_updates(query: str, allowed_fields: list[str]) -> dict[str, Any]:
    system = """
Extract any user profile values from the message.
Return JSON only with keys among:
name, age, sex, height_cm, weight_kg, goal, activity_level,
diet_preference, foods_to_avoid, allergies, fitness_level, equipment, workout_days.
If absent, omit the key.
Normalize age to int, height_cm and weight_kg to float when possible.
""".strip()
    data = _llm_json(system, query)
    # Accept only known profile keys and non-empty values.
    # Why: prevents accidental schema pollution from model hallucinated keys.
    allowed_set = set(allowed_fields)
    allowed = {k: v for k, v in data.items() if k in allowed_set and v not in (None, "")}
    if "sex" in allowed:
        # Normalize shorthand/user variants to stable enum-like values.
        # Why: downstream logic and prompting remain consistent.
        mapped = SEX_MAP.get(str(allowed["sex"]).strip().lower())
        if mapped:
            allowed["sex"] = mapped
    return allowed


def _extract_number(query: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", query)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_single_field(field: str, query: str) -> Any | None:
    text = query.strip()
    lower = text.lower().strip()

    if field == "sex":
        return SEX_MAP.get(lower)

    if field == "age":
        value = _extract_number(text)
        if value is None:
            return None
        age = int(round(value))
        return age if 10 <= age <= 100 else None

    if field in {"height_cm", "weight_kg"}:
        value = _extract_number(text)
        if value is None:
            return None
        return round(float(value), 1)

    if field == "goal":
        for key, norm in GOAL_KEYWORDS.items():
            if key in lower:
                return norm
        return None

    if field == "activity_level":
        for key, norm in ACTIVITY_KEYWORDS.items():
            if key in lower:
                return norm
        return None

    if field == "fitness_level":
        levels = {"beginner", "intermediate", "advanced"}
        for level in levels:
            if level in lower:
                return level
        return None

    if field == "workout_days":
        value = _extract_number(text)
        if value is None:
            return None
        days = int(round(value))
        return days if 1 <= days <= 7 else None

    if field in {"diet_preference", "foods_to_avoid", "allergies", "equipment"}:
        cleaned = re.sub(r"\s+", " ", text).strip(" .,!?")
        lower_cleaned = cleaned.lower()
        profile_markers = [
            "name",
            "age",
            "male",
            "female",
            "height",
            "weight",
            "goal",
            "activity",
            "cm",
            "kg",
        ]
        if len(cleaned.split()) > 8 and any(marker in lower_cleaned for marker in profile_markers):
            return None
        return cleaned or None

    if field == "name":
        if re.search(r"\d", text):
            return None
        cleaned = re.sub(r"\s+", " ", text).strip(" .,!?")
        if not cleaned:
            return None
        words = cleaned.split()
        if len(words) > 4:
            return None
        return cleaned

    return None


def extract_profile_updates_with_fallback(query: str, expected_fields: list[str], allowed_fields: list[str]) -> dict[str, Any]:
    # Primary extraction via LLM for flexible natural language understanding.
    updates = extract_profile_updates(query, allowed_fields)

    allowed_set = set(allowed_fields)
    normalized_expected = [field for field in expected_fields if field in allowed_set]

    # If user gives a short reply (e.g., "83" or "M"), map it to the next expected field.
    if normalized_expected and len(updates) == 0:
        first_expected = normalized_expected[0]
        parsed = _parse_single_field(first_expected, query)
        if parsed is not None:
            updates[first_expected] = parsed

    # Also try filling any remaining expected fields directly.
    # Why: improves robustness for short replies and partially structured inputs.
    direct_parse_fields = {
        "name",
        "age",
        "sex",
        "height_cm",
        "weight_kg",
        "goal",
        "activity_level",
        "fitness_level",
        "workout_days",
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


def required_fields_for_domain(domain: str) -> list[str]:
    return list(DOMAIN_REQUIRED_FIELDS.get(domain, BASE_PROFILE_FIELDS))


def missing_profile_fields(profile: dict[str, Any], required_fields: list[str]) -> list[str]:
    # Return all missing fields in canonical order.
    # Why: deterministic order makes prompts predictable and easier to test.
    return [field for field in required_fields if field not in profile or profile[field] in (None, "")]


def append_completed_step(
    current_workflow: dict[str, Any],
    overrides: dict[str, Any],
    step_name: str,
) -> dict[str, Any]:
    """Merge overrides into current workflow and append step to completed_steps."""
    next_workflow = dict(current_workflow)
    next_workflow.update(overrides)
    completed = list(next_workflow.get("completed_steps") or [])
    if step_name not in completed:
        completed.append(step_name)
    next_workflow["completed_steps"] = completed
    return next_workflow


def build_profile_confirmation(profile: dict[str, Any], required_fields: list[str]) -> str:
    lines: list[str] = []
    for field in required_fields:
        if field in profile and profile[field] not in (None, ""):
            label = field.replace("_", " ").title()
            lines.append(f"- {label}: {profile[field]}")

    joined = "\n".join(lines) if lines else "- (No fields mapped yet)"
    return (
        "I mapped these details:\n"
        f"{joined}\n\n"
        "Reply yes to confirm, or share corrections."
    )


def build_profile_bulk_question(fields: list[str]) -> str:
    # Ask for all pending fields together.
    # Why: reduces turn count and prevents repetitive one-by-one questioning.
    lines = [FIELD_QUESTION[field] for field in fields if field in FIELD_QUESTION]
    checklist = "\n".join(f"- {line}" for line in lines)
    return (
        "Please share the following details in one message so I can build your plan:\n"
        f"{checklist}\n\n"
        "Example: Name Kushagra, age 28, male, height 183 cm, weight 83 kg, goal fat loss, activity moderate."
    )


def generate_plan(
    domain: str,
    profile: dict[str, Any],
    query: str,
    system_prompt: str,
    *,
    existing_plan: str = "",
) -> str:
    """Generate or update a plan. When *existing_plan* is provided the LLM is
    instructed to modify that plan according to the user's latest request while
    preserving everything else (calories, macros, structure)."""
    llm = ChatOpenAI(model=_LLM_MODEL, temperature=0.5)

    if existing_plan:
        prompt = (
            f"The user already has this {domain} plan:\n"
            f"--- EXISTING PLAN START ---\n{existing_plan}\n--- EXISTING PLAN END ---\n\n"
            f"User profile:\n{json.dumps(profile, indent=2)}\n\n"
            f"The user wants the following change: {query}\n\n"
            "RULES:\n"
            "- Apply ONLY the requested change. Do NOT recalculate calories, macros, "
            "or restructure the plan unless the user explicitly asks for that.\n"
            "- Keep the same calorie target, macro split, and meal structure from the "
            "existing plan.\n"
            "- Output the COMPLETE updated plan once (do NOT output it twice).\n"
        )
    else:
        prompt = (
            f"Create a personalized {domain} plan using this profile:\n"
            f"{json.dumps(profile, indent=2)}\n\n"
            f"Latest user instruction: {query}\n"
            "Provide practical, safe, actionable steps with concise structure.\n"
            "Output the plan exactly ONCE — do NOT repeat or duplicate any tables."
        )

    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ])
    return resp.content


def build_response(
    *,
    assistant_message: str,
    state_id: str,
    user_email: str,
    workflow: dict[str, Any],
    user_profile: dict[str, Any],
    state_manager: StateManager,
    extra: dict[str, Any] | None = None,
) -> str:
    # Tool response contract consumed by state_sync + UI layers.
    # Why: explicitly separates user-facing text from machine state updates.
    payload: dict[str, Any] = {
        "assistant_message": assistant_message,
        "state_updates": {
            "context_id": state_id,
            "state_id": state_id,
            "user_email": user_email,
            "workflow": workflow,
            "user_profile": user_profile,
        },
    }
    if extra:
        payload["extra"] = extra

    state_manager.persist(
        user_profile=user_profile,
        workflow=workflow,
        user_email=user_email,
        calendar_sync_requested=(
            bool(extra.get("calendar_sync_requested"))
            if extra and "calendar_sync_requested" in extra
            else state_manager.calendar_sync_requested
        ),
    )
    return json.dumps(payload, ensure_ascii=False)


def handle_multi_turn(
    *,
    domain: str,
    query: str,
    state: dict[str, Any],
    plan_system_prompt: str,
) -> str:
    # 1) Resolve and hydrate context through StateManager.
    state_manager = StateManager.from_state(state)
    state_id = state_manager.context_id
    user_email = state_manager.user_email
    workflow = dict(state_manager.workflow or {})
    profile = dict(state_manager.user_profile or {})

    required_fields = required_fields_for_domain(domain)
    active_intent = workflow.get("intent")
    stage = workflow.get("stage")

    # Determine what question the system last asked (if any), for LLM context.
    pending_question = workflow.get("pending_question")

    # 2) Single centralised intent classification — replaces detect_intent,
    #    parse_yes_no, fallback_yes_no, looks_like_modify, is_generic_modify,
    #    and resolve_intent with one context-aware LLM call.
    classification = classify_user_intent(
        query,
        domain=domain,
        stage=stage,
        active_intent=active_intent,
        user_profile=profile,
        pending_question=pending_question,
    )

    classified_intent = classification["intent"]
    is_confirmation = classification["is_confirmation"]
    has_modification_details = classification["has_modification_details"]
    is_generic = classification["is_generic_request"]

    logger.info(
        "[Workflow:%s] classified intent=%s (active=%s, stage=%s)",
        domain,
        classified_intent,
        active_intent,
        stage,
    )

    # 3) Resolve effective intent: CRUD intents always take priority;
    #    confirm/reject are stage-local; otherwise keep active_intent.
    if classified_intent in {"get", "delete", "update", "create"}:
        intent = classified_intent
    elif classified_intent in {"confirm", "reject"}:
        # confirm/reject are handled inline by each stage below;
        # keep active_intent so we stay in the current flow.
        intent = active_intent or "other"
    else:
        intent = active_intent or classified_intent

    if intent != active_intent and active_intent is not None:
        stage = None
        workflow.pop("stage", None)

    if stage == "calendar_sync":
        stage_intent = workflow.get("intent") or active_intent
        # If user switches to a different CRUD intent, interrupt calendar sync.
        if classified_intent in {"get", "delete", "update", "create"} and classified_intent != stage_intent:
            logger.info("[Workflow:%s] calendar_sync interrupted by intent=%s", domain, classified_intent)
            workflow = append_completed_step(
                workflow,
                {"intent": classified_intent, "domain": domain},
                "calendar_interrupted_by_intent",
            )
            stage = None
            workflow.pop("stage", None)
        else:
            if is_confirmation:
                # Generate Google OAuth URL and store plan text for the push stage.
                try:
                    from agent.tools.calendar_integration import get_authorization_url
                    auth_url, oauth_state = get_authorization_url()
                except Exception as e:
                    logger.warning("[Workflow:%s] Calendar OAuth setup failed: %s", domain, e)
                    auth_url = None
                    oauth_state = None

                if auth_url:
                    update_calendar_sync(state_id, True)
                    # Store plan_text and oauth_state so the Streamlit callback can push events.
                    plan_text = workflow.get("plan_text", "")
                    workflow = append_completed_step(
                        workflow,
                        {"stage": "calendar_oauth_pending", "domain": domain,
                         "plan_text": plan_text, "oauth_state": oauth_state},
                        "calendar_oauth_started",
                    )
                    return build_response(
                        assistant_message=(
                            "🔗 **Ready to connect Google Calendar!**\n\n"
                            "Click the **\"📅 Connect Google Calendar\"** button in the sidebar "
                            "to sign in with Google and sync your plan.\n\n"
                            f"Or open this link directly: [Authorize FITGEN.AI]({auth_url})"
                        ),
                        state_id=state_id,
                        user_email=user_email,
                        workflow=workflow,
                        user_profile=profile,
                        state_manager=state_manager,
                        extra={"calendar_sync_requested": True, "calendar_auth_url": auth_url},
                    )
                else:
                    # Fallback if Google credentials not configured.
                    update_calendar_sync(state_id, True)
                    workflow = append_completed_step(
                        workflow,
                        {"stage": "", "pending_question": ""},
                        "calendar_sync_enabled_no_oauth",
                    )
                    return build_response(
                        assistant_message=(
                            "I marked your plan for Google Calendar sync, but the Google Calendar "
                            "integration is not configured yet. Please add GOOGLE_CLIENT_ID and "
                            "GOOGLE_CLIENT_SECRET to your .env file."
                        ),
                        state_id=state_id,
                        user_email=user_email,
                        workflow=workflow,
                        user_profile=profile,
                        state_manager=state_manager,
                        extra={"calendar_sync_requested": True},
                    )

            if classified_intent == "reject":
                update_calendar_sync(state_id, False)
                workflow = append_completed_step(
                    workflow,
                    {"stage": "", "pending_question": ""},
                    "calendar_sync_skipped",
                )
                return build_response(
                    assistant_message="No problem — I won't sync it to Google Calendar.",
                    state_id=state_id,
                    user_email=user_email,
                    workflow=workflow,
                    user_profile=profile,
                    state_manager=state_manager,
                    extra={"calendar_sync_requested": False},
                )

            # Could not determine yes/no — re-ask.
            workflow = append_completed_step(
                workflow,
                {"intent": workflow.get("intent", "create"), "stage": "calendar_sync", "domain": domain,
                 "pending_question": "Do you want Google Calendar sync? (yes/no)"},
                "calendar_prompted",
            )
            return build_response(
                assistant_message="Please reply yes or no: do you want Google Calendar sync?",
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile=profile,
                state_manager=state_manager,
            )

    if stage == "plan_feedback":
        stage_intent = workflow.get("intent") or active_intent
        # If user switches to a different CRUD intent, interrupt plan feedback.
        if classified_intent in {"get", "delete"} and classified_intent != stage_intent:
            logger.info("[Workflow:%s] plan_feedback interrupted by intent=%s", domain, classified_intent)
            workflow = append_completed_step(
                workflow,
                {"intent": classified_intent, "domain": domain},
                "plan_feedback_interrupted_by_intent",
            )
            stage = None
            workflow.pop("stage", None)
        elif is_confirmation:
            workflow = append_completed_step(
                workflow,
                {"intent": workflow.get("intent", "create"), "stage": "calendar_sync", "domain": domain,
                 "pending_question": "Would you like me to sync this to Google Calendar? (yes/no)"},
                "plan_confirmed",
            )
            return build_response(
                assistant_message="Great — glad this works for you. Would you like me to sync this to Google Calendar? (yes/no)",
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile=profile,
                state_manager=state_manager,
            )
        elif classified_intent == "reject" or classified_intent == "update":
            # User either rejected the plan or wants to modify it.
            workflow = append_completed_step(
                workflow,
                {"intent": "update", "stage": "await_update_instructions", "domain": domain},
                "plan_rejected",
            )
            msg = (
                f"No problem — tell me what you'd like to change in your {domain} plan "
                "(e.g., foods to avoid, calories, goal, schedule)."
            )
            return build_response(
                assistant_message=msg,
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile=profile,
                state_manager=state_manager,
            )
        else:
            # Could not determine — re-ask.
            workflow = append_completed_step(
                workflow,
                {"intent": workflow.get("intent", "create"), "stage": "plan_feedback", "domain": domain,
                 "pending_question": "Do you want to keep this plan as-is? Reply yes to keep it, or tell me what you want changed."},
                "plan_feedback_prompted",
            )
            return build_response(
                assistant_message=(
                    "Do you want to keep this plan as-is? Reply yes to keep it, "
                    "or tell me what you want changed."
                ),
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile=profile,
                state_manager=state_manager,
            )
    if intent == "get":
        record = get_record(state_id)
        if not record:
            workflow = append_completed_step(workflow, {}, "get_no_record")
            return build_response(
                assistant_message=f"I couldn't find a saved {domain} plan for this context yet.",
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile=profile,
                state_manager=state_manager,
            )

        stored_profile = record.get("profile") or {}
        summary_lines = [f"- {key.replace('_', ' ').title()}: {value}" for key, value in stored_profile.items()]
        summary = "\n".join(summary_lines) if summary_lines else "- (No profile fields saved)"
        workflow = append_completed_step(workflow, {}, "get_success")
        return build_response(
            assistant_message=(
                f"Here is your current {domain} context:\n{summary}\n\n"
                f"Saved plan:\n\n{record.get('plan_text', '')}"
            ),
            state_id=state_id,
            user_email=user_email,
            workflow=workflow,
            user_profile=stored_profile,
            state_manager=state_manager,
        )

    if intent == "create":
        expected_fields = workflow.get("missing_fields") or []
        pending_field = workflow.get("pending_field")
        if pending_field and pending_field in required_fields and pending_field not in expected_fields:
            expected_fields = [pending_field, *expected_fields]
        if not expected_fields:
            expected_fields = missing_profile_fields(profile, required_fields)

        query_lower = query.lower()
        looks_like_create_command = any(token in query_lower for token in ["create", "make", "build", "plan", "chart"])
        has_profile_signals = bool(re.search(r"\d", query)) or any(
            token in query_lower
            for token in ["male", "female", "other", "goal", "activity", "cm", "kg", "age"]
        )

        # Reset stale profile for a fresh generic create command.
        # Why: avoids leaking old context into a brand-new plan request.
        if stage != "collect_profile" and looks_like_create_command and not has_profile_signals:
            profile = {}
            updates = {}
        else:
            updates = extract_profile_updates_with_fallback(query, expected_fields, required_fields)
        profile.update(updates)

        missing_fields = missing_profile_fields(profile, required_fields)
        if missing_fields:
            workflow = append_completed_step(
                workflow,
                {
                    "intent": "create",
                    "stage": "collect_profile",
                    "domain": domain,
                    "missing_fields": missing_fields,
                },
                "profile_collection_started",
            )
            return build_response(
                assistant_message=build_profile_bulk_question(missing_fields),
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile=profile,
                state_manager=state_manager,
            )

        if stage != "confirm_profile":
            confirm_msg = build_profile_confirmation(profile, required_fields)
            workflow = append_completed_step(
                workflow,
                {
                    "intent": "create",
                    "stage": "confirm_profile",
                    "domain": domain,
                    "required_fields": required_fields,
                    "pending_question": confirm_msg,
                },
                "profile_mapped",
            )
            return build_response(
                assistant_message=confirm_msg,
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile=profile,
                state_manager=state_manager,
            )

        # Stage IS confirm_profile — use the centralised classification.
        if is_confirmation:
            pass  # Fall through to plan generation below.
        elif classified_intent == "reject" or has_modification_details:
            # User rejected or provided corrections.
            corrections = extract_profile_updates_with_fallback(
                query,
                missing_profile_fields(profile, required_fields),
                required_fields,
            )
            profile.update(corrections)
            workflow = append_completed_step(
                workflow,
                {
                    "intent": "create",
                    "stage": "collect_profile",
                    "domain": domain,
                    "missing_fields": missing_profile_fields(profile, required_fields),
                },
                "profile_confirmation_rejected",
            )
            return build_response(
                assistant_message="Got it. Please share corrected profile details and I will remap them.",
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile=profile,
                state_manager=state_manager,
            )
        else:
            # Could not determine — re-ask.
            workflow = append_completed_step(
                workflow,
                {
                    "intent": "create",
                    "stage": "confirm_profile",
                    "domain": domain,
                    "required_fields": required_fields,
                    "pending_question": "Please reply yes to confirm the mapped details, or share corrections.",
                },
                "profile_confirmation_reprompted",
            )
            return build_response(
                assistant_message="Please reply yes to confirm the mapped details, or share corrections.",
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile=profile,
                state_manager=state_manager,
            )

        plan = generate_plan(domain, profile, query, plan_system_prompt)
        upsert_record(state_id=state_id, domain=domain, profile=profile, plan_text=plan, calendar_sync=False)
        feedback_msg = (
            f"Great — your {domain} plan is ready.\n\n{plan}\n\n"
            "Please review it and tell me if you want any changes."
        )
        workflow = append_completed_step(
            workflow,
            {"intent": "create", "stage": "plan_feedback", "domain": domain,
             "plan_text": plan,
             "pending_question": "Please review the plan and tell me if you want any changes."},
            "plan_generated",
        )
        return build_response(
            assistant_message=feedback_msg,
            state_id=state_id,
            user_email=user_email,
            workflow=workflow,
            user_profile=profile,
            state_manager=state_manager,
            extra={"plan_text": plan},
        )

    if intent == "update":
        record = get_record(state_id)
        if not record:
            missing_fields = missing_profile_fields(profile, required_fields)
            workflow = append_completed_step(
                workflow,
                {
                    "intent": "create",
                    "stage": "collect_profile",
                    "domain": domain,
                    "missing_fields": missing_fields,
                },
                "update_without_record_fallback_to_create",
            )
            return build_response(
                assistant_message=(
                    "I couldn't find an existing plan to update. Let's create one first.\n\n"
                    + build_profile_bulk_question(missing_fields)
                ),
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile=profile,
                state_manager=state_manager,
            )

        merged_profile = dict(record.get("profile") or {})
        merged_profile.update(profile)
        updates = extract_profile_updates(query, required_fields)
        merged_profile.update(updates)

        # Generic modify requests without concrete details should trigger clarification.
        # Why: prevents low-quality regenerations from underspecified instructions.
        if not updates and is_generic:
            workflow = append_completed_step(
                workflow,
                {"intent": "update", "stage": "await_update_instructions", "domain": domain},
                "update_needs_details",
            )
            return build_response(
                assistant_message=(
                    f"Sure — tell me what you'd like to update in your current {domain} plan "
                    "(e.g., goal, weight, activity level, schedule)."
                ),
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile=merged_profile,
                state_manager=state_manager,
            )

        existing_plan_text = record.get("plan_text", "") or workflow.get("plan_text", "")
        plan = generate_plan(
            domain, merged_profile, query, plan_system_prompt,
            existing_plan=existing_plan_text,
        )
        upsert_record(state_id=state_id, domain=domain, profile=merged_profile, plan_text=plan, calendar_sync=record.get("calendar_sync", False))
        update_feedback_msg = (
            f"Done — I updated your {domain} plan.\n\n{plan}\n\n"
            "Please review the update and tell me if you want any more changes."
        )
        workflow = append_completed_step(
            workflow,
            {"intent": "update", "stage": "plan_feedback", "domain": domain,
             "plan_text": plan,
             "pending_question": "Please review the update and tell me if you want any more changes."},
            "plan_updated",
        )
        return build_response(
            assistant_message=update_feedback_msg,
            state_id=state_id,
            user_email=user_email,
            workflow=workflow,
            user_profile=merged_profile,
            state_manager=state_manager,
            extra={"plan_text": plan},
        )

    if intent == "delete":
        if stage != "confirm_delete":
            workflow = append_completed_step(
                workflow,
                {"intent": "delete", "stage": "confirm_delete", "domain": domain,
                 "pending_question": "Please confirm: do you want me to delete your stored plan and profile? (yes/no)"},
                "delete_confirmation_requested",
            )
            return build_response(
                assistant_message="Please confirm: do you want me to delete your stored plan and profile? (yes/no)",
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile=profile,
                state_manager=state_manager,
            )

        # Stage IS confirm_delete — use the centralised classification.
        if is_confirmation:
            deleted = delete_record(state_id)
            workflow = append_completed_step(workflow, {}, "delete_completed")
            return build_response(
                assistant_message=(
                    "Your record has been deleted." if deleted else "No saved record was found, so nothing was deleted."
                ),
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile={},
                state_manager=state_manager,
            )

        if classified_intent == "reject":
            workflow = append_completed_step(workflow, {}, "delete_canceled")
            return build_response(
                assistant_message="Deletion canceled. Your record is unchanged.",
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile=profile,
                state_manager=state_manager,
            )

        # Could not determine — re-ask.
        workflow = append_completed_step(
            workflow,
            {"intent": "delete", "stage": "confirm_delete", "domain": domain,
             "pending_question": "I need a clear yes or no to proceed with deletion."},
            "delete_confirmation_reprompted",
        )
        return build_response(
            assistant_message="I need a clear yes or no to proceed with deletion.",
            state_id=state_id,
            user_email=user_email,
            workflow=workflow,
            user_profile=profile,
            state_manager=state_manager,
        )

    workflow = append_completed_step(workflow, {}, "fallback_help")
    return build_response(
        assistant_message=(
            f"I can help you create, update, delete, or get your {domain} plan. "
            "Tell me which one you want to do."
        ),
        state_id=state_id,
        user_email=user_email,
        workflow=workflow,
        user_profile=profile,
        state_manager=state_manager,
    )


def execute(
    *,
    domain: str,
    query: str,
    state: dict[str, Any],
    plan_system_prompt: str,
) -> str:
    """Execution entrypoint for context-aware workflow state machine."""
    return handle_multi_turn(
        domain=domain,
        query=query,
        state=state,
        plan_system_prompt=plan_system_prompt,
    )
