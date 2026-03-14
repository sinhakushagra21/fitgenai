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
_LLM_MODEL = os.getenv("FITGEN_LLM_MODEL", "gpt-4o-mini")

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


def detect_intent(query: str) -> tuple[str, str]:
    """Detect intent and return (intent, reason) from LLM JSON output."""
    system = (
        "Classify user intent into exactly one of: create, update, delete, get, other. "
        "Also provide a short reason. "
        "Return JSON only: {\"intent\":\"...\",\"reason\":\"...\"}."
    )
    data = _llm_json(system, query)
    intent = str(data.get("intent", "other")).strip().lower()
    reason = str(data.get("reason", "")).strip()
    if intent == "modify":
        intent = "update"
    if intent not in {"create", "update", "delete", "get", "other"}:
        return "other", reason
    return intent, reason


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


def parse_yes_no(query: str) -> str:
    # Always use LLM for yes/no/unknown classification.
    # Why: caller requested intent understanding to be model-driven rather than heuristic.
    system = "Return JSON only: {\"answer\":\"yes\"|\"no\"|\"unknown\"}."
    data = _llm_json(system, query)
    ans = str(data.get("answer", "unknown")).lower().strip()
    if ans not in {"yes", "no", "unknown"}:
        return "unknown"
    return ans


def fallback_yes_no_from_text(query: str, *, allow_positive_feedback: bool = False) -> str:
    """Deterministic fallback when LLM yes/no classification returns unknown."""
    normalized = re.sub(r"\s+", " ", query.lower()).strip(" .!?")

    yes_tokens = {"yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm", "confirmed"}
    no_tokens = {"no", "n", "nope", "nah", "cancel", "not now"}

    if normalized in yes_tokens:
        return "yes"
    if normalized in no_tokens:
        return "no"

    if allow_positive_feedback:
        positive_feedback = {
            "perfect",
            "perfect plan",
            "looks good",
            "looks great",
            "great",
            "awesome",
            "all good",
            "sounds good",
            "works for me",
            "done",
        }
        if normalized in positive_feedback:
            return "yes"

    return "unknown"


def looks_like_modify_request(query: str) -> bool:
    # Heuristic intent override for update-like language.
    # Why: users often send direct modification requests while in confirmation stages.
    lower = query.lower()
    modify_markers = [
        "modify",
        "update",
        "change",
        "replace",
        "remove",
        "add",
        "avoid",
        "don't eat",
        "dont eat",
        "allergy",
        "intoler",
        "no seafood",
        "no sea food",
        "vegetarian",
        "vegan",
    ]
    return any(marker in lower for marker in modify_markers)


def is_generic_modify_request(query: str) -> bool:
    # Detect vague modify commands that lack actionable details.
    # Why: lets us ask a clarifying follow-up instead of regenerating a plan blindly.
    cleaned = re.sub(r"\s+", " ", query.lower()).strip(" .!?")
    generic_phrases = {
        "modify",
        "modify it",
        "update",
        "update it",
        "change",
        "change it",
        "please update",
        "please modify",
        "please change",
    }
    return cleaned in generic_phrases


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


def generate_plan(domain: str, profile: dict[str, Any], query: str, system_prompt: str) -> str:
    # Single plan generator used by both create/modify branches.
    # Why: keeps output style and constraints consistent.
    llm = ChatOpenAI(model=_LLM_MODEL, temperature=0.5)
    prompt = (
        f"Create a personalized {domain} plan using this profile:\n"
        f"{json.dumps(profile, indent=2)}\n\n"
        f"Latest user instruction: {query}\n"
        "Provide practical, safe, actionable steps with concise structure."
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


def resolve_intent(
    detected_intent: str,
    active_intent: str | None,
    stage: str | None,
    query: str,
) -> str:
    """Single source of truth for intent resolution across all stages."""
    if detected_intent in {"get", "delete", "update"}:
        return detected_intent
    if stage in {"calendar_sync", "plan_feedback"} and looks_like_modify_request(query):
        return "update"
    return active_intent or detected_intent


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

    # 3) Always classify current-turn intent, then decide if it should override workflow.
    # Why: user may change objective mid-flow (e.g., "get my plan", "delete it").
    detected_intent, intent_reason = detect_intent(query)
    logger.info(
        "[Workflow:%s] detected intent=%s reason=%s (active=%s, stage=%s)",
        domain,
        detected_intent,
        intent_reason or "(none)",
        active_intent,
        stage,
    )

    intent = resolve_intent(detected_intent, active_intent, stage, query)

    if intent != active_intent and active_intent is not None:
        stage = None
        workflow.pop("stage", None)

    if stage == "calendar_sync":
        stage_intent = workflow.get("intent") or active_intent
        if stage_intent is not None and intent != stage_intent:
            logger.info("[Workflow:%s] calendar_sync interrupted by resolved intent=%s", domain, intent)
            workflow = append_completed_step(
                workflow,
                {"intent": intent, "domain": domain},
                "calendar_interrupted_by_intent",
            )
            stage = None
            workflow.pop("stage", None)
        else:
            answer = parse_yes_no(query)
            if answer == "unknown":
                answer = fallback_yes_no_from_text(query, allow_positive_feedback=False)

            if answer == "unknown":
                workflow = append_completed_step(
                    workflow,
                    {"intent": workflow.get("intent", "create"), "stage": "calendar_sync", "domain": domain},
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

            if answer == "yes":
                update_calendar_sync(state_id, True)
                workflow = append_completed_step(workflow, {}, "calendar_sync_enabled")
                return build_response(
                    assistant_message=(
                        "Great — I marked your plan for Google Calendar sync. "
                        "Next step: connect your Google account in the integration settings."
                    ),
                    state_id=state_id,
                    user_email=user_email,
                    workflow=workflow,
                    user_profile=profile,
                    state_manager=state_manager,
                    extra={"calendar_sync_requested": True},
                )

            if answer == "no":
                update_calendar_sync(state_id, False)
                workflow = append_completed_step(workflow, {}, "calendar_sync_skipped")
                return build_response(
                    assistant_message="No problem — I won’t sync it to Google Calendar.",
                    state_id=state_id,
                    user_email=user_email,
                    workflow=workflow,
                    user_profile=profile,
                    state_manager=state_manager,
                    extra={"calendar_sync_requested": False},
                )

    if stage == "plan_feedback":
        stage_intent = workflow.get("intent") or active_intent
        if stage_intent is not None and intent != stage_intent:
            logger.info("[Workflow:%s] plan_feedback interrupted by resolved intent=%s", domain, intent)
            workflow = append_completed_step(
                workflow,
                {"intent": intent, "domain": domain},
                "plan_feedback_interrupted_by_intent",
            )
            stage = None
            workflow.pop("stage", None)
        else:
            answer = parse_yes_no(query)
            if answer == "unknown":
                answer = fallback_yes_no_from_text(query, allow_positive_feedback=True)
            if answer == "yes":
                workflow = append_completed_step(
                    workflow,
                    {"intent": workflow.get("intent", "create"), "stage": "calendar_sync", "domain": domain},
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

            if answer == "no":
                workflow = append_completed_step(
                    workflow,
                    {"intent": "update", "stage": "await_update_instructions", "domain": domain},
                    "plan_rejected",
                )
                return build_response(
                    assistant_message=(
                        f"No problem — tell me what you'd like to change in your {domain} plan "
                        "(e.g., foods to avoid, calories, goal, schedule)."
                    ),
                    state_id=state_id,
                    user_email=user_email,
                    workflow=workflow,
                    user_profile=profile,
                    state_manager=state_manager,
                )

            workflow = append_completed_step(
                workflow,
                {"intent": workflow.get("intent", "create"), "stage": "plan_feedback", "domain": domain},
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
            workflow = append_completed_step(
                workflow,
                {
                    "intent": "create",
                    "stage": "confirm_profile",
                    "domain": domain,
                    "required_fields": required_fields,
                },
                "profile_mapped",
            )
            return build_response(
                assistant_message=build_profile_confirmation(profile, required_fields),
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile=profile,
                state_manager=state_manager,
            )

        answer = parse_yes_no(query)
        if answer == "unknown":
            normalized = re.sub(r"\s+", " ", query.lower()).strip(" .!?")
            if normalized in {"yes", "y", "yeah", "yep", "correct", "confirm"}:
                answer = "yes"
            elif normalized in {"no", "n", "nope", "nah", "incorrect"}:
                answer = "no"

        if answer == "unknown":
            workflow = append_completed_step(
                workflow,
                {
                    "intent": "create",
                    "stage": "confirm_profile",
                    "domain": domain,
                    "required_fields": required_fields,
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

        if answer != "yes":
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

        plan = generate_plan(domain, profile, query, plan_system_prompt)
        upsert_record(state_id=state_id, domain=domain, profile=profile, plan_text=plan, calendar_sync=False)
        workflow = append_completed_step(
            workflow,
            {"intent": "create", "stage": "plan_feedback", "domain": domain},
            "plan_generated",
        )
        return build_response(
            assistant_message=(
                f"Great — your {domain} plan is ready.\n\n{plan}\n\n"
                "Please review it and tell me if you want any changes."
            ),
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
        if not updates and is_generic_modify_request(query):
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

        plan = generate_plan(domain, merged_profile, query, plan_system_prompt)
        upsert_record(state_id=state_id, domain=domain, profile=merged_profile, plan_text=plan, calendar_sync=record.get("calendar_sync", False))
        workflow = append_completed_step(
            workflow,
            {"intent": "update", "stage": "plan_feedback", "domain": domain},
            "plan_updated",
        )
        return build_response(
            assistant_message=(
                f"Done — I updated your {domain} plan.\n\n{plan}\n\n"
                "Please review the update and tell me if you want any more changes."
            ),
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
                {"intent": "delete", "stage": "confirm_delete", "domain": domain},
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

        answer = parse_yes_no(query)
        if answer == "yes":
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

        if answer == "no":
            workflow = append_completed_step(workflow, {}, "delete_canceled")
            return build_response(
                assistant_message="Deletion canceled. Your record is unchanged.",
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile=profile,
                state_manager=state_manager,
            )

        workflow = append_completed_step(
            workflow,
            {"intent": "delete", "stage": "confirm_delete", "domain": domain},
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
