"""
agent/tools/conversation_workflow.py
────────────────────────────────────
Shared multi-turn CRUD + intake workflow logic for diet/workout tools.

FIX LOG:
  [FIX-1] plan_feedback else branch was a dead-end loop. Added follow-up
          question handling and relaxed modification detection.
  [FIX-2] has_modification_details flag from classifier was ignored in
          plan_feedback. Now checked alongside classified_intent.
  [FIX-3] await_update_instructions stage had no explicit handler. Added
          dedicated block that routes user's change details into update flow.
  [FIX-4] generate_plan on updates didn't carry previous plan text. Added
          existing_plan param so LLM makes incremental changes, not from scratch.
  [FIX-5] Added answer_followup_question() for in-context questions during
          plan_feedback (e.g., "which brand should I use for protein?").
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
    """Centralised LLM-based intent classifier."""
    profile_summary = json.dumps(user_profile, indent=2) if user_profile else "(empty)"

    # [FIX-1] Added "followup_question" as a valid intent so the classifier
    # can distinguish follow-up questions from generic "other" messages.
    system = f"""\
You are an intent classifier for FITGEN.AI, a fitness coaching chatbot.
Your ONLY job is to decide what the user wants to do based on their message,
the current conversation context, and the last question the system asked.

Return ONLY valid JSON with these keys:
  intent: exactly one of "create", "update", "delete", "get", "confirm", "reject", "followup_question", "other"
  is_confirmation: true if the user is saying yes / agreeing / affirming ("yes", "sure", "looks good", "perfect", "correct")
  has_modification_details: true if the user provides specific changes (e.g., "change protein to 150g", "remove dairy", "goal muscle gain", "make it 1800 calories", "add more carbs", "swap chicken for paneer")
  is_generic_request: true if the user asks for a change but gives NO specifics (e.g., "modify it", "update", "change it")
  is_followup_question: true if the user is asking a question ABOUT the plan or related nutrition/fitness topic (e.g., "which brand of whey?", "what time should I eat this?", "can I swap rice for oats?", "brand konsa lu?")
  reason: one-sentence explanation of your decision

Classification rules:
1. If the user says yes / agrees / affirms / gives positive feedback ("looks good",
   "perfect", "awesome", "works for me", "correct", "confirmed") → intent="confirm", is_confirmation=true.
2. If the user says no / declines / disagrees ("no", "nope", "nah", "not now",
   "cancel", "incorrect") → intent="reject".
3. If the user wants a new plan created ("create a plan", "make me a plan",
   "build a workout") → intent="create".
4. If the user wants to modify / change / update something with specifics
   ("change my goal to muscle gain", "remove dairy", "add more protein",
   "make it 1800 calories", "swap chicken for fish") →
   intent="update", has_modification_details=true.
5. If the user wants to modify but gives NO specifics ("modify it", "update",
   "change it", "please modify") → intent="update", is_generic_request=true,
   has_modification_details=false.
6. If the user wants to delete their plan → intent="delete".
7. If the user wants to see / retrieve their plan → intent="get".
8. If the user is asking a QUESTION about the plan, a food, a supplement,
   a brand, timing, substitution, or any nutrition/fitness topic →
   intent="followup_question", is_followup_question=true.
   This includes questions in ANY language (Hindi, Hinglish, etc.).
   Examples: "brand konsa lu?", "which protein powder?", "kya mai oats kha sakta hu?",
   "can I replace banana with apple?", "what time should I have the pre-workout snack?"
9. If the user's message is a greeting or you truly cannot determine
   the intent → intent="other".

IMPORTANT: Messages that contain a question mark, or ask "which", "what",
"when", "how", "can I", "should I", "kya", "konsa", "kaise", "kab" are
almost always "followup_question", NOT "other". Do NOT classify questions
as "other".

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
    valid_intents = {"create", "update", "delete", "get", "confirm", "reject", "followup_question", "other"}
    if intent not in valid_intents:
        intent = "other"

    result = {
        "intent": intent,
        "is_confirmation": bool(data.get("is_confirmation", intent == "confirm")),
        "has_modification_details": bool(data.get("has_modification_details", False)),
        "is_generic_request": bool(data.get("is_generic_request", False)),
        "is_followup_question": bool(data.get("is_followup_question", intent == "followup_question")),
        "reason": str(data.get("reason", "")).strip(),
    }

    logger.info(
        "[classify_user_intent] query=%r → intent=%s confirm=%s mod_details=%s generic=%s followup=%s reason=%s",
        query[:80],
        result["intent"],
        result["is_confirmation"],
        result["has_modification_details"],
        result["is_generic_request"],
        result["is_followup_question"],
        result["reason"][:80] if result["reason"] else "(none)",
    )
    return result


# ──────────────────────────────────────────────────────────────────────
# [FIX-5] New: answer follow-up questions in the context of the plan
# ──────────────────────────────────────────────────────────────────────
def answer_followup_question(
    domain: str,
    query: str,
    profile: dict[str, Any],
    plan_text: str,
    system_prompt: str,
) -> str:
    """Answer a follow-up question about the plan without regenerating it.

    Handles questions like "which brand of whey should I use?",
    "can I swap rice for oats?", "what time should I eat dinner?", etc.
    """
    llm = ChatOpenAI(model=_LLM_MODEL, temperature=0.4)
    prompt = (
        f"The user has an active {domain} plan. Here is their profile:\n"
        f"{json.dumps(profile, indent=2)}\n\n"
        f"Here is their current plan:\n{plan_text}\n\n"
        f"The user is asking a follow-up question about this plan:\n"
        f"\"{query}\"\n\n"
        "Answer their question directly and concisely. Stay within the "
        "context of their plan and profile. Do NOT regenerate the full plan. "
        "Just answer the specific question."
    )
    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ])
    return resp.content


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
    allowed_set = set(allowed_fields)
    allowed = {k: v for k, v in data.items() if k in allowed_set and v not in (None, "")}
    if "sex" in allowed:
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
            "name", "age", "male", "female", "height", "weight",
            "goal", "activity", "cm", "kg",
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
    updates = extract_profile_updates(query, allowed_fields)
    allowed_set = set(allowed_fields)
    normalized_expected = [field for field in expected_fields if field in allowed_set]

    if normalized_expected and len(updates) == 0:
        first_expected = normalized_expected[0]
        parsed = _parse_single_field(first_expected, query)
        if parsed is not None:
            updates[first_expected] = parsed

    direct_parse_fields = {
        "name", "age", "sex", "height_cm", "weight_kg", "goal",
        "activity_level", "fitness_level", "workout_days",
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
    lines = [FIELD_QUESTION[field] for field in fields if field in FIELD_QUESTION]
    checklist = "\n".join(f"- {line}" for line in lines)
    return (
        "Please share the following details in one message so I can build your plan:\n"
        f"{checklist}\n\n"
        "Example: Name Kushagra, age 28, male, height 183 cm, weight 83 kg, goal fat loss, activity moderate."
    )


# ──────────────────────────────────────────────────────────────────────
# [FIX-4] generate_plan now accepts optional existing_plan for updates
# ──────────────────────────────────────────────────────────────────────
def generate_plan(
    domain: str,
    profile: dict[str, Any],
    query: str,
    system_prompt: str,
    *,
    existing_plan: str | None = None,
) -> str:
    """Generate or regenerate a plan.

    When existing_plan is provided (update flow), the LLM receives the
    current plan and the user's modification request so it can make
    targeted, incremental changes instead of rebuilding from scratch.
    """
    llm = ChatOpenAI(model=_LLM_MODEL, temperature=0.5)

    if existing_plan:
        # [FIX-4] Update flow: include existing plan + modification request
        prompt = (
            f"The user has an existing {domain} plan. Here is their profile:\n"
            f"{json.dumps(profile, indent=2)}\n\n"
            f"Here is their CURRENT plan:\n"
            f"{existing_plan}\n\n"
            f"The user wants the following changes:\n"
            f"\"{query}\"\n\n"
            f"Regenerate the FULL {domain} plan with these changes applied. "
            f"Keep everything else from the current plan that wasn't asked to change. "
            f"Provide practical, safe, actionable steps with concise structure."
        )
    else:
        # Create flow: generate from scratch
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
    pending_question = workflow.get("pending_question")

    # 2) Single centralised intent classification
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
    is_followup = classification.get("is_followup_question", False)

    logger.info(
        "[Workflow:%s] classified intent=%s (active=%s, stage=%s, followup=%s)",
        domain,
        classified_intent,
        active_intent,
        stage,
        is_followup,
    )

    # 3) Resolve effective intent
    if classified_intent in {"get", "delete", "update", "create"}:
        intent = classified_intent
    elif classified_intent in {"confirm", "reject"}:
        intent = active_intent or "other"
    elif classified_intent == "followup_question":
        # [FIX-1] Follow-up questions preserve the current intent flow
        intent = active_intent or "other"
    else:
        intent = active_intent or classified_intent

    if intent != active_intent and active_intent is not None:
        # [FIX-1] Don't reset stage for follow-up questions
        if classified_intent != "followup_question":
            stage = None
            workflow.pop("stage", None)

    # ──────────────────────────────────────────────────────────────────
    # STAGE: calendar_sync
    # ──────────────────────────────────────────────────────────────────
    if stage == "calendar_sync":
        stage_intent = workflow.get("intent") or active_intent
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
                try:
                    from agent.tools.calendar_integration import get_authorization_url
                    auth_url, oauth_state = get_authorization_url()
                except Exception as e:
                    logger.warning("[Workflow:%s] Calendar OAuth setup failed: %s", domain, e)
                    auth_url = None
                    oauth_state = None

                if auth_url:
                    update_calendar_sync(state_id, True)
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
                    update_calendar_sync(state_id, True)
                    workflow = append_completed_step(workflow, {}, "calendar_sync_enabled_no_oauth")
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
                workflow = append_completed_step(workflow, {}, "calendar_sync_skipped")
                return build_response(
                    assistant_message="No problem — I won't sync it to Google Calendar.",
                    state_id=state_id,
                    user_email=user_email,
                    workflow=workflow,
                    user_profile=profile,
                    state_manager=state_manager,
                    extra={"calendar_sync_requested": False},
                )

            # [FIX-1] Handle follow-up questions during calendar_sync
            if is_followup or classified_intent == "followup_question":
                plan_text = workflow.get("plan_text", "")
                if plan_text:
                    answer = answer_followup_question(
                        domain, query, profile, plan_text, plan_system_prompt,
                    )
                    # Stay in calendar_sync stage after answering
                    return build_response(
                        assistant_message=(
                            f"{answer}\n\n"
                            "By the way, would you still like to sync your plan to Google Calendar? (yes/no)"
                        ),
                        state_id=state_id,
                        user_email=user_email,
                        workflow=workflow,
                        user_profile=profile,
                        state_manager=state_manager,
                    )

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

    # ──────────────────────────────────────────────────────────────────
    # STAGE: plan_feedback  [FIX-1] [FIX-2] Major rewrite
    # ──────────────────────────────────────────────────────────────────
    if stage == "plan_feedback":
        stage_intent = workflow.get("intent") or active_intent

        # Interrupt: user switches to a different CRUD intent
        if classified_intent in {"get", "delete"} and classified_intent != stage_intent:
            logger.info("[Workflow:%s] plan_feedback interrupted by intent=%s", domain, classified_intent)
            workflow = append_completed_step(
                workflow,
                {"intent": classified_intent, "domain": domain},
                "plan_feedback_interrupted_by_intent",
            )
            stage = None
            workflow.pop("stage", None)
            # Fall through to CRUD handlers below

        # Path A: User confirms the plan
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

        # Path B: User rejects or wants modifications
        # [FIX-2] Now also checks has_modification_details flag
        elif (
            classified_intent == "reject"
            or classified_intent == "update"
            or has_modification_details
        ):
            if has_modification_details:
                # [FIX-2] User gave specific changes — go directly to update
                # instead of asking "what do you want to change?"
                logger.info("[Workflow:%s] plan_feedback → direct update with details", domain)

                record = get_record(state_id)
                existing_plan = (record or {}).get("plan_text", "") or workflow.get("plan_text", "")
                merged_profile = dict((record or {}).get("profile") or profile)
                updates = extract_profile_updates(query, required_fields)
                merged_profile.update(updates)

                # [FIX-4] Pass existing plan for incremental modification
                plan = generate_plan(
                    domain, merged_profile, query, plan_system_prompt,
                    existing_plan=existing_plan,
                )
                upsert_record(
                    state_id=state_id, domain=domain, profile=merged_profile,
                    plan_text=plan, calendar_sync=(record or {}).get("calendar_sync", False),
                )
                workflow = append_completed_step(
                    workflow,
                    {"intent": "update", "stage": "plan_feedback", "domain": domain,
                     "plan_text": plan,
                     "pending_question": "Please review the updated plan. Reply yes to confirm, or tell me what else to change."},
                    "plan_updated_from_feedback",
                )
                return build_response(
                    assistant_message=(
                        f"Done — I updated your {domain} plan:\n\n{plan}\n\n"
                        "Please review. Reply yes to confirm, or tell me what else to change."
                    ),
                    state_id=state_id,
                    user_email=user_email,
                    workflow=workflow,
                    user_profile=merged_profile,
                    state_manager=state_manager,
                    extra={"plan_text": plan},
                )
            else:
                # Generic rejection / "modify it" without specifics
                workflow = append_completed_step(
                    workflow,
                    {"intent": "update", "stage": "await_update_instructions", "domain": domain,
                     "pending_question": f"Tell me what you'd like to change in your {domain} plan."},
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

        # [FIX-1] Path C: Follow-up question about the plan
        elif is_followup or classified_intent == "followup_question":
            plan_text = workflow.get("plan_text", "")
            if plan_text:
                answer = answer_followup_question(
                    domain, query, profile, plan_text, plan_system_prompt,
                )
                # Stay in plan_feedback — don't break the flow
                workflow = append_completed_step(
                    workflow,
                    {"intent": workflow.get("intent", "create"), "stage": "plan_feedback",
                     "domain": domain, "plan_text": plan_text,
                     "pending_question": "Let me know if you want any changes to the plan, or reply yes to confirm it."},
                    "followup_answered",
                )
                return build_response(
                    assistant_message=(
                        f"{answer}\n\n"
                        "Let me know if you want any changes to the plan, or reply yes to confirm it."
                    ),
                    state_id=state_id,
                    user_email=user_email,
                    workflow=workflow,
                    user_profile=profile,
                    state_manager=state_manager,
                )
            else:
                # No plan text in context — shouldn't happen but handle gracefully
                logger.warning("[Workflow:%s] followup question but no plan_text in workflow", domain)
                workflow = append_completed_step(
                    workflow,
                    {"intent": workflow.get("intent", "create"), "stage": "plan_feedback", "domain": domain},
                    "followup_no_plan_text",
                )
                return build_response(
                    assistant_message=(
                        "I don't have your plan loaded in this context. "
                        "Would you like me to regenerate it, or would you like to start fresh?"
                    ),
                    state_id=state_id,
                    user_email=user_email,
                    workflow=workflow,
                    user_profile=profile,
                    state_manager=state_manager,
                )

        # [FIX-1] Path D: Anything else — be lenient, not a dead-end
        else:
            # Instead of blindly re-asking, try to interpret the message
            # as a modification request (user might be giving details in
            # a way the classifier didn't catch)
            query_length = len(query.strip().split())
            if query_length >= 3:
                # Non-trivial message — treat as modification attempt
                logger.info(
                    "[Workflow:%s] plan_feedback fallback: treating %d-word message as update",
                    domain, query_length,
                )
                record = get_record(state_id)
                existing_plan = (record or {}).get("plan_text", "") or workflow.get("plan_text", "")
                merged_profile = dict((record or {}).get("profile") or profile)
                updates = extract_profile_updates(query, required_fields)
                merged_profile.update(updates)

                plan = generate_plan(
                    domain, merged_profile, query, plan_system_prompt,
                    existing_plan=existing_plan,
                )
                upsert_record(
                    state_id=state_id, domain=domain, profile=merged_profile,
                    plan_text=plan, calendar_sync=(record or {}).get("calendar_sync", False),
                )
                workflow = append_completed_step(
                    workflow,
                    {"intent": "update", "stage": "plan_feedback", "domain": domain,
                     "plan_text": plan,
                     "pending_question": "Please review the updated plan. Reply yes to confirm, or tell me what else to change."},
                    "plan_updated_from_fallback",
                )
                return build_response(
                    assistant_message=(
                        f"I updated your {domain} plan based on your input:\n\n{plan}\n\n"
                        "Please review. Reply yes to confirm, or tell me what else to change."
                    ),
                    state_id=state_id,
                    user_email=user_email,
                    workflow=workflow,
                    user_profile=merged_profile,
                    state_manager=state_manager,
                    extra={"plan_text": plan},
                )
            else:
                # Very short ambiguous message (1-2 words) — re-ask once
                workflow = append_completed_step(
                    workflow,
                    {"intent": workflow.get("intent", "create"), "stage": "plan_feedback", "domain": domain,
                     "pending_question": "Reply yes to keep the plan, or tell me what you'd like changed."},
                    "plan_feedback_reprompted",
                )
                return build_response(
                    assistant_message=(
                        "Reply **yes** to keep the plan, or tell me what you'd like changed."
                    ),
                    state_id=state_id,
                    user_email=user_email,
                    workflow=workflow,
                    user_profile=profile,
                    state_manager=state_manager,
                )

    # ──────────────────────────────────────────────────────────────────
    # [FIX-3] STAGE: await_update_instructions (NEW explicit handler)
    # ──────────────────────────────────────────────────────────────────
    if stage == "await_update_instructions":
        # User was asked "what do you want to change?" — this message
        # IS their change request. Route directly to update.
        logger.info("[Workflow:%s] await_update_instructions — routing to update", domain)

        record = get_record(state_id)
        if not record:
            # No saved plan — fall back to create
            missing_fields = missing_profile_fields(profile, required_fields)
            workflow = append_completed_step(
                workflow,
                {"intent": "create", "stage": "collect_profile", "domain": domain,
                 "missing_fields": missing_fields},
                "await_update_no_record_fallback",
            )
            return build_response(
                assistant_message=(
                    "I couldn't find an existing plan to update. Let's create one.\n\n"
                    + build_profile_bulk_question(missing_fields)
                ),
                state_id=state_id,
                user_email=user_email,
                workflow=workflow,
                user_profile=profile,
                state_manager=state_manager,
            )

        existing_plan = record.get("plan_text", "")
        merged_profile = dict(record.get("profile") or {})
        merged_profile.update(profile)
        updates = extract_profile_updates(query, required_fields)
        merged_profile.update(updates)

        # [FIX-4] Pass existing plan for incremental modification
        plan = generate_plan(
            domain, merged_profile, query, plan_system_prompt,
            existing_plan=existing_plan,
        )
        upsert_record(
            state_id=state_id, domain=domain, profile=merged_profile,
            plan_text=plan, calendar_sync=record.get("calendar_sync", False),
        )
        workflow = append_completed_step(
            workflow,
            {"intent": "update", "stage": "plan_feedback", "domain": domain,
             "plan_text": plan,
             "pending_question": "Please review the updated plan. Reply yes to confirm, or tell me what else to change."},
            "plan_updated_from_instructions",
        )
        return build_response(
            assistant_message=(
                f"Done — I updated your {domain} plan:\n\n{plan}\n\n"
                "Please review. Reply yes to confirm, or tell me what else to change."
            ),
            state_id=state_id,
            user_email=user_email,
            workflow=workflow,
            user_profile=merged_profile,
            state_manager=state_manager,
            extra={"plan_text": plan},
        )

    # ──────────────────────────────────────────────────────────────────
    # INTENT: get
    # ──────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────
    # INTENT: create
    # ──────────────────────────────────────────────────────────────────
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

        if is_confirmation:
            pass  # Fall through to plan generation below
        elif classified_intent == "reject" or has_modification_details:
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
            "Please review it and tell me if you want any changes, or reply yes to confirm."
        )
        workflow = append_completed_step(
            workflow,
            {"intent": "create", "stage": "plan_feedback", "domain": domain,
             "plan_text": plan,
             "pending_question": "Please review the plan and tell me if you want any changes, or reply yes to confirm."},
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

    # ──────────────────────────────────────────────────────────────────
    # INTENT: update
    # ──────────────────────────────────────────────────────────────────
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

        existing_plan = record.get("plan_text", "")
        merged_profile = dict(record.get("profile") or {})
        merged_profile.update(profile)
        updates = extract_profile_updates(query, required_fields)
        merged_profile.update(updates)

        if not updates and is_generic:
            workflow = append_completed_step(
                workflow,
                {"intent": "update", "stage": "await_update_instructions", "domain": domain,
                 "pending_question": f"Tell me what you'd like to update in your {domain} plan."},
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

        # [FIX-4] Pass existing plan for incremental modification
        plan = generate_plan(
            domain, merged_profile, query, plan_system_prompt,
            existing_plan=existing_plan,
        )
        upsert_record(
            state_id=state_id, domain=domain, profile=merged_profile,
            plan_text=plan, calendar_sync=record.get("calendar_sync", False),
        )
        update_feedback_msg = (
            f"Done — I updated your {domain} plan:\n\n{plan}\n\n"
            "Please review. Reply yes to confirm, or tell me what else to change."
        )
        workflow = append_completed_step(
            workflow,
            {"intent": "update", "stage": "plan_feedback", "domain": domain,
             "plan_text": plan,
             "pending_question": "Please review. Reply yes to confirm, or tell me what else to change."},
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

    # ──────────────────────────────────────────────────────────────────
    # INTENT: delete
    # ──────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────
    # FALLBACK
    # ──────────────────────────────────────────────────────────────────
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