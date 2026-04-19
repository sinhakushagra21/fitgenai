"""
agent/tools/workout_tool.py
────────────────────────────
FITGEN.AI Workout Specialist Tool.

Self-contained multi-turn workflow for workout plan creation, modification,
retrieval, deletion, calendar/fit sync, and general fitness queries.

Flow: workout_tool() → execute() → handle_multi_turn()

Same architecture as diet_tool.py — each user message is classified into a
WorkoutIntent by the LLM, then dispatched to the corresponding handler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Annotated, Any, get_args

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from agent.prompts.workout_prompts import WORKOUT_PROMPTS
from agent.shared.llm_helpers import (
    answer_followup_question,
    answer_plan_question,
    classify_intent,
    extract_profile_updates,
    extract_profile_updates_with_fallback,
    generate_plan_name,
)
from agent.shared.plan_generation_loop import generate_plan_with_feedback
from agent.shared.profile_utils import (
    build_profile_bulk_question,
    build_profile_confirmation,
    missing_profile_fields,
    validate_profile_field,
)
from agent.shared.response_builder import append_completed_step, build_response
from agent.shared.types import (
    WORKOUT_ALL_FIELDS,
    WORKOUT_REQUIRED_FIELDS,
    WorkoutIntent,
)
from agent.db.repositories.user_repo import UserRepository
from agent.db.repositories.workout_plan_repo import WorkoutPlanRepository
from agent.error_utils import handle_exception
from agent.shared.plan_data import extract_plan_structured_data
from agent.state_manager import StateManager
from agent.tools.youtube_service import enrich_plan_with_videos
from agent.tracing import log_event, trace

logger = logging.getLogger("fitgen.workout_tool")

# ── Constants ────────────────────────────────────────────────────────

_DOMAIN = "workout"
_SYSTEM_PROMPT = WORKOUT_PROMPTS["few_shot"]
_VALID_INTENTS: list[str] = list(get_args(WorkoutIntent))

# Steps where a draft plan exists and can still be updated/confirmed.
_DRAFT_PLAN_STEPS = frozenset({
    "workout_plan_generated",
    "updated_workout_plan",
})

# Steps that indicate the workflow is fully complete.
_TERMINAL_STEPS = frozenset({
    "workout_plan_synced_to_google_calendar",
    "workout_plan_synced_to_google_fit",
})

# Intents that belong to the post-confirm sync flow.
_SYNC_INTENTS = frozenset({
    "sync_workout_to_google_calendar",
    "sync_workout_to_google_fit",
})


# ── Session Context ──────────────────────────────────────────────────

@dataclass
class WorkoutSessionContext:
    """All context needed for a single workout tool invocation."""

    state_manager: StateManager
    session_id: str
    user_email: str
    user_id: str
    profile: dict[str, Any]
    workflow: dict[str, Any]
    plan_text: str
    step_completed: str | None
    completed_steps: list[str] = field(default_factory=list)
    pending_question: str | None = None
    system_prompt: str = _SYSTEM_PROMPT

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> WorkoutSessionContext:
        """Build context from LangGraph state via StateManager."""
        sm = StateManager.from_state(state)
        wf = dict(sm.workflow or {})

        # ── Auto-reset truly completed / cross-domain workflows ─────
        # `workout_confirmed` is NOT a reset trigger — the user is still
        # being prompted for a sync decision, so preserve the full
        # workflow so the sync/both/skip reply can route correctly.
        _step = wf.get("step_completed")
        _wf_domain = wf.get("domain")
        _is_completed = (
            _step in _TERMINAL_STEPS
            or (_step and "synced" in _step)   # any *_synced_to_* step
        )
        _is_cross_domain = _wf_domain and _wf_domain != _DOMAIN
        if _is_completed or _is_cross_domain:
            _plan_id = wf.get("plan_id") if not _is_cross_domain else None
            logger.info(
                "[WorkoutFlow] Resetting workflow in from_state "
                "(step=%s, domain=%s, cross=%s, plan_id=%s)",
                _step, _wf_domain, _is_cross_domain, _plan_id,
            )
            wf = {"plan_id": _plan_id} if _plan_id else {}

        # Merge MongoDB stored profile (baseline) with session profile (overrides)
        profile: dict[str, Any] = {}
        if sm.user_email:
            try:
                profile = UserRepository.get_merged_profile(sm.user_email, domain="workout")
            except Exception as exc:  # noqa: BLE001
                handle_exception(
                    exc,
                    module="workout_tool",
                    context="load merged profile from MongoDB",
                    level="WARNING",
                    extra={"user_email": sm.user_email},
                )
                profile = {}
        profile.update(dict(sm.user_profile or {}))

        return cls(
            state_manager=sm,
            session_id=sm.context_id,
            user_email=sm.user_email,
            user_id=getattr(sm, "user_id", ""),
            profile=profile,
            workflow=wf,
            plan_text=wf.get("plan_text", ""),
            step_completed=wf.get("step_completed"),
            completed_steps=list(wf.get("completed_steps") or []),
            pending_question=wf.get("pending_question"),
        )


# ── Helpers ──────────────────────────────────────────────────────────

_BASE_FIELDS = frozenset({
    "name", "age", "sex", "height_cm", "weight_kg", "goal",
    "sleep_hours", "stress_level", "job_type",
})


def _split_profile_fields(
    profile: dict[str, Any], *, domain: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a flat profile dict into (base_fields, domain_fields)."""
    base: dict[str, Any] = {}
    domain_specific: dict[str, Any] = {}
    for k, v in profile.items():
        if v in (None, ""):
            continue
        (base if k in _BASE_FIELDS else domain_specific)[k] = v
    return base, domain_specific


def _get_raw_user_query(state: dict[str, Any]) -> str:
    """Extract the last HumanMessage content from graph state messages."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _build(
    message: str,
    ctx: WorkoutSessionContext,
    *,
    extra: dict[str, Any] | None = None,
) -> str:
    """Convenience wrapper around build_response."""
    return build_response(
        assistant_message=message,
        state_id=ctx.session_id,
        user_email=ctx.user_email,
        workflow=ctx.workflow,
        user_profile=ctx.profile,
        state_manager=ctx.state_manager,
        extra=extra,
    )


def _restore_side_query_state(
    tool_response_json: str,
    *,
    orig_workflow: dict[str, Any],
    orig_profile: dict[str, Any],
    orig_calendar: bool,
    context_id: str,
    user_email: str,
) -> str:
    """Post-process a side-query response so it doesn't clobber the
    active workflow belonging to the OTHER domain. Mirrors the helper
    in diet_tool.
    """
    import json as _json

    from agent.persistence import upsert_context_state

    try:
        payload = _json.loads(tool_response_json)
    except Exception:  # noqa: BLE001
        return tool_response_json

    su = payload.get("state_updates") or {}
    su["workflow"] = dict(orig_workflow or {})
    su["user_profile"] = dict(orig_profile or {})
    payload["state_updates"] = su
    payload.pop("extra", None)

    try:
        upsert_context_state(
            context_id=str(context_id),
            user_email=str(user_email or ""),
            user_profile=dict(orig_profile or {}),
            workflow=dict(orig_workflow or {}),
            calendar_sync_requested=bool(orig_calendar),
        )
        logger.info(
            "[WorkoutFlow] Side-query: restored caller state "
            "(workflow_domain=%s, step=%s)",
            (orig_workflow or {}).get("domain"),
            (orig_workflow or {}).get("step_completed"),
        )
    except Exception as exc:  # noqa: BLE001
        handle_exception(
            exc, module="workout_tool",
            context="restore side-query state",
            level="WARNING",
            extra={"context_id": str(context_id)},
        )

    return _json.dumps(payload)


def _update_workflow(
    ctx: WorkoutSessionContext,
    step_completed: str,
    step_name: str,
    **overrides: Any,
) -> None:
    """Update ctx.workflow in-place with step tracking + overrides."""
    merged = {
        "intent": overrides.pop("intent", ctx.workflow.get("intent")),
        "step_completed": step_completed,
        "domain": _DOMAIN,
        **overrides,
    }
    ctx.workflow = append_completed_step(ctx.workflow, merged, step_name)
    ctx.step_completed = step_completed


def _validate_extracted_profile(
    updates: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Run sanity checks on extracted profile fields."""
    errors: list[str] = []
    cleaned: dict[str, Any] = {}
    for field_name, value in updates.items():
        is_valid, error_msg = validate_profile_field(field_name, value)
        if is_valid:
            cleaned[field_name] = value
        else:
            errors.append(error_msg)
    return cleaned, errors


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry Points
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@tool
@trace(name="Workout Tool", run_type="tool", tags=["workout", "tool"])
def workout_tool(
    query: str,
    state: Annotated[dict[str, Any], InjectedState],
    side_query: bool = False,
) -> str:
    """Exercise & training specialist. Handles workout plans, training \
splits, reps/sets, body-part and lift guidance (chest, back, traps, legs, \
biceps, deadlift, squat, etc.), cardio, mobility, sync to Google Calendar \
/ Google Fit, and general exercise Q&A.

    Use this tool for anything workout-, exercise-, lift-, or body-part-
    related, including creating, updating, retrieving, confirming, or
    deleting a workout plan, and for on-topic follow-up training questions.

    The ``query`` MUST be the user's EXACT message, verbatim.
    Returns JSON with assistant_message and state_updates.
    """
    raw_query = _get_raw_user_query(state)
    effective_query = raw_query or query

    logger.info(
        "[WorkoutTool] query: %s  side_query=%s",
        effective_query[:120], side_query,
    )

    return execute(effective_query, state, side_query=side_query)


def execute(query: str, state: dict[str, Any], *, side_query: bool = False) -> str:
    """Execution entrypoint — delegates to handle_multi_turn."""
    return handle_multi_turn(query=query, state=state, side_query=side_query)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Orchestrator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_INTENT_HANDLERS: dict[str, Any] = {}


def handle_multi_turn(
    *,
    query: str,
    state: dict[str, Any],
    side_query: bool = False,
) -> str:
    """Orchestrate the workout multi-turn workflow.

    When ``side_query=True`` we skip classification, force intent to
    ``general_workout_query``, and restore the caller's original
    workflow/profile afterwards so a cross-domain probe never clobbers
    the active diet flow.
    """
    _orig_workflow = dict(state.get("workflow") or {}) if side_query else None
    _orig_profile = dict(state.get("user_profile") or {}) if side_query else None
    _orig_calendar = bool(state.get("calendar_sync_requested", False))

    ctx = WorkoutSessionContext.from_state(state)

    logger.info(
        "[WorkoutFlow] session=%s step=%s profile_fields=%d plan_len=%d",
        ctx.session_id,
        ctx.step_completed,
        len(ctx.profile),
        len(ctx.plan_text),
    )

    if side_query:
        logger.info("[WorkoutFlow] Side-query mode — forcing general_workout_query")
        try:
            result = _handle_general_workout_query(query, ctx)
        except Exception as exc:  # noqa: BLE001
            handle_exception(
                exc, module="workout_tool",
                context="handler:general_workout_query (side)",
                extra={"session_id": ctx.session_id},
            )
            result = _build(
                "I had trouble answering that. Please try again.", ctx,
            )
        return _restore_side_query_state(
            result,
            orig_workflow=_orig_workflow or {},
            orig_profile=_orig_profile or {},
            orig_calendar=_orig_calendar,
            context_id=ctx.session_id,
            user_email=ctx.user_email,
        )

    # has_plan tells the classifier whether *any* plan exists — draft
    # (in session) OR confirmed (in MongoDB). Used by classifier RULE 4
    # to gate sync requests. Without including "workout_confirmed", a
    # "sync to calendar" reply at the post-confirm prompt gets
    # mis-classified as general_workout_query.
    _has_draft_plan = (
        (bool(ctx.plan_text) and ctx.step_completed in _DRAFT_PLAN_STEPS)
        or ctx.step_completed == "workout_confirmed"
    )
    try:
        classification = classify_intent(
            query,
            domain=_DOMAIN,
            valid_intents=_VALID_INTENTS,
            step_completed=ctx.step_completed,
            user_profile=ctx.profile,
            pending_question=ctx.pending_question,
            has_plan=_has_draft_plan,
        )
    except Exception as exc:  # noqa: BLE001
        handle_exception(
            exc,
            module="workout_tool",
            context="intent classification",
            extra={
                "session_id": ctx.session_id,
                "step": ctx.step_completed,
                "query_preview": query[:120],
            },
        )
        return _build(
            "I had trouble understanding your request. Please try again.",
            ctx,
        )

    intent = classification["user_intent"]
    reason = classification.get("reason", "")
    log_event(
        "workout.intent_classified",
        module="workout_tool",
        intent=intent,
        reason=reason[:120],
        session_id=ctx.session_id,
    )

    handler = _INTENT_HANDLERS.get(intent)
    if handler is None:
        log_event(
            "workout.missing_handler",
            level="WARNING",
            module="workout_tool",
            intent=intent,
            session_id=ctx.session_id,
        )
        return _build(
            f"I can help you create, update, delete, or view your {_DOMAIN} "
            "plan. What would you like to do?",
            ctx,
        )

    try:
        return handler(query, ctx)
    except Exception as exc:  # noqa: BLE001
        handle_exception(
            exc,
            module="workout_tool",
            context=f"handler:{intent}",
            extra={
                "session_id": ctx.session_id,
                "intent": intent,
                "step": ctx.step_completed,
            },
        )
        return _build(
            "I had trouble processing your request. Please try again.",
            ctx,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Intent Handlers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _handle_create_workout(query: str, ctx: WorkoutSessionContext) -> str:
    """Handle create_workout intent — multi-step profile collection + plan generation."""
    # ── Cross-domain reset: clear stale workflow from a different tool ──
    _wf_domain = ctx.workflow.get("domain")
    if _wf_domain and _wf_domain != _DOMAIN:
        logger.info("[WorkoutFlow] Resetting stale %s workflow for new workout create", _wf_domain)
        ctx.workflow = {}
        ctx.plan_text = ""
        ctx.step_completed = None
        ctx.completed_steps = []
        ctx.pending_question = None

    step = ctx.step_completed

    # ── Step A: Fresh create or re-entering ──
    if step is None or step in ("workout_confirmed", "workout_plan_generated",
                                 "updated_workout_plan",
                                 "workout_plan_synced_to_google_calendar",
                                 "workout_plan_synced_to_google_fit"):
        updates = extract_profile_updates_with_fallback(
            query, WORKOUT_REQUIRED_FIELDS, WORKOUT_ALL_FIELDS
        )
        cleaned, validation_errors = _validate_extracted_profile(updates)
        ctx.profile.update(cleaned)

        missing = missing_profile_fields(ctx.profile, WORKOUT_REQUIRED_FIELDS)
        if missing:
            _update_workflow(
                ctx,
                step_completed="prompted_for_user_profile_data",
                step_name="profile_collection_started",
                intent="create_workout",
                missing_fields=missing,
                pending_question=build_profile_bulk_question(missing),
            )
            msg = build_profile_bulk_question(missing)
            if validation_errors:
                msg = (
                    "Some values seem off:\n"
                    + "\n".join(f"- {e}" for e in validation_errors)
                    + "\n\n" + msg
                )
            return _build(msg, ctx)

        # Show ALL fields the user provided, not just required ones
        confirm_msg = build_profile_confirmation(ctx.profile, WORKOUT_ALL_FIELDS)
        _update_workflow(
            ctx,
            step_completed="user_profile_mapped",
            step_name="profile_mapped",
            intent="create_workout",
            pending_question=confirm_msg,
        )
        return _build(confirm_msg, ctx)

    # ── Step B: User providing profile data ──
    if step == "prompted_for_user_profile_data":
        expected = ctx.workflow.get("missing_fields") or WORKOUT_REQUIRED_FIELDS
        updates = extract_profile_updates_with_fallback(
            query, expected, WORKOUT_ALL_FIELDS
        )
        cleaned, validation_errors = _validate_extracted_profile(updates)
        ctx.profile.update(cleaned)

        missing = missing_profile_fields(ctx.profile, WORKOUT_REQUIRED_FIELDS)
        if missing:
            _update_workflow(
                ctx,
                step_completed="prompted_for_user_profile_data",
                step_name="profile_reprompted",
                intent="create_workout",
                missing_fields=missing,
                pending_question=build_profile_bulk_question(missing),
            )
            msg = f"Thanks! I still need a few more details:\n\n{build_profile_bulk_question(missing)}"
            if validation_errors:
                msg = (
                    "Some values seem off:\n"
                    + "\n".join(f"- {e}" for e in validation_errors)
                    + "\n\n" + msg
                )
            return _build(msg, ctx)

        # Show ALL fields the user provided, not just required ones
        confirm_msg = build_profile_confirmation(ctx.profile, WORKOUT_ALL_FIELDS)
        _update_workflow(
            ctx,
            step_completed="user_profile_mapped",
            step_name="profile_mapped",
            intent="create_workout",
            pending_question=confirm_msg,
        )
        return _build(confirm_msg, ctx)

    # ── Step C: Profile confirmed → generate plan ──
    if step == "user_profile_mapped":
        # The user may say "yes" OR "yes but change X to Y".
        # Extract any last-minute profile tweaks before generating.
        _tweaks = extract_profile_updates(query, WORKOUT_ALL_FIELDS)
        if _tweaks:
            logger.info("[WorkoutFlow] Applying last-minute profile tweaks: %s", _tweaks)
            ctx.profile.update(_tweaks)

        raw_plan, _eval_meta = generate_plan_with_feedback(
            _DOMAIN, ctx.profile, query, ctx.system_prompt,
        )
        logger.info(
            "[WorkoutFlow] plan_eval_loop: attempts=%s chosen=%s hard_pass=%s "
            "light=%s",
            _eval_meta.get("attempts"),
            _eval_meta.get("chosen_index"),
            _eval_meta.get("hard_passed"),
            _eval_meta.get("light_score"),
        )

        # Extract structured data (schedule) and strip from display
        plan_markdown, structured_data = extract_plan_structured_data(raw_plan)

        # Enrich exercise tables with YouTube tutorial links
        plan_markdown = enrich_plan_with_videos(plan_markdown)

        # NOTE: Plan is NOT saved to MongoDB here — only on confirm.
        # This keeps the DB clean: only confirmed plans are persisted.

        _update_workflow(
            ctx,
            step_completed="workout_plan_generated",
            step_name="plan_generated",
            intent="create_workout",
            plan_text=plan_markdown,
            structured_data=structured_data,
            pending_question=(
                "Please review your plan and tell me if you want any "
                "changes, or confirm to proceed."
            ),
        )

        return _build(
            f"Here's your personalized workout plan:\n\n{plan_markdown}\n\n"
            "Please review. Reply **yes** to confirm, or tell me what "
            "you'd like to change.",
            ctx,
            extra={"plan_text": plan_markdown},
        )

    logger.warning("[WorkoutFlow] create_workout: unexpected step=%s", step)
    return _build("Let's start fresh. What would you like me to create?", ctx)


def _handle_confirm_workout(query: str, ctx: WorkoutSessionContext) -> str:
    """Handle confirm_workout intent — save profile, offer sync options.

    Guard: Only actually confirm if a plan has been generated.
    If the user says "yes" during profile collection/mapping, delegate
    back to _handle_create_workout to continue the create flow.
    """
    # ── Guard: plan must exist before we can confirm it ──
    if ctx.step_completed in (
        None,
        "prompted_for_user_profile_data",
        "user_profile_mapped",
    ):
        logger.info(
            "[WorkoutFlow] confirm_workout at step=%s → delegating to "
            "create_workout (plan not yet generated)",
            ctx.step_completed,
        )
        return _handle_create_workout(query, ctx)

    if ctx.step_completed not in ("workout_plan_generated", "updated_workout_plan"):
        return _build(
            "There's no plan to confirm yet. Would you like me to create "
            "a workout plan?",
            ctx,
        )

    # ── Persist to MongoDB only on confirm ──
    _plan_id = None
    if ctx.user_email:
        # 1. Ensure user exists in MongoDB
        user_doc = UserRepository.find_or_create(ctx.user_email)
        _user_id = str(user_doc["_id"])

        # 2. Save / update profile (base + workout sub-docs)
        _base, _workout = _split_profile_fields(ctx.profile, domain="workout")
        UserRepository.update_profile(ctx.user_email, base=_base, workout=_workout)
        logger.info("[WorkoutFlow] Profile saved to users collection for %s", ctx.user_email)

        # 3. Generate a catchy plan name via LLM
        plan_text = ctx.workflow.get("plan_text", ctx.plan_text)
        _structured = ctx.workflow.get("structured_data", {})
        _plan_name = ""
        if plan_text:
            try:
                _plan_name = generate_plan_name(_DOMAIN, ctx.profile, plan_text)
            except Exception as exc:
                logger.warning("[WorkoutFlow] Plan naming failed: %s", exc)

        # 4. Create confirmed plan in workout_plans collection
        if plan_text:
            _plan_id = WorkoutPlanRepository.create(
                user_id=_user_id,
                session_id=ctx.session_id,
                profile_snapshot=dict(ctx.profile),
                plan_markdown=plan_text,
                structured_data=_structured,
                status="confirmed",
                name=_plan_name,
            )
            logger.info("[WorkoutFlow] Plan created & confirmed: plan_id=%s name=%r", _plan_id, _plan_name)
    else:
        logger.warning(
            "[WorkoutFlow] confirm_workout: no user_email — skipping MongoDB save. "
            "Set FITGEN_USER_EMAIL in .env to enable persistence."
        )

    # ── Preserve workflow context for the sync follow-up ──────────────
    # We intentionally KEEP `workflow.domain="workout"` and
    # `step_completed="workout_confirmed"` so the router can
    # deterministically dispatch the user's next reply ("calendar" /
    # "both" / "skip" / …) to this tool. Plan markdown + structured_data
    # are cleared because the plan now lives in MongoDB; sync handlers
    # pull it via _resolve_plan_text.
    ctx.plan_text = ""
    ctx.completed_steps = []

    _update_workflow(
        ctx,
        step_completed="workout_confirmed",
        step_name="workout_confirmed",
        intent="confirm_workout",
        plan_id=str(_plan_id) if _plan_id else None,
        plan_text="",
        structured_data={},
        pending_question=(
            "Would you like to sync to Google Calendar, Google Fit, or both?"
        ),
    )

    return _build(
        "Your workout plan is confirmed! 💪\n\n"
        "Would you like to sync it to:\n"
        "- **Google Calendar** (schedule workouts as events)\n"
        "- **Google Fit** (log activity data)\n"
        "- **Both**\n\n"
        "Or just say **done** if you're all set!",
        ctx,
    )


def _wipe_sync_workflow(ctx: WorkoutSessionContext) -> None:
    """Clear all workflow state after sync decision is terminal."""
    ctx.workflow = {}
    ctx.plan_text = ""
    ctx.step_completed = None
    ctx.completed_steps = []
    ctx.pending_question = None


def _handle_skip_sync_workout(query: str, ctx: WorkoutSessionContext) -> str:
    """User declined sync after confirming the plan — wipe workflow."""
    _wipe_sync_workflow(ctx)
    return _build(
        "All set! Your workout plan is confirmed. "
        "Let me know whenever you need anything else. 💪",
        ctx,
    )


def _handle_sync_to_both(query: str, ctx: WorkoutSessionContext) -> str:
    """Sync the plan to BOTH Google Calendar and Google Fit."""
    _plan = _resolve_plan_text(ctx)
    if not _plan:
        return _build(
            "You don't have a workout plan to sync yet. Create one first!",
            ctx,
        )

    try:
        from agent.tools.calendar_integration import (
            get_authorization_url, save_oauth_context,
        )

        auth_url, oauth_state = get_authorization_url()
        save_oauth_context(
            plan_text=_plan, domain=_DOMAIN,
            profile=ctx.profile, sync_target="both",
        )

        _plan_id = ctx.workflow.get("plan_id")
        if _plan_id:
            WorkoutPlanRepository.update_plan(
                _plan_id, calendar_synced=True, fit_synced=True,
            )

        _update_workflow(
            ctx,
            step_completed="workout_plan_synced_to_google_calendar",
            step_name="sync_both_started",
            intent="sync_workout_to_both",
            oauth_state=oauth_state,
            pending_question="Click the button in the sidebar to complete the sync.",
        )

        _wipe_sync_workflow(ctx)

        return _build(
            "🔗 **Ready to connect Google — syncing to both Calendar & Fit!**\n\n"
            "Click the **\"📅 Connect Google Calendar\"** button in the sidebar "
            "to authorise. One sign-in covers both Google Calendar and Google "
            "Fit.\n\n"
            f"Or open this link directly: [Authorize FITGEN.AI]({auth_url})",
            ctx,
            extra={
                "calendar_sync_requested": True,
                "google_fit_sync_requested": True,
                "calendar_auth_url": auth_url,
            },
        )

    except Exception as exc:  # noqa: BLE001
        handle_exception(
            exc,
            module="workout_tool",
            context="sync to both (calendar + fit)",
            extra={"session_id": ctx.session_id},
        )
        return _build(
            "Google integration is not configured. Please check "
            "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in your .env file.",
            ctx,
        )


def _handle_update_workout(query: str, ctx: WorkoutSessionContext) -> str:
    """Handle update_workout intent — map changes, regenerate plan."""
    if not ctx.plan_text:
        logger.info("[WorkoutFlow] update_workout: no existing plan")
        _update_workflow(
            ctx,
            step_completed="prompted_for_user_profile_data",
            step_name="update_no_plan_redirect",
            intent="create_workout",
            missing_fields=missing_profile_fields(ctx.profile, WORKOUT_REQUIRED_FIELDS),
        )
        missing = missing_profile_fields(ctx.profile, WORKOUT_REQUIRED_FIELDS)
        if missing:
            return _build(
                "I don't have an existing workout plan to update. "
                "Let's create one first!\n\n"
                + build_profile_bulk_question(missing),
                ctx,
            )
        ctx.step_completed = "user_profile_mapped"
        return _handle_create_workout(query, ctx)

    updates = extract_profile_updates(query, WORKOUT_ALL_FIELDS)
    cleaned, _ = _validate_extracted_profile(updates)
    ctx.profile.update(cleaned)

    # Generate updated plan (pass existing plan for incremental changes)
    plan_markdown, _eval_meta = generate_plan_with_feedback(
        _DOMAIN, ctx.profile, query, ctx.system_prompt,
        existing_plan=ctx.plan_text,
    )
    logger.info(
        "[WorkoutFlow] plan_eval_loop (update): attempts=%s chosen=%s "
        "hard_pass=%s light=%s",
        _eval_meta.get("attempts"),
        _eval_meta.get("chosen_index"),
        _eval_meta.get("hard_passed"),
        _eval_meta.get("light_score"),
    )
    # Enrich exercise tables with YouTube tutorial links
    plan_markdown = enrich_plan_with_videos(plan_markdown)

    # NOTE: Updated plan is NOT saved to MongoDB here — only on confirm.

    _update_workflow(
        ctx,
        step_completed="updated_workout_plan",
        step_name="plan_updated",
        intent="update_workout",
        plan_text=plan_markdown,
        pending_question=(
            "Please review the updated plan. Reply yes to confirm, "
            "or tell me what else to change."
        ),
    )

    return _build(
        f"Done — I've updated your workout plan:\n\n{plan_markdown}\n\n"
        "Please review. Reply **yes** to confirm, or tell me what else "
        "to change.",
        ctx,
        extra={"plan_text": plan_markdown},
    )


def _handle_get_workout(query: str, ctx: WorkoutSessionContext) -> str:
    """Handle get_workout intent — resolve which plan, then let the LLM answer.

    Model C: if the resolver returns an archived plan (matched on a
    descriptor like "strength" / "ppl"), we answer from it and prompt
    the user to ``restore`` it to active.
    """
    resolved = None
    if ctx.user_id:
        try:
            from agent.rag.personal.plan_resolver import resolve_plan
            resolved = resolve_plan(ctx.user_id, query, plan_type="workout")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[workout_tool] plan_resolver failed: %s", exc)

    _plan = ""
    _plan_id = None
    _is_archived = False
    _match_reason = ""
    if resolved:
        p = resolved["plan"]
        _plan = p.get("plan_markdown", "") or ""
        _plan_id = p.get("_id")
        _is_archived = bool(resolved.get("is_archived"))
        _match_reason = str(resolved.get("match_reason") or "")
    elif ctx.user_id:
        latest = WorkoutPlanRepository.find_latest_by_user(ctx.user_id)
        if latest:
            _plan = latest.get("plan_markdown", "") or ""
            _plan_id = latest.get("_id")

    if not _plan:
        return _build(
            "No workout plan found. Would you like me to "
            "create one? Just say **create a workout plan**!",
            ctx,
        )

    # get is a one-shot operation — clear workflow so next call starts fresh
    ctx.workflow = {}
    ctx.plan_text = ""
    ctx.step_completed = None
    ctx.completed_steps = []
    ctx.pending_question = None

    retrieved_context = ""
    if ctx.user_id and not _is_archived:
        try:
            from agent.rag.personal.retriever import PersonalRAG
            chunks = PersonalRAG.retrieve(
                user_id=ctx.user_id, query=query,
                plan_type="workout", top_k=5,
            )
            if chunks:
                retrieved_context = "\n\n".join(c.render() for c in chunks)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[workout_tool] RAG retrieve failed: %s", exc)

    answer = answer_plan_question(
        _DOMAIN, _plan, query, context=retrieved_context,
    )

    if _is_archived and _plan_id is not None:
        ctx.workflow = {
            "intent": "get_workout",
            "domain": _DOMAIN,
            "pending_restore_plan_id": str(_plan_id),
        }
        ctx.step_completed = "archived_plan_surfaced"
        ctx.pending_question = (
            "Reply **restore** to make this plan active again."
        )
        suffix = (
            "\n\n_📌 This is an **archived** plan "
            f"(match: {_match_reason or 'descriptor'}). "
            "Reply **restore** to make it your active workout plan again._"
        )
        answer = f"{answer}{suffix}"

    return _build(answer, ctx)


def _handle_delete_workout(query: str, ctx: WorkoutSessionContext) -> str:
    """Handle delete_workout intent — two-step confirmation."""
    if ctx.step_completed != "delete_confirmation_pending":
        _update_workflow(
            ctx,
            step_completed="delete_confirmation_pending",
            step_name="delete_requested",
            intent="delete_workout",
            pending_question="Confirm deletion? (yes/no)",
        )
        return _build(
            "Are you sure you want to delete your workout plan and profile? "
            "This cannot be undone. Reply **yes** to confirm or **no** "
            "to cancel.",
            ctx,
        )

    classification = classify_intent(
        query,
        domain=_DOMAIN,
        valid_intents=_VALID_INTENTS,
        step_completed=ctx.step_completed,
        user_profile=ctx.profile,
        pending_question=ctx.pending_question,
        has_plan=bool(ctx.plan_text) and ctx.step_completed in _DRAFT_PLAN_STEPS,
    )

    if classification["user_intent"] == "confirm_workout":
        # Archive plan in workout_plans collection
        _plan_id = ctx.workflow.get("plan_id")
        if _plan_id:
            WorkoutPlanRepository.archive(_plan_id)
            logger.info("[WorkoutFlow] Plan archived: plan_id=%s", _plan_id)

        ctx.profile = {}
        ctx.workflow = {}
        ctx.plan_text = ""
        _update_workflow(ctx, step_completed=None, step_name="delete_completed")
        return _build(
            "Your workout plan and profile have been deleted. "
            "Let me know if you'd like to create a new one!",
            ctx,
        )

    _update_workflow(
        ctx,
        step_completed=ctx.workflow.get("_prev_step"),
        step_name="delete_cancelled",
    )
    return _build("Deletion cancelled. Your workout plan is unchanged.", ctx)


def _resolve_plan_text(ctx: WorkoutSessionContext) -> str:
    """Get plan markdown — from ctx, workflow, or MongoDB (by plan_id)."""
    if ctx.plan_text:
        return ctx.plan_text
    _plan_id = ctx.workflow.get("plan_id")
    if _plan_id:
        doc = WorkoutPlanRepository.find_by_id(_plan_id)
        if doc and doc.get("plan_markdown"):
            return doc["plan_markdown"]
    if ctx.user_id:
        doc = WorkoutPlanRepository.find_latest_by_user(ctx.user_id, status="confirmed")
        if doc and doc.get("plan_markdown"):
            return doc["plan_markdown"]
    return ""


def _handle_sync_to_google_calendar(
    query: str, ctx: WorkoutSessionContext,
) -> str:
    """Handle sync_workout_to_google_calendar intent."""
    _plan = _resolve_plan_text(ctx)
    if not _plan:
        return _build("You don't have a workout plan to sync yet. Create one first!", ctx)

    try:
        from agent.tools.calendar_integration import (
            get_authorization_url, save_oauth_context,
        )

        auth_url, oauth_state = get_authorization_url()
        save_oauth_context(
            plan_text=_plan, domain=_DOMAIN,
            profile=ctx.profile, sync_target="calendar",
        )

        # Mark plan as calendar-synced in MongoDB
        _plan_id = ctx.workflow.get("plan_id")
        if _plan_id:
            WorkoutPlanRepository.update_plan(_plan_id, calendar_synced=True)

        _update_workflow(
            ctx,
            step_completed="workout_plan_synced_to_google_calendar",
            step_name="calendar_sync_started",
            intent="sync_workout_to_google_calendar",
            oauth_state=oauth_state,
        )

        # Terminal branch — wipe residual workflow.
        _wipe_sync_workflow(ctx)

        return _build(
            "🔗 **Ready to connect Google Calendar!**\n\n"
            "Click the **\"📅 Connect Google Calendar\"** button in the "
            "sidebar to sign in with Google and sync your workouts.\n\n"
            f"Or open this link directly: [Authorize FITGEN.AI]({auth_url})",
            ctx,
            extra={"calendar_sync_requested": True, "calendar_auth_url": auth_url},
        )

    except Exception as exc:  # noqa: BLE001
        handle_exception(
            exc,
            module="workout_tool",
            context="google calendar sync",
            extra={"session_id": ctx.session_id},
        )
        return _build(
            "Google Calendar integration is not configured. "
            "Please check your .env file.",
            ctx,
        )


def _handle_sync_to_google_fit(
    query: str, ctx: WorkoutSessionContext,
) -> str:
    """Handle sync_workout_to_google_fit intent."""
    _plan = _resolve_plan_text(ctx)
    if not _plan:
        return _build("You don't have a workout plan to sync yet. Create one first!", ctx)

    try:
        from agent.tools.calendar_integration import (
            get_authorization_url, save_oauth_context,
        )

        auth_url, _ = get_authorization_url()
        save_oauth_context(
            plan_text=_plan, domain=_DOMAIN,
            profile=ctx.profile, sync_target="google_fit",
        )

        # Mark plan as fit-synced in MongoDB
        _plan_id = ctx.workflow.get("plan_id")
        if _plan_id:
            WorkoutPlanRepository.update_plan(_plan_id, fit_synced=True)

        _update_workflow(
            ctx,
            step_completed="workout_plan_synced_to_google_fit",
            step_name="google_fit_sync_started",
            intent="sync_workout_to_google_fit",
        )

        # Terminal branch — wipe residual workflow.
        _wipe_sync_workflow(ctx)

        return _build(
            "💪 **Ready to connect Google Fit!**\n\n"
            "Click the **\"💪 Sync to Google Fit\"** button in the sidebar "
            "to sign in with Google and sync your plan data.\n\n"
            f"Or open this link directly: [Authorize Google Fit]({auth_url})",
            ctx,
            extra={"google_fit_sync_requested": True},
        )

    except Exception as exc:  # noqa: BLE001
        handle_exception(
            exc,
            module="workout_tool",
            context="google fit sync",
            extra={"session_id": ctx.session_id},
        )
        return _build(
            "Google Fit integration is not configured. "
            "Please check your .env file.",
            ctx,
        )


def _handle_general_workout_query(
    query: str, ctx: WorkoutSessionContext,
) -> str:
    """Handle general_workout_query intent — contextual Q&A."""
    # Use workout plan from MongoDB as context (avoids cross-domain mix-up)
    _plan = ""
    if ctx.user_id:
        latest = WorkoutPlanRepository.find_latest_by_user(ctx.user_id)
        if latest and latest.get("plan_markdown"):
            _plan = latest["plan_markdown"]
    if not _plan and ctx.workflow.get("domain") == "workout":
        _plan = ctx.plan_text  # fallback to session only if same domain

    # Personal RAG — fetch top-k chunks from the user's own plan + memory.
    retrieved_context = ""
    if ctx.user_id:
        try:
            from agent.rag.personal.retriever import PersonalRAG
            chunks = PersonalRAG.retrieve(
                user_id=ctx.user_id, query=query,
                plan_type="workout", top_k=5,
            )
            if chunks:
                retrieved_context = "\n\n".join(c.render() for c in chunks)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[workout_tool] RAG retrieve failed: %s", exc)

    answer = answer_followup_question(
        _DOMAIN, query, ctx.profile, _plan, ctx.system_prompt,
        context=retrieved_context,
    )
    return _build(answer, ctx)


# ── Register intent handlers ─────────────────────────────────────────

def _handle_restore_workout_plan(
    query: str, ctx: WorkoutSessionContext,
) -> str:
    """Handle restore_workout_plan — reactivate an archived plan.

    Mirrors ``_handle_restore_diet_plan`` in diet_tool: picks a target
    plan (from pending_restore_plan_id or the PlanResolver), archives
    the current active plan, flips the target to confirmed, and re-
    indexes chunks.
    """
    target_id: str | None = ctx.workflow.get("pending_restore_plan_id")
    target_plan: dict[str, Any] | None = None

    if target_id:
        target_plan = WorkoutPlanRepository.find_by_id(target_id)

    if target_plan is None and ctx.user_id:
        try:
            from agent.rag.personal.plan_resolver import resolve_plan
            resolved = resolve_plan(ctx.user_id, query, plan_type="workout")
            if resolved and resolved.get("is_archived"):
                target_plan = resolved["plan"]
                target_id = str(target_plan["_id"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("[workout_tool] restore resolver failed: %s", exc)

    if target_plan is None:
        return _build(
            "I couldn't find an archived workout plan to restore. "
            "Ask me to **get my old push-pull-legs plan** (or similar) "
            "first, then reply **restore**.",
            ctx,
        )

    try:
        _oid = target_plan["_id"]
        if ctx.user_id:
            active = WorkoutPlanRepository.find_latest_by_user(
                ctx.user_id, status="confirmed",
            ) or WorkoutPlanRepository.find_latest_by_user(
                ctx.user_id, status="draft",
            )
            if active and active["_id"] != _oid:
                WorkoutPlanRepository.archive(active["_id"])

        WorkoutPlanRepository.update_plan(_oid, status="confirmed")
        try:
            from agent.db.repositories.plan_chunks_repo import (
                PlanChunksRepository,
            )
            PlanChunksRepository.reactivate_by_plan(_oid, status="confirmed")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[workout_tool] reactivate chunks failed: %s", exc)
        logger.info("[WorkoutFlow] Plan restored: plan_id=%s", _oid)
    except Exception as exc:  # noqa: BLE001
        handle_exception(exc, "workout_tool._handle_restore_workout_plan",
                         level="ERROR")
        return _build(
            "Sorry — I couldn't restore that plan right now. "
            "Please try again.",
            ctx,
        )

    plan_name = target_plan.get("name") or "your workout plan"
    ctx.plan_text = ""
    ctx.workflow = {
        "intent": "restore_workout_plan",
        "domain": _DOMAIN,
        "plan_id": str(_oid),
    }
    ctx.step_completed = "workout_confirmed"
    ctx.pending_question = None

    return _build(
        f"✅ Restored **{plan_name}** as your active workout plan. "
        "Ask me anything about it, or say **sync to calendar** / "
        "**sync to fit** to push it out.",
        ctx,
    )


_INTENT_HANDLERS.update({
    "create_workout": _handle_create_workout,
    "update_workout": _handle_update_workout,
    "get_workout": _handle_get_workout,
    "delete_workout": _handle_delete_workout,
    "confirm_workout": _handle_confirm_workout,
    "sync_workout_to_google_calendar": _handle_sync_to_google_calendar,
    "sync_workout_to_google_fit": _handle_sync_to_google_fit,
    "sync_workout_to_both": _handle_sync_to_both,
    "skip_sync_workout": _handle_skip_sync_workout,
    "general_workout_query": _handle_general_workout_query,
    "restore_workout_plan": _handle_restore_workout_plan,
})
