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
    classify_intent,
    extract_profile_updates,
    extract_profile_updates_with_fallback,
    generate_plan,
)
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
from agent.state_manager import StateManager
from agent.tracing import trace

logger = logging.getLogger("fitgen.workout_tool")

# ── Constants ────────────────────────────────────────────────────────

_DOMAIN = "workout"
_SYSTEM_PROMPT = WORKOUT_PROMPTS["few_shot"]
_VALID_INTENTS: list[str] = list(get_args(WorkoutIntent))


# ── Session Context ──────────────────────────────────────────────────

@dataclass
class WorkoutSessionContext:
    """All context needed for a single workout tool invocation."""

    state_manager: StateManager
    session_id: str
    user_email: str
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
        return cls(
            state_manager=sm,
            session_id=sm.context_id,
            user_email=sm.user_email,
            profile=dict(sm.user_profile or {}),
            workflow=wf,
            plan_text=wf.get("plan_text", ""),
            step_completed=wf.get("step_completed"),
            completed_steps=list(wf.get("completed_steps") or []),
            pending_question=wf.get("pending_question"),
        )


# ── Helpers ──────────────────────────────────────────────────────────

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
) -> str:
    """Multi-turn workout agent with create/modify/delete/sync workflow.

    IMPORTANT: The 'query' parameter must be the user's EXACT message,
    word-for-word. Do NOT paraphrase, expand, or add instructions.

    Returns JSON with assistant_message and state_updates.
    """
    raw_query = _get_raw_user_query(state)
    effective_query = raw_query or query

    logger.info("[WorkoutTool] query: %s", effective_query[:120])
    if raw_query != query:
        logger.debug(
            "[WorkoutTool] Overrode LLM tool arg (len=%d) with raw user "
            "message (len=%d)",
            len(query),
            len(raw_query),
        )

    return execute(effective_query, state)


def execute(query: str, state: dict[str, Any]) -> str:
    """Execution entrypoint — delegates to handle_multi_turn."""
    return handle_multi_turn(query=query, state=state)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Orchestrator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_INTENT_HANDLERS: dict[str, Any] = {}


def handle_multi_turn(*, query: str, state: dict[str, Any]) -> str:
    """Orchestrate the workout multi-turn workflow.

    Steps:
      1. Load session via StateManager
      2. Extract session context
      3. Classify intent via LLM → WorkoutIntent
      4. Dispatch to the matching intent handler
    """
    ctx = WorkoutSessionContext.from_state(state)

    logger.info(
        "[WorkoutFlow] session=%s step=%s profile_fields=%d plan_len=%d",
        ctx.session_id,
        ctx.step_completed,
        len(ctx.profile),
        len(ctx.plan_text),
    )

    # Classify intent via LLM (always, no shortcuts)
    try:
        classification = classify_intent(
            query,
            domain=_DOMAIN,
            valid_intents=_VALID_INTENTS,
            step_completed=ctx.step_completed,
            user_profile=ctx.profile,
            pending_question=ctx.pending_question,
            has_plan=bool(ctx.plan_text),
        )
    except Exception as exc:
        logger.error("[WorkoutFlow] Intent classification failed: %s", exc)
        return _build(
            "I had trouble understanding your request. Please try again.",
            ctx,
        )

    intent = classification["user_intent"]
    reason = classification.get("reason", "")
    logger.info("[WorkoutFlow] intent=%s reason=%s", intent, reason[:80])

    handler = _INTENT_HANDLERS.get(intent)
    if handler is None:
        logger.warning("[WorkoutFlow] No handler for intent=%s", intent)
        return _build(
            f"I can help you create, update, delete, or view your {_DOMAIN} "
            "plan. What would you like to do?",
            ctx,
        )

    try:
        return handler(query, ctx)
    except Exception as exc:
        logger.error(
            "[WorkoutFlow] Handler %s failed: %s", intent, exc, exc_info=True
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
    step = ctx.step_completed

    # ── Step A: Fresh create or re-entering ──
    if step is None or step in ("workout_confirmed", "workout_plan_generated",
                                 "updated_workout_plan"):
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
        plan_markdown = generate_plan(
            _DOMAIN, ctx.profile, query, ctx.system_prompt
        )

        _update_workflow(
            ctx,
            step_completed="workout_plan_generated",
            step_name="plan_generated",
            intent="create_workout",
            plan_text=plan_markdown,
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

    # TODO: Save user_profile in DB (skipped for now)

    _update_workflow(
        ctx,
        step_completed="workout_confirmed",
        step_name="workout_confirmed",
        intent="confirm_workout",
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
    plan_markdown = generate_plan(
        _DOMAIN, ctx.profile, query, ctx.system_prompt,
        existing_plan=ctx.plan_text,
    )

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
    """Handle get_workout intent — display current plan or prompt to create."""
    if ctx.plan_text:
        plan_display = ctx.plan_text

        summary_lines = [
            f"- {k.replace('_', ' ').title()}: {v}"
            for k, v in ctx.profile.items()
            if v not in (None, "")
        ]
        summary = "\n".join(summary_lines) if summary_lines else "- (No profile data)"

        return _build(
            f"Here's your current workout plan:\n\n"
            f"**Your Profile:**\n{summary}\n\n"
            f"**Your Plan:**\n{plan_display}",
            ctx,
        )

    return _build(
        "No workout plan found for this session. Would you like me to "
        "create one? Just say **create a workout plan**!",
        ctx,
    )


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
        has_plan=bool(ctx.plan_text),
    )

    if classification["user_intent"] == "confirm_workout":
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


def _handle_sync_to_google_calendar(
    query: str, ctx: WorkoutSessionContext,
) -> str:
    """Handle sync_workout_to_google_calendar intent."""
    if not ctx.plan_text:
        return _build("You don't have a workout plan to sync yet. Create one first!", ctx)

    try:
        from agent.tools.calendar_integration import (
            get_authorization_url, save_oauth_context,
        )

        auth_url, oauth_state = get_authorization_url()
        save_oauth_context(
            plan_text=ctx.plan_text, domain=_DOMAIN,
            profile=ctx.profile, sync_target="calendar",
        )

        _update_workflow(
            ctx,
            step_completed="workout_plan_synced_to_google_calendar",
            step_name="calendar_sync_started",
            intent="sync_workout_to_google_calendar",
            oauth_state=oauth_state,
        )

        return _build(
            "🔗 **Ready to connect Google Calendar!**\n\n"
            "Click the **\"📅 Connect Google Calendar\"** button in the "
            "sidebar to sign in with Google and sync your workouts.\n\n"
            f"Or open this link directly: [Authorize FITGEN.AI]({auth_url})",
            ctx,
            extra={"calendar_sync_requested": True, "calendar_auth_url": auth_url},
        )

    except Exception as exc:
        logger.error("[WorkoutFlow] Calendar sync failed: %s", exc)
        return _build(
            "Google Calendar integration is not configured. "
            "Please check your .env file.",
            ctx,
        )


def _handle_sync_to_google_fit(
    query: str, ctx: WorkoutSessionContext,
) -> str:
    """Handle sync_workout_to_google_fit intent."""
    if not ctx.plan_text:
        return _build("You don't have a workout plan to sync yet. Create one first!", ctx)

    try:
        from agent.tools.calendar_integration import (
            get_authorization_url, save_oauth_context,
        )

        auth_url, _ = get_authorization_url()
        save_oauth_context(
            plan_text=ctx.plan_text, domain=_DOMAIN,
            profile=ctx.profile, sync_target="google_fit",
        )

        _update_workflow(
            ctx,
            step_completed="workout_plan_synced_to_google_fit",
            step_name="google_fit_sync_started",
            intent="sync_workout_to_google_fit",
        )

        return _build(
            "💪 **Ready to connect Google Fit!**\n\n"
            "Click the **\"💪 Sync to Google Fit\"** button in the sidebar "
            "to sign in with Google and sync your plan data.\n\n"
            f"Or open this link directly: [Authorize Google Fit]({auth_url})",
            ctx,
            extra={"google_fit_sync_requested": True},
        )

    except Exception as exc:
        logger.error("[WorkoutFlow] Google Fit sync failed: %s", exc)
        return _build(
            "Google Fit integration is not configured. "
            "Please check your .env file.",
            ctx,
        )


def _handle_general_workout_query(
    query: str, ctx: WorkoutSessionContext,
) -> str:
    """Handle general_workout_query intent — contextual Q&A."""
    answer = answer_followup_question(
        _DOMAIN, query, ctx.profile, ctx.plan_text, ctx.system_prompt,
    )
    return _build(answer, ctx)


# ── Register intent handlers ─────────────────────────────────────────

_INTENT_HANDLERS.update({
    "create_workout": _handle_create_workout,
    "update_workout": _handle_update_workout,
    "get_workout": _handle_get_workout,
    "delete_workout": _handle_delete_workout,
    "confirm_workout": _handle_confirm_workout,
    "sync_workout_to_google_calendar": _handle_sync_to_google_calendar,
    "sync_workout_to_google_fit": _handle_sync_to_google_fit,
    "general_workout_query": _handle_general_workout_query,
})
