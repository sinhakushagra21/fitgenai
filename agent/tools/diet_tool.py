"""
agent/tools/diet_tool.py
────────────────────────
FITGEN.AI Diet & Nutrition Specialist Tool.

Self-contained multi-turn workflow for diet plan creation, modification,
retrieval, deletion, calendar/fit sync, and general nutrition queries.

Flow: diet_tool() → execute() → handle_multi_turn()

Each user message is classified into a DietIntent by the LLM, then
dispatched to the corresponding handler. Session state is tracked via
step_completed markers and persisted through StateManager.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Annotated, Any, get_args

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from agent.prompts.diet_prompts import DIET_PROMPTS
from agent.shared.llm_helpers import (
    answer_followup_question,
    answer_plan_question,
    classify_intent,
    extract_profile_updates,
    extract_profile_updates_with_fallback,
    generate_plan,
    generate_plan_name,
)
from agent.shared.profile_utils import (
    build_profile_bulk_question,
    build_profile_confirmation,
    missing_profile_fields,
    validate_profile_field,
)
from agent.shared.response_builder import append_completed_step, build_response
from agent.shared.types import (
    DIET_ALL_FIELDS,
    DIET_REQUIRED_FIELDS,
    DietIntent,
)
from agent.db.repositories.diet_plan_repo import DietPlanRepository
from agent.db.repositories.user_repo import UserRepository
from agent.error_utils import handle_exception
from agent.shared.plan_data import extract_plan_structured_data
from agent.state_manager import StateManager
from agent.tracing import log_event, trace

logger = logging.getLogger("fitgen.diet_tool")

# ── Constants ────────────────────────────────────────────────────────

_DOMAIN = "diet"
_SYSTEM_PROMPT = DIET_PROMPTS["few_shot"]
_VALID_INTENTS: list[str] = list(get_args(DietIntent))

# Steps where a draft plan exists and can still be updated/confirmed.
# Once confirmed or synced, the plan is in MongoDB — no longer a "draft".
_DRAFT_PLAN_STEPS = frozenset({
    "diet_plan_generated",
    "updated_diet_plan",
})

# Steps that indicate the workflow is fully complete.
_TERMINAL_STEPS = frozenset({
    "diet_plan_synced_to_google_calendar",
    "diet_plan_synced_to_google_fit",
})

# Intents that belong to the post-confirm sync flow.
_SYNC_INTENTS = frozenset({
    "sync_diet_to_google_calendar",
    "sync_diet_to_google_fit",
})


# ── Session Context ──────────────────────────────────────────────────

@dataclass
class DietSessionContext:
    """All context needed for a single diet tool invocation.

    Built once at the start of ``handle_multi_turn`` and passed to every
    intent handler to avoid redundant state lookups.
    """

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
    def from_state(cls, state: dict[str, Any]) -> DietSessionContext:
        """Build context from LangGraph state via StateManager."""
        sm = StateManager.from_state(state)
        wf = dict(sm.workflow or {})

        # ── Auto-reset truly completed / cross-domain workflows ─────
        # After a sync kicks off (`*_synced_to_*`) the plan is safely in
        # MongoDB and OAuth hand-off has started — that's a terminal
        # state, so we clear the in-memory workflow for the next turn.
        #
        # NOTE: `diet_confirmed` is NOT a reset trigger. The user is
        # still being prompted for a sync decision, so we MUST preserve
        # the full workflow (step_completed, domain, plan_id, intent,
        # pending_question) so classify_intent and the handlers can
        # route the follow-up "sync / both / skip" reply correctly.
        #
        # Also reset if the workflow belongs to a different domain
        # (e.g. workout workflow leftover when diet tool is called).
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
                "[DietFlow] Resetting workflow in from_state "
                "(step=%s, domain=%s, cross=%s, plan_id=%s)",
                _step, _wf_domain, _is_cross_domain, _plan_id,
            )
            wf = {"plan_id": _plan_id} if _plan_id else {}

        # Merge MongoDB stored profile (baseline) with session profile (overrides)
        profile: dict[str, Any] = {}
        if sm.user_email:
            try:
                profile = UserRepository.get_merged_profile(sm.user_email, domain="diet")
            except Exception as exc:  # noqa: BLE001
                handle_exception(
                    exc,
                    module="diet_tool",
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
    ctx: DietSessionContext,
    *,
    extra: dict[str, Any] | None = None,
) -> str:
    """Convenience wrapper around build_response using DietSessionContext."""
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
    active workflow belonging to the OTHER domain.

    * Rewrites ``state_updates.workflow`` and ``state_updates.user_profile``
      in the returned JSON to the caller's original snapshots.
    * Re-persists the original context-state to MongoDB (undoing whatever
      the side-query handler wrote during _build()).
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
            "[DietFlow] Side-query: restored caller state "
            "(workflow_domain=%s, step=%s)",
            (orig_workflow or {}).get("domain"),
            (orig_workflow or {}).get("step_completed"),
        )
    except Exception as exc:  # noqa: BLE001
        handle_exception(
            exc, module="diet_tool",
            context="restore side-query state",
            level="WARNING",
            extra={"context_id": str(context_id)},
        )

    return _json.dumps(payload)


def _update_workflow(
    ctx: DietSessionContext,
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
    """Run sanity checks on extracted profile fields.

    Returns the cleaned updates dict and a list of error messages.
    """
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
@trace(name="Diet Tool", run_type="tool", tags=["diet", "tool"])
def diet_tool(
    query: str,
    state: Annotated[dict[str, Any], InjectedState],
    side_query: bool = False,
) -> str:
    """Diet & nutrition specialist. Handles meal plans, macros, calorie \
targets, food recommendations, allergies/preferences, snacking habits, \
sync to Google Calendar / Google Fit, and general nutrition Q&A.

    Use this tool for anything food-, meal-, nutrition-, or calorie-related,
    including creating, updating, retrieving, confirming, or deleting a
    diet plan, and for on-topic follow-up nutrition questions.

    The ``query`` MUST be the user's EXACT message, verbatim.
    Returns JSON with assistant_message and state_updates.
    """
    raw_query = _get_raw_user_query(state)
    effective_query = raw_query or query

    logger.info(
        "[DietTool] query: %s  side_query=%s",
        effective_query[:120], side_query,
    )

    return execute(effective_query, state, side_query=side_query)


def execute(query: str, state: dict[str, Any], *, side_query: bool = False) -> str:
    """Execution entrypoint — delegates to handle_multi_turn."""
    return handle_multi_turn(query=query, state=state, side_query=side_query)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Orchestrator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Intent → handler dispatch table (populated after handler definitions)
_INTENT_HANDLERS: dict[str, Any] = {}


def handle_multi_turn(
    *,
    query: str,
    state: dict[str, Any],
    side_query: bool = False,
) -> str:
    """Orchestrate the diet multi-turn workflow.

    When ``side_query=True`` the tool is being called from the OTHER
    domain's active workflow for a one-shot nutrition question. In that
    case we:
      * force intent = ``general_diet_query`` (no classifier call),
      * after the handler runs, RESTORE the caller's original workflow
        and profile in both the returned ``state_updates`` AND the
        persisted context-state document — so the active workout flow
        is not corrupted by our intermediate writes.
    """
    # Snapshot the true caller state BEFORE from_state rewrites anything.
    _orig_workflow = dict(state.get("workflow") or {}) if side_query else None
    _orig_profile = dict(state.get("user_profile") or {}) if side_query else None
    _orig_calendar = bool(state.get("calendar_sync_requested", False))

    # Step 1-2: Load session context
    ctx = DietSessionContext.from_state(state)

    logger.info(
        "[DietFlow] session=%s step=%s profile_fields=%d plan_len=%d",
        ctx.session_id,
        ctx.step_completed,
        len(ctx.profile),
        len(ctx.plan_text),
    )

    # Side-query fast-path: skip classification, force general_diet_query,
    # run the handler, then RESTORE the caller's original workflow/profile
    # so the cross-domain active workflow is not clobbered.
    if side_query:
        logger.info("[DietFlow] Side-query mode — forcing general_diet_query")
        try:
            result = _handle_general_diet_query(query, ctx)
        except Exception as exc:  # noqa: BLE001
            handle_exception(
                exc, module="diet_tool", context="handler:general_diet_query (side)",
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

    # Step 3: Classify intent via LLM (always, no shortcuts)
    #
    # has_plan tells the classifier whether *any* plan exists for this
    # user — draft (in session) OR confirmed (in MongoDB). This is what
    # RULE 4 of the classifier prompt checks to decide if sync requests
    # are legal.
    #   • Draft: ctx.plan_text set AND step ∈ _DRAFT_PLAN_STEPS
    #   • Confirmed: step == "diet_confirmed" (plan lives in Mongo;
    #     ctx.plan_text is cleared because sync handlers re-resolve via
    #     plan_id). Without this, "sync to calendar" at step
    #     `diet_confirmed` would be mis-classified as general_diet_query.
    _has_draft_plan = (
        (bool(ctx.plan_text) and ctx.step_completed in _DRAFT_PLAN_STEPS)
        or ctx.step_completed == "diet_confirmed"
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
            module="diet_tool",
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
    logger.info("Classified intent: %s (reason: %s)", intent, reason[:80])
    log_event(
        "diet.intent_classified",
        module="diet_tool",
        intent=intent,
        step=ctx.step_completed,
    )

    # Step 4: Dispatch to handler
    handler = _INTENT_HANDLERS.get(intent)
    if handler is None:
        logger.warning("No handler registered for intent: %s", intent)
        log_event(
            "diet.missing_handler",
            level="WARNING",
            module="diet_tool",
            intent=intent,
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
            module="diet_tool",
            context=f"handler:{intent}",
            extra={
                "session_id": ctx.session_id,
                "step": ctx.step_completed,
                "intent": intent,
            },
        )
        return _build(
            "I had trouble processing your request. Please try again.",
            ctx,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Intent Handlers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _handle_create_diet(query: str, ctx: DietSessionContext) -> str:
    """Handle create_diet intent — multi-step profile collection + plan generation.

    Steps based on ctx.step_completed:
      None            → prompt for profile data (show form)
      prompted_for_*  → extract/map fields, re-prompt if missing required
      user_profile_*  → generate plan, return for confirmation
    """
    # ── Cross-domain reset: clear stale workflow from a different tool ──
    _wf_domain = ctx.workflow.get("domain")
    if _wf_domain and _wf_domain != _DOMAIN:
        logger.info("[DietFlow] Resetting stale %s workflow for new diet create", _wf_domain)
        ctx.workflow = {}
        ctx.plan_text = ""
        ctx.step_completed = None
        ctx.completed_steps = []
        ctx.pending_question = None

    step = ctx.step_completed

    # ── Step A: Fresh create or re-entering create flow ──
    if step is None or step in ("diet_confirmed", "diet_plan_generated",
                                 "updated_diet_plan",
                                 "diet_plan_synced_to_google_calendar",
                                 "diet_plan_synced_to_google_fit"):
        # Extract any profile data from the initial query
        updates = extract_profile_updates_with_fallback(
            query, DIET_REQUIRED_FIELDS, DIET_ALL_FIELDS
        )
        cleaned, validation_errors = _validate_extracted_profile(updates)
        ctx.profile.update(cleaned)

        missing = missing_profile_fields(ctx.profile, DIET_REQUIRED_FIELDS)
        if missing:
            _update_workflow(
                ctx,
                step_completed="prompted_for_user_profile_data",
                step_name="profile_collection_started",
                intent="create_diet",
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

        # All required fields present from query → skip to confirmation
        # Show ALL fields the user provided, not just required ones
        confirm_msg = build_profile_confirmation(ctx.profile, DIET_ALL_FIELDS)
        _update_workflow(
            ctx,
            step_completed="user_profile_mapped",
            step_name="profile_mapped",
            intent="create_diet",
            pending_question=confirm_msg,
        )
        return _build(confirm_msg, ctx)

    # ── Step B: User is providing profile data after form/prompt ──
    if step == "prompted_for_user_profile_data":
        expected = ctx.workflow.get("missing_fields") or DIET_REQUIRED_FIELDS
        updates = extract_profile_updates_with_fallback(
            query, expected, DIET_ALL_FIELDS
        )
        cleaned, validation_errors = _validate_extracted_profile(updates)
        ctx.profile.update(cleaned)

        missing = missing_profile_fields(ctx.profile, DIET_REQUIRED_FIELDS)
        if missing:
            # Still missing required fields — re-prompt
            _update_workflow(
                ctx,
                step_completed="prompted_for_user_profile_data",
                step_name="profile_reprompted",
                intent="create_diet",
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

        # All required fields collected → show confirmation
        # Show ALL fields the user provided, not just required ones
        confirm_msg = build_profile_confirmation(ctx.profile, DIET_ALL_FIELDS)
        _update_workflow(
            ctx,
            step_completed="user_profile_mapped",
            step_name="profile_mapped",
            intent="create_diet",
            pending_question=confirm_msg,
        )
        return _build(confirm_msg, ctx)

    # ── Step C: Profile confirmed → generate plan ──
    if step == "user_profile_mapped":
        # The user may say "yes" OR "yes but change X to Y".
        # Extract any last-minute profile tweaks before generating.
        _tweaks = extract_profile_updates(query, DIET_ALL_FIELDS)
        if _tweaks:
            logger.info("[DietFlow] Applying last-minute profile tweaks: %s", _tweaks)
            ctx.profile.update(_tweaks)

        raw_plan = generate_plan(
            _DOMAIN, ctx.profile, query, ctx.system_prompt
        )

        # Extract structured data (macros, hydration) and strip from display
        plan_markdown, structured_data = extract_plan_structured_data(raw_plan)

        # NOTE: Plan is NOT saved to MongoDB here — only on confirm.
        # This keeps the DB clean: only confirmed plans are persisted.

        _update_workflow(
            ctx,
            step_completed="diet_plan_generated",
            step_name="plan_generated",
            intent="create_diet",
            plan_text=plan_markdown,
            structured_data=structured_data,
            pending_question=(
                "Please review your plan and tell me if you want any "
                "changes, or confirm to proceed."
            ),
        )

        return _build(
            f"Here's your personalized diet plan:\n\n{plan_markdown}\n\n"
            "Please review. Reply **yes** to confirm, or tell me what "
            "you'd like to change.",
            ctx,
            extra={"plan_text": plan_markdown},
        )

    # Fallback for unexpected step states in create flow
    logger.warning("[DietFlow] create_diet: unexpected step=%s", step)
    return _build(
        "Let's start fresh. What would you like me to create?",
        ctx,
    )


def _handle_confirm_diet(query: str, ctx: DietSessionContext) -> str:
    """Handle confirm_diet intent — save profile, offer sync options.

    Guard: Only actually confirm if a plan has been generated.
    If the user says "yes" during profile collection/mapping, delegate
    back to _handle_create_diet to continue the create flow (generate plan).
    """
    # ── Guard: plan must exist before we can confirm it ──
    if ctx.step_completed in (
        None,
        "prompted_for_user_profile_data",
        "user_profile_mapped",
    ):
        logger.info(
            "[DietFlow] confirm_diet at step=%s → delegating to create_diet "
            "(plan not yet generated)",
            ctx.step_completed,
        )
        return _handle_create_diet(query, ctx)

    if ctx.step_completed not in ("diet_plan_generated", "updated_diet_plan"):
        return _build(
            "There's no plan to confirm yet. Would you like me to create "
            "a diet plan?",
            ctx,
        )

    # ── Persist to MongoDB only on confirm ──
    _plan_id = None
    if ctx.user_email:
        # 1. Ensure user exists in MongoDB
        user_doc = UserRepository.find_or_create(ctx.user_email)
        _user_id = str(user_doc["_id"])

        # 2. Save / update profile (base + diet sub-docs)
        _base, _diet = _split_profile_fields(ctx.profile, domain="diet")
        UserRepository.update_profile(ctx.user_email, base=_base, diet=_diet)
        logger.info("[DietFlow] Profile saved to users collection for %s", ctx.user_email)

        # 3. Generate a catchy plan name via LLM
        plan_text = ctx.workflow.get("plan_text", ctx.plan_text)
        _structured = ctx.workflow.get("structured_data", {})
        _plan_name = ""
        if plan_text:
            try:
                _plan_name = generate_plan_name(_DOMAIN, ctx.profile, plan_text)
            except Exception as exc:
                logger.warning("[DietFlow] Plan naming failed: %s", exc)

        # 4. Create confirmed plan in diet_plans collection
        if plan_text:
            _plan_id = DietPlanRepository.create(
                user_id=_user_id,
                session_id=ctx.session_id,
                profile_snapshot=dict(ctx.profile),
                plan_markdown=plan_text,
                structured_data=_structured,
                status="confirmed",
                name=_plan_name,
            )
            logger.info("[DietFlow] Plan created & confirmed: plan_id=%s name=%r", _plan_id, _plan_name)
    else:
        logger.warning(
            "[DietFlow] confirm_diet: no user_email — skipping MongoDB save. "
            "Set FITGEN_USER_EMAIL in .env to enable persistence."
        )

    # ── Preserve workflow context for the sync follow-up ──────────────
    # We intentionally KEEP `workflow.domain="diet"` and
    # `step_completed="diet_confirmed"` so the router can deterministically
    # route the user's next reply ("calendar" / "both" / "skip" / …) to
    # this tool. Plan markdown + structured_data are cleared because the
    # plan now lives in MongoDB; sync handlers pull it via _resolve_plan_text.
    ctx.plan_text = ""
    ctx.completed_steps = []

    _update_workflow(
        ctx,
        step_completed="diet_confirmed",
        step_name="diet_confirmed",
        intent="confirm_diet",
        plan_id=str(_plan_id) if _plan_id else None,
        plan_text="",
        structured_data={},
        pending_question=(
            "Would you like to sync to Google Calendar, Google Fit, or both?"
        ),
    )

    return _build(
        "Your diet plan is confirmed! 🎉\n\n"
        "Would you like to sync it to:\n"
        "- **Google Calendar** (schedule meals as events)\n"
        "- **Google Fit** (log nutrition data)\n"
        "- **Both**\n\n"
        "Or just say **done** if you're all set!",
        ctx,
    )


def _wipe_sync_workflow(ctx: DietSessionContext) -> None:
    """Clear all workflow state after sync decision is terminal.

    Called from the three terminal branches:
      • sync_diet_to_google_calendar  (OAuth hand-off started)
      • sync_diet_to_google_fit       (OAuth hand-off started)
      • skip_sync_diet                (user declined sync)

    After this runs the next user message will go through the router's
    LLM classifier as a fresh conversation — so domain switches and new
    requests work cleanly.
    """
    ctx.workflow = {}
    ctx.plan_text = ""
    ctx.step_completed = None
    ctx.completed_steps = []
    ctx.pending_question = None


def _handle_skip_sync_diet(query: str, ctx: DietSessionContext) -> str:
    """User declined sync after confirming the plan — wipe workflow."""
    _wipe_sync_workflow(ctx)
    return _build(
        "All set! Your diet plan is confirmed. "
        "Let me know whenever you need anything else. 💪",
        ctx,
    )


def _handle_sync_to_both(query: str, ctx: DietSessionContext) -> str:
    """Sync the plan to BOTH Google Calendar and Google Fit.

    We chain the two existing handlers. Each handler kicks off its own
    OAuth hand-off and marks the plan in MongoDB. The final response
    combines both CTAs for the sidebar.
    """
    _plan = _resolve_plan_text(ctx)
    if not _plan:
        return _build(
            "You don't have a diet plan to sync yet. Create one first!",
            ctx,
        )

    try:
        from agent.tools.calendar_integration import (
            get_authorization_url,
            save_oauth_context,
        )

        auth_url, oauth_state = get_authorization_url()

        # One OAuth session authorises both scopes — save context with
        # sync_target="both" so the post-OAuth handler performs both syncs.
        save_oauth_context(
            plan_text=_plan,
            domain=_DOMAIN,
            profile=ctx.profile,
            sync_target="both",
        )

        _plan_id = ctx.workflow.get("plan_id")
        if _plan_id:
            DietPlanRepository.update_plan(
                _plan_id, calendar_synced=True, fit_synced=True,
            )

        _update_workflow(
            ctx,
            step_completed="diet_plan_synced_to_google_calendar",
            step_name="sync_both_started",
            intent="sync_diet_to_both",
            oauth_state=oauth_state,
            pending_question="Click the button in the sidebar to complete the sync.",
        )

        # Wipe AFTER _update_workflow has captured the terminal step —
        # the state_sync layer reads from ctx.workflow on this turn.
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
            module="diet_tool",
            context="sync to both (calendar + fit)",
            extra={"session_id": ctx.session_id},
        )
        return _build(
            "I'd love to sync to Google Calendar and Google Fit, but the "
            "integration is not configured. Please check GOOGLE_CLIENT_ID "
            "and GOOGLE_CLIENT_SECRET in your .env file.",
            ctx,
        )


def _handle_update_diet(query: str, ctx: DietSessionContext) -> str:
    """Handle update_diet intent — map changes, regenerate plan incrementally."""
    # If no existing plan → redirect to create
    if not ctx.plan_text:
        logger.info("[DietFlow] update_diet: no existing plan, redirecting to create")
        _update_workflow(
            ctx,
            step_completed="prompted_for_user_profile_data",
            step_name="update_no_plan_redirect",
            intent="create_diet",
            missing_fields=missing_profile_fields(ctx.profile, DIET_REQUIRED_FIELDS),
        )
        missing = missing_profile_fields(ctx.profile, DIET_REQUIRED_FIELDS)
        if missing:
            return _build(
                "I don't have an existing diet plan to update. "
                "Let's create one first!\n\n"
                + build_profile_bulk_question(missing),
                ctx,
            )
        # Profile is complete but no plan — go straight to generation
        ctx.step_completed = "user_profile_mapped"
        return _handle_create_diet(query, ctx)

    # Extract field changes from the update request
    updates = extract_profile_updates(query, DIET_ALL_FIELDS)
    cleaned, _ = _validate_extracted_profile(updates)
    ctx.profile.update(cleaned)

    # Generate updated plan (pass existing plan for incremental changes)
    plan_markdown = generate_plan(
        _DOMAIN, ctx.profile, query, ctx.system_prompt,
        existing_plan=ctx.plan_text,
    )

    # NOTE: Updated plan is NOT saved to MongoDB here — only on confirm.

    _update_workflow(
        ctx,
        step_completed="updated_diet_plan",
        step_name="plan_updated",
        intent="update_diet",
        plan_text=plan_markdown,
        pending_question=(
            "Please review the updated plan. Reply yes to confirm, "
            "or tell me what else to change."
        ),
    )

    return _build(
        f"Done — I've updated your diet plan:\n\n{plan_markdown}\n\n"
        "Please review. Reply **yes** to confirm, or tell me what else "
        "to change.",
        ctx,
        extra={"plan_text": plan_markdown},
    )


def _handle_get_diet(query: str, ctx: DietSessionContext) -> str:
    """Handle get_diet intent — fetch diet plan from DB, let LLM answer."""
    # Always fetch the diet plan from MongoDB — single source of truth.
    _plan = ""
    if ctx.user_id:
        latest = DietPlanRepository.find_latest_by_user(ctx.user_id)
        if latest and latest.get("plan_markdown"):
            _plan = latest["plan_markdown"]

    if not _plan:
        return _build(
            "No diet plan found. Would you like me to "
            "create one? Just say **create a diet plan**!",
            ctx,
        )

    # get is a one-shot operation — clear workflow so next call starts fresh
    ctx.workflow = {}
    ctx.plan_text = ""
    ctx.step_completed = None
    ctx.completed_steps = []
    ctx.pending_question = None

    answer = answer_plan_question(_DOMAIN, _plan, query)
    return _build(answer, ctx)


def _handle_delete_diet(query: str, ctx: DietSessionContext) -> str:
    """Handle delete_diet intent — two-step confirmation then clear."""
    if ctx.step_completed != "delete_confirmation_pending":
        # First ask for confirmation
        _update_workflow(
            ctx,
            step_completed="delete_confirmation_pending",
            step_name="delete_requested",
            intent="delete_diet",
            pending_question=(
                "Please confirm: do you want to delete your diet plan "
                "and profile? (yes/no)"
            ),
        )
        return _build(
            "Are you sure you want to delete your diet plan and profile? "
            "This cannot be undone. Reply **yes** to confirm or **no** "
            "to cancel.",
            ctx,
        )

    # User already asked — this is their confirmation response
    # Re-classify to see if they confirmed
    classification = classify_intent(
        query,
        domain=_DOMAIN,
        valid_intents=_VALID_INTENTS,
        step_completed=ctx.step_completed,
        user_profile=ctx.profile,
        pending_question=ctx.pending_question,
        has_plan=bool(ctx.plan_text) and ctx.step_completed in _DRAFT_PLAN_STEPS,
    )

    if classification["user_intent"] == "confirm_diet":
        # Archive plan in diet_plans collection
        _plan_id = ctx.workflow.get("plan_id")
        if _plan_id:
            DietPlanRepository.archive(_plan_id)
            logger.info("[DietFlow] Plan archived: plan_id=%s", _plan_id)

        # Clear everything
        ctx.profile = {}
        ctx.workflow = {}
        ctx.plan_text = ""
        _update_workflow(
            ctx,
            step_completed=None,
            step_name="delete_completed",
        )
        return _build(
            "Your diet plan and profile have been deleted. "
            "Let me know if you'd like to create a new one!",
            ctx,
        )

    # They didn't confirm — cancel
    _update_workflow(
        ctx,
        step_completed=ctx.workflow.get("_prev_step"),
        step_name="delete_cancelled",
    )
    return _build(
        "Deletion cancelled. Your diet plan is unchanged.",
        ctx,
    )


def _resolve_plan_text(ctx: DietSessionContext) -> str:
    """Get plan markdown — from ctx, workflow, or MongoDB (by plan_id)."""
    if ctx.plan_text:
        return ctx.plan_text
    # After confirm, from_state() clears plan_text. Fetch from MongoDB.
    _plan_id = ctx.workflow.get("plan_id")
    if _plan_id:
        doc = DietPlanRepository.find_by_id(_plan_id)
        if doc and doc.get("plan_markdown"):
            return doc["plan_markdown"]
    # Last resort: latest confirmed plan for user
    if ctx.user_id:
        doc = DietPlanRepository.find_latest_by_user(ctx.user_id, status="confirmed")
        if doc and doc.get("plan_markdown"):
            return doc["plan_markdown"]
    return ""


def _handle_sync_to_google_calendar(
    query: str,
    ctx: DietSessionContext,
) -> str:
    """Handle sync_diet_to_google_calendar intent."""
    _plan = _resolve_plan_text(ctx)
    if not _plan:
        return _build(
            "You don't have a diet plan to sync yet. Create one first!",
            ctx,
        )

    try:
        from agent.tools.calendar_integration import (
            get_authorization_url,
            save_oauth_context,
        )

        auth_url, oauth_state = get_authorization_url()

        save_oauth_context(
            plan_text=_plan,
            domain=_DOMAIN,
            profile=ctx.profile,
            sync_target="calendar",
        )

        # Mark plan as calendar-synced in MongoDB
        _plan_id = ctx.workflow.get("plan_id")
        if _plan_id:
            DietPlanRepository.update_plan(_plan_id, calendar_synced=True)

        _update_workflow(
            ctx,
            step_completed="diet_plan_synced_to_google_calendar",
            step_name="calendar_sync_started",
            intent="sync_diet_to_google_calendar",
            oauth_state=oauth_state,
            pending_question="Click the button in the sidebar to complete the sync.",
        )

        # Terminal branch — wipe residual workflow so the next user message
        # starts fresh at the router's LLM classifier.
        _wipe_sync_workflow(ctx)

        return _build(
            "🔗 **Ready to connect Google Calendar!**\n\n"
            "Click the **\"📅 Connect Google Calendar\"** button in the "
            "sidebar to sign in with Google and sync your plan.\n\n"
            f"Or open this link directly: [Authorize FITGEN.AI]({auth_url})",
            ctx,
            extra={"calendar_sync_requested": True, "calendar_auth_url": auth_url},
        )

    except Exception as exc:  # noqa: BLE001
        handle_exception(
            exc,
            module="diet_tool",
            context="google calendar sync",
            extra={"session_id": ctx.session_id},
        )
        return _build(
            "I'd love to sync to Google Calendar, but the integration "
            "is not configured yet. Please check that GOOGLE_CLIENT_ID "
            "and GOOGLE_CLIENT_SECRET are set in your .env file.",
            ctx,
        )


def _handle_sync_to_google_fit(
    query: str,
    ctx: DietSessionContext,
) -> str:
    """Handle sync_diet_to_google_fit intent."""
    _plan = _resolve_plan_text(ctx)
    if not _plan:
        return _build(
            "You don't have a diet plan to sync yet. Create one first!",
            ctx,
        )

    try:
        from agent.tools.calendar_integration import (
            get_authorization_url,
            save_oauth_context,
        )

        auth_url, _oauth_state = get_authorization_url()

        save_oauth_context(
            plan_text=_plan,
            domain=_DOMAIN,
            profile=ctx.profile,
            sync_target="google_fit",
        )

        # Mark plan as fit-synced in MongoDB
        _plan_id = ctx.workflow.get("plan_id")
        if _plan_id:
            DietPlanRepository.update_plan(_plan_id, fit_synced=True)

        _update_workflow(
            ctx,
            step_completed="diet_plan_synced_to_google_fit",
            step_name="google_fit_sync_started",
            intent="sync_diet_to_google_fit",
            pending_question="Click the button in the sidebar to complete the sync.",
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
            module="diet_tool",
            context="google fit sync",
            extra={"session_id": ctx.session_id},
        )
        return _build(
            "I'd love to sync to Google Fit, but the integration "
            "is not configured yet. Please check that GOOGLE_CLIENT_ID "
            "and GOOGLE_CLIENT_SECRET are set in your .env file.",
            ctx,
        )


def _handle_general_diet_query(
    query: str,
    ctx: DietSessionContext,
) -> str:
    """Handle general_diet_query intent — contextual Q&A.

    Feeds the user's profile and current diet plan to the LLM as context,
    then answers the question without regenerating the plan.
    """
    # Use diet plan from MongoDB as context (avoids cross-domain mix-up)
    _plan = ""
    if ctx.user_id:
        latest = DietPlanRepository.find_latest_by_user(ctx.user_id)
        if latest and latest.get("plan_markdown"):
            _plan = latest["plan_markdown"]
    if not _plan and ctx.workflow.get("domain") == "diet":
        _plan = ctx.plan_text  # fallback to session only if same domain

    answer = answer_followup_question(
        _DOMAIN, query, ctx.profile, _plan, ctx.system_prompt,
    )

    # Don't change step_completed — stay in current flow position
    return _build(answer, ctx)


# ── Register intent handlers ─────────────────────────────────────────

_INTENT_HANDLERS.update({
    "create_diet": _handle_create_diet,
    "update_diet": _handle_update_diet,
    "get_diet": _handle_get_diet,
    "delete_diet": _handle_delete_diet,
    "confirm_diet": _handle_confirm_diet,
    "sync_diet_to_google_calendar": _handle_sync_to_google_calendar,
    "sync_diet_to_google_fit": _handle_sync_to_google_fit,
    "sync_diet_to_both": _handle_sync_to_both,
    "skip_sync_diet": _handle_skip_sync_diet,
    "general_diet_query": _handle_general_diet_query,
})
