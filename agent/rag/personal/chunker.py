"""
agent/rag/personal/chunker.py
─────────────────────────────
Section-aware markdown chunker for diet and workout plans.

Design:
  * Split on markdown heading levels (# / ## / ###).
  * Derive a typed ``section_type`` from the heading text.
  * Extract ``day_of_week`` for meal_day / workout_day sections.
  * Keep each chunk self-contained: embed text includes a breadcrumb
    preamble + the heading + the section body + tag annotations so the
    resulting vector captures "this is Tuesday's meals" and not just
    "a list of foods".
  * Hard-cap very long sections at ~1200 tokens with graceful sub-splits.

Pure functions, no network, no DB — safe to unit-test cheaply.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from agent.rag.personal.schema import PlanChunk, SectionType

logger = logging.getLogger("fitgen.rag.personal.chunker")

# ── Token estimator ──────────────────────────────────────────────
# Rough ~4 chars/token heuristic — good enough for chunk-sizing logic;
# we don't count tokens for billing here.

def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


_MAX_CHUNK_TOKENS = 1200
_SOFT_CHUNK_TOKENS = 900


# ── Heading parsing ──────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

_DAYS = ("monday", "tuesday", "wednesday", "thursday",
         "friday", "saturday", "sunday")
_DAY_RE = re.compile(
    r"\b(" + "|".join(_DAYS) + r")\b", re.IGNORECASE,
)


def _strip_emoji(text: str) -> str:
    """Drop non-ASCII glyphs (emojis, fancy punctuation) for tag matching."""
    return "".join(ch for ch in text if ord(ch) < 128)


def _extract_day_of_week(heading: str) -> str | None:
    m = _DAY_RE.search(heading)
    if m:
        return m.group(1).lower()
    return None


# ── Section-type classification ──────────────────────────────────

# Order matters: more specific phrases first. These are matched against
# the lower-cased, ASCII-only heading. The defaults below are tuned for
# the current DIET_PROMPTS / WORKOUT_PROMPTS output structure.

_DIET_PATTERNS: list[tuple[str, str]] = [
    (r"\bcalori(e|c)\s*(calculat|target|intake|estimat)", SectionType.CALORIE_CALC.value),
    (r"\btdee\b|\bbmr\b", SectionType.CALORIE_CALC.value),
    (r"\bmacro", SectionType.MACROS.value),
    (r"\bmeal\s*plan\b|7[-\s]?day", SectionType.MEAL_DAY.value),
    (r"\bsnack\s*swap", SectionType.SNACK_SWAPS.value),
    (r"\bpersonal.*rule|fat\s*loss\s*rule|lifestyle\s*rule", SectionType.RULES.value),
    (r"\btimeline|\bprojection|\brealistic\b", SectionType.TIMELINE.value),
    (r"\bhydrat|water\s*target", SectionType.HYDRATION.value),
    (r"\bsupplement", SectionType.SUPPLEMENTS.value),
]

_WORKOUT_PATTERNS: list[tuple[str, str]] = [
    (r"\bsplit\b|\boverview\b|\bweekly\s*structure", SectionType.SPLIT_OVERVIEW.value),
    (r"\bwarm[-\s]?up\b|\bmobility\s*prep", SectionType.WARMUP.value),
    (r"\bmain\s*lift|\bcompound|\bstrength\s*work", SectionType.MAIN_LIFTS.value),
    (r"\baccessor(y|ies)|\bisolation", SectionType.ACCESSORIES.value),
    (r"\bcardio|\bconditioning|\bliss\b|\bhiit\b", SectionType.CARDIO.value),
    (r"\bmobility|\bstretch|\bflexib", SectionType.MOBILITY.value),
    (r"\bprogression|\bweek[-\s]?by[-\s]?week|\bdeload", SectionType.PROGRESSION.value),
]


def _classify_section(heading: str, *, domain: str) -> str:
    ascii_h = _strip_emoji(heading).lower()

    if domain == "diet" and _DAY_RE.search(ascii_h):
        # "Monday – ..." / "Tuesday ..." — it's a meal_day section
        return SectionType.MEAL_DAY.value
    if domain == "workout" and (_DAY_RE.search(ascii_h)
                                or re.search(r"\bday\s*\d+", ascii_h)):
        return SectionType.WORKOUT_DAY.value

    patterns = _DIET_PATTERNS if domain == "diet" else _WORKOUT_PATTERNS
    for pat, tag in patterns:
        if re.search(pat, ascii_h):
            return tag

    return SectionType.NOTES.value


# ── Splitting ────────────────────────────────────────────────────

def _split_sections(markdown: str) -> list[tuple[str, str, int]]:
    """Split markdown into ``(heading, body, level)`` tuples.

    A section starts at a heading line and ends just before the next
    heading of equal or higher level (smaller ``#`` count).

    Fenced code blocks are protected so ``#`` inside them doesn't
    accidentally become a heading.
    """
    # Mask code fences so ``#`` inside them is ignored.
    fences: list[str] = []

    def _mask(m: re.Match[str]) -> str:
        fences.append(m.group(0))
        return f"\x00FENCE{len(fences) - 1}\x00"

    masked = _FENCE_RE.sub(_mask, markdown)

    matches = list(_HEADING_RE.finditer(masked))
    if not matches:
        return [("", markdown.strip(), 1)]

    out: list[tuple[str, str, int]] = []

    # Content before the first heading is a synthetic intro section.
    first_start = matches[0].start()
    if first_start > 0:
        intro = masked[:first_start].strip()
        if intro:
            out.append(("Overview", intro, 2))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(masked)
        body = masked[body_start:body_end].strip()

        # Restore fences inside the body.
        def _unmask(s: str) -> str:
            def _repl(mm: re.Match[str]) -> str:
                return fences[int(mm.group(1))]
            return re.sub(r"\x00FENCE(\d+)\x00", _repl, s)

        out.append((heading, _unmask(body), level))

    return out


def _split_oversized(
    heading: str, body: str, max_tokens: int = _MAX_CHUNK_TOKENS,
) -> list[tuple[str, str]]:
    """Split an oversized section into sub-chunks on sub-headings, then
    on blank-line boundaries as a fallback. Returns ``(sub_heading, body)``
    pairs; ``sub_heading`` is blank for whole-paragraph fallbacks.
    """
    if _approx_tokens(body) <= max_tokens:
        return [("", body)]

    # 1. Try to split on ### sub-headings inside this section.
    sub_heads = list(re.finditer(r"^###\s+(.*?)$", body, re.MULTILINE))
    if sub_heads:
        chunks: list[tuple[str, str]] = []
        for j, m in enumerate(sub_heads):
            sub_heading = m.group(1).strip()
            start = m.end()
            end = sub_heads[j + 1].start() if j + 1 < len(sub_heads) else len(body)
            chunks.append((sub_heading, body[start:end].strip()))
        return chunks

    # 2. Fallback: paragraph packing under soft cap.
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks_p: list[tuple[str, str]] = []
    buf: list[str] = []
    buf_tokens = 0
    for p in paragraphs:
        t = _approx_tokens(p)
        if buf and buf_tokens + t > _SOFT_CHUNK_TOKENS:
            chunks_p.append(("", "\n\n".join(buf)))
            buf, buf_tokens = [], 0
        buf.append(p)
        buf_tokens += t
    if buf:
        chunks_p.append(("", "\n\n".join(buf)))
    return chunks_p


# ── Public API ───────────────────────────────────────────────────

def _build_preamble(plan_type: str, top_heading: str, sub_heading: str) -> str:
    root = "Diet plan" if plan_type == "diet" else "Workout plan"
    parts = [root]
    if top_heading and top_heading != "Overview":
        parts.append(top_heading)
    if sub_heading:
        parts.append(sub_heading)
    return " › ".join(parts)


def _build_embedded_text(
    *,
    preamble: str,
    heading: str,
    body: str,
    section_type: str,
    day_of_week: str | None,
    plan_type: str,
) -> str:
    tag_bits = [f"plan_type={plan_type}", f"section={section_type}"]
    if day_of_week:
        tag_bits.append(f"day={day_of_week}")
    tags = ", ".join(tag_bits)
    return f"{preamble}\n{heading}\n\n{body}\n\n[Tags: {tags}]"


def _content_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def _chunk(
    *,
    markdown: str,
    plan_type: str,
    user_id: str,
    plan_id: str,
    plan_version: int,
    plan_status: str,
    profile_snapshot: dict[str, Any],
) -> list[PlanChunk]:
    if not markdown or not markdown.strip():
        return []

    # Shrink the profile snapshot we store per chunk (keep retrieval
    # payload small).
    keep = ("goal", "goal_weight", "diet_preference", "allergies",
            "experience_level", "training_days_per_week")
    digest = {k: profile_snapshot[k] for k in keep if k in profile_snapshot}

    sections = _split_sections(markdown)
    out: list[PlanChunk] = []

    # Track an active day-of-week context so deeper sub-sections inherit
    # it (e.g. "## Monday — Push" → "### Main Lifts" should still be
    # tagged day=monday even though the inner H3 heading doesn't repeat
    # the day).
    inherited_day: str | None = None
    inherited_level: int = 0

    for heading, body, level in sections:
        own_day = _extract_day_of_week(heading)
        if own_day:
            inherited_day = own_day
            inherited_level = level
        elif inherited_level and level <= inherited_level:
            # Left the scope of the day-bearing ancestor.
            inherited_day = None
            inherited_level = 0

        if not body.strip():
            continue
        section_type = _classify_section(heading, domain=plan_type)
        day_of_week = own_day or inherited_day

        for sub_heading, sub_body in _split_oversized(heading, body):
            final_heading = (
                f"{heading} — {sub_heading}" if sub_heading else heading
            )
            # Refresh day_of_week from the sub-heading if the parent
            # section itself didn't carry one (e.g. "7-Day Meal Plan"
            # with ### Monday inside).
            effective_day = day_of_week or _extract_day_of_week(sub_heading)
            effective_type = section_type
            if effective_type == SectionType.NOTES.value and effective_day:
                effective_type = (
                    SectionType.MEAL_DAY.value if plan_type == "diet"
                    else SectionType.WORKOUT_DAY.value
                )

            preamble = _build_preamble(plan_type, heading, sub_heading)
            chunk_text = sub_body.strip()
            embedded_text = _build_embedded_text(
                preamble=preamble,
                heading=final_heading,
                body=chunk_text,
                section_type=effective_type,
                day_of_week=effective_day,
                plan_type=plan_type,
            )
            out.append(
                PlanChunk(
                    user_id=user_id,
                    plan_id=plan_id,
                    plan_type=plan_type,
                    plan_status=plan_status,
                    plan_version=plan_version,
                    section_type=effective_type,
                    day_of_week=effective_day,
                    heading=final_heading,
                    preamble=preamble,
                    chunk_text=chunk_text,
                    embedded_text=embedded_text,
                    chunk_tokens=_approx_tokens(embedded_text),
                    source_content_hash=_content_hash(
                        plan_type, str(plan_id), str(plan_version),
                        effective_type, effective_day or "",
                        final_heading, chunk_text,
                    ),
                    profile_snapshot_digest=digest,
                )
            )

    logger.info(
        "[chunker] plan_type=%s plan_id=%s sections=%d chunks=%d",
        plan_type, plan_id, len(sections), len(out),
    )
    return out


def chunk_diet_plan(
    markdown: str,
    *,
    user_id: str,
    plan_id: str,
    plan_version: int = 1,
    plan_status: str = "draft",
    profile_snapshot: dict[str, Any] | None = None,
) -> list[PlanChunk]:
    """Chunk a diet plan markdown into PlanChunk objects."""
    return _chunk(
        markdown=markdown,
        plan_type="diet",
        user_id=user_id,
        plan_id=plan_id,
        plan_version=plan_version,
        plan_status=plan_status,
        profile_snapshot=profile_snapshot or {},
    )


def chunk_workout_plan(
    markdown: str,
    *,
    user_id: str,
    plan_id: str,
    plan_version: int = 1,
    plan_status: str = "draft",
    profile_snapshot: dict[str, Any] | None = None,
) -> list[PlanChunk]:
    """Chunk a workout plan markdown into PlanChunk objects."""
    return _chunk(
        markdown=markdown,
        plan_type="workout",
        user_id=user_id,
        plan_id=plan_id,
        plan_version=plan_version,
        plan_status=plan_status,
        profile_snapshot=profile_snapshot or {},
    )
