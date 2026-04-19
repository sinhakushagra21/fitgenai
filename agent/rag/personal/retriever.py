"""
agent/rag/personal/retriever.py
───────────────────────────────
Personal-RAG retrieval.

Given a user's question, returns the most relevant chunks from the
user's OWN plan(s) and profile memory. Called from the two
``general_{domain}_query`` handlers.

Pipeline:
  1. (cheap) rewrite_query    — expand contractions, resolve simple
     temporal cues ("tomorrow" → day_of_week), strip filler. Pure-
     function fast LLM.
  2. metadata_extract         — sniff ``plan_type`` / ``day_of_week``
     hints so we can narrow the vector_search filter.
  3. embed_query              — OpenAI 1536-dim vector.
  4. plan_chunks.vector_search (MANDATORY user_id filter)
  5. anchor chunks            — always include macros / calorie_calc
     (diet) OR split_overview (workout) so totals + weekly structure
     are available in the prompt.
  6. user_memory.vector_search — optional profile/preference recall.
  7. dedupe + sort by score, return top_k.

Output: ``list[RetrievedChunk]`` — a simple dataclass the callers turn
into a context string before handing to ``answer_followup_question``.

Fails soft: any step that errors returns an empty result, letting the
caller fall back to the legacy ``ctx.plan_text`` path.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.config import FAST_MODEL
from agent.db.repositories.plan_chunks_repo import PlanChunksRepository
from agent.db.repositories.user_memory_repo import UserMemoryRepository
from agent.error_utils import handle_exception
from agent.rag.personal.embedder import embed_query
from agent.rag.personal.schema import SectionType
from agent.tracing import log_event

logger = logging.getLogger("fitgen.rag.personal.retriever")

_DAYS = ("monday", "tuesday", "wednesday", "thursday",
         "friday", "saturday", "sunday")
_DAY_RE = re.compile(r"\b(" + "|".join(_DAYS) + r")\b", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────
# Output type
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """One retrieved piece of context.

    ``source`` is either ``"plan_chunks"`` or ``"user_memory"`` — the
    caller uses this to label the context block in the LLM prompt.
    """

    source: str                          # plan_chunks | user_memory
    score: float
    plan_type: str = ""
    section_type: str = ""
    day_of_week: str | None = None
    heading: str = ""
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """Format as a single block suitable for insertion into a prompt."""
        head_bits: list[str] = []
        if self.plan_type:
            head_bits.append(self.plan_type)
        if self.section_type:
            head_bits.append(self.section_type)
        if self.day_of_week:
            head_bits.append(self.day_of_week)
        tag = " / ".join(head_bits)
        prefix = f"[{self.source} · {tag}]" if tag else f"[{self.source}]"
        heading = f"\n{self.heading}" if self.heading else ""
        return f"{prefix}{heading}\n{self.text}".strip()


# ─────────────────────────────────────────────────────────────────────
# Query rewriting / metadata sniff (small, deterministic FAST_MODEL)
# ─────────────────────────────────────────────────────────────────────

_REWRITE_SYSTEM = """\
You rewrite a user's fitness question for vector-search retrieval.

Input: the user's latest message plus an optional brief chat history.
Output: ONE JSON object with these keys:
  "query":        string — a short self-contained search query (<= 25 words),
                  with pronouns resolved and temporal cues (like "tomorrow",
                  "this week") expanded into concrete terms when possible.
  "plan_type":    "diet" | "workout" | "" — infer from topic; "" if unclear.
  "day_of_week":  one of monday..sunday, or "" — only if the user clearly
                  refers to a specific day.
  "section_hint": "" or one of: macros, calorie_calc, meal_day, snack_swaps,
                  rules, hydration, supplements, split_overview, workout_day,
                  warmup, main_lifts, accessories, cardio, mobility, progression.

Return ONLY valid JSON. No markdown fences, no prose.
"""


def _rewrite_query(
    query: str, *, history: list[str] | None = None,
) -> dict[str, Any]:
    """Run a tiny FAST_MODEL call to produce a cleaner query + filters.

    Returns a dict; on failure returns a safe default reusing the raw
    query so retrieval still runs.
    """
    default = {"query": query, "plan_type": "", "day_of_week": "",
               "section_hint": ""}
    try:
        hist_str = ""
        if history:
            # Keep last 4 turns max, truncated.
            tail = [h[:240] for h in history[-4:]]
            hist_str = "Recent history:\n- " + "\n- ".join(tail) + "\n\n"
        llm = ChatOpenAI(model=FAST_MODEL, temperature=0, max_tokens=200)
        resp = llm.invoke([
            SystemMessage(content=_REWRITE_SYSTEM),
            HumanMessage(content=f"{hist_str}User message: {query}"),
        ])
        raw = (resp.content or "").strip()
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return default
        data = json.loads(m.group(0))
        return {
            "query": str(data.get("query") or query).strip() or query,
            "plan_type": str(data.get("plan_type") or "").strip().lower(),
            "day_of_week": str(data.get("day_of_week") or "").strip().lower(),
            "section_hint": str(data.get("section_hint") or "").strip().lower(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[retriever] rewrite failed: %s — using raw query", exc)
        return default


def _regex_day_hint(text: str) -> str:
    m = _DAY_RE.search(text)
    return m.group(1).lower() if m else ""


# ─────────────────────────────────────────────────────────────────────
# PersonalRAG
# ─────────────────────────────────────────────────────────────────────

class PersonalRAG:
    """Retrieval façade used by the tools.

    Usage::

        chunks = PersonalRAG.retrieve(
            user_id=ctx.user_id, query=q,
            plan_type="diet", history=recent,
        )
        context = "\\n\\n".join(c.render() for c in chunks)
    """

    # Anchor chunks we always include if available — keeps totals &
    # structure in-context even when the user asks a narrow question.
    _DIET_ANCHORS = (
        SectionType.MACROS.value,
        SectionType.CALORIE_CALC.value,
    )
    _WORKOUT_ANCHORS = (
        SectionType.SPLIT_OVERVIEW.value,
    )

    @classmethod
    def retrieve(
        cls,
        *,
        user_id: str,
        query: str,
        plan_type: str | None = None,
        include_memory: bool = True,
        top_k: int = 5,
        history: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return up to ``top_k`` ranked RetrievedChunk entries.

        Safe on failure — returns ``[]`` if anything goes wrong; the
        caller falls back to its legacy path.
        """
        if not user_id or not query or not query.strip():
            return []

        try:
            # 1 + 2. Rewrite + metadata sniff.
            meta = _rewrite_query(query, history=history)
            effective_query = meta["query"] or query
            effective_plan_type = plan_type or meta["plan_type"] or None
            effective_day = meta["day_of_week"] or _regex_day_hint(query) or None

            # 3. Embed.
            qvec = embed_query(effective_query)
            if not qvec or not any(qvec):
                return []

            # 4. Vector search over plan_chunks (user-scoped).
            limit = max(top_k * 2, 8)
            ann_hits = PlanChunksRepository.vector_search(
                user_id=user_id,
                query_vector=qvec,
                plan_type=effective_plan_type,
                day_of_week=effective_day,
                num_candidates=max(50, limit * 10),
                limit=limit,
            )

            # 5. Anchor chunks.
            anchors: list[dict[str, Any]] = []
            anchor_types = (
                cls._DIET_ANCHORS if effective_plan_type == "diet"
                else cls._WORKOUT_ANCHORS if effective_plan_type == "workout"
                else cls._DIET_ANCHORS + cls._WORKOUT_ANCHORS
            )
            for sec in anchor_types:
                try:
                    # Try matching plan_type first; if none, leave unset.
                    atype = (effective_plan_type
                             or ("diet" if sec in cls._DIET_ANCHORS else "workout"))
                    doc = PlanChunksRepository.find_anchor(
                        user_id, plan_type=atype, section_type=sec,
                    )
                    if doc:
                        anchors.append(doc)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[retriever] anchor fetch failed: %s", exc)

            # 6. user_memory (optional).
            memory_hits: list[dict[str, Any]] = []
            if include_memory:
                try:
                    memory_hits = UserMemoryRepository.vector_search(
                        user_id=user_id,
                        query_vector=qvec,
                        limit=3,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[retriever] user_memory search failed: %s", exc)

            # 7. Merge + dedupe + rank.
            seen: set[str] = set()
            merged: list[RetrievedChunk] = []

            # Anchors first with a small synthetic score boost so they
            # stay in the top_k window even if ANN is noisy.
            for doc in anchors:
                key = f"pc:{doc.get('_id')}"
                if key in seen:
                    continue
                seen.add(key)
                merged.append(_plan_doc_to_chunk(doc, score_fallback=0.95))

            for doc in ann_hits:
                key = f"pc:{doc.get('_id')}"
                if key in seen:
                    continue
                seen.add(key)
                merged.append(_plan_doc_to_chunk(doc))

            for doc in memory_hits:
                key = f"um:{doc.get('_id')}"
                if key in seen:
                    continue
                seen.add(key)
                merged.append(_memory_doc_to_chunk(doc))

            merged.sort(key=lambda c: c.score, reverse=True)
            result = merged[:top_k]

            log_event(
                "rag.retrieve.ok",
                module="rag.retriever",
                user_id=str(user_id),
                query_preview=query[:120],
                plan_type=effective_plan_type or "",
                day=effective_day or "",
                ann=len(ann_hits),
                anchors=len(anchors),
                memory=len(memory_hits),
                returned=len(result),
            )
            return result

        except Exception as exc:  # noqa: BLE001
            handle_exception(
                exc,
                module="rag.retriever",
                context="PersonalRAG.retrieve",
                level="WARNING",
                extra={"user_id": str(user_id), "query_preview": query[:120]},
            )
            return []


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _plan_doc_to_chunk(
    doc: dict[str, Any], *, score_fallback: float = 0.0,
) -> RetrievedChunk:
    return RetrievedChunk(
        source="plan_chunks",
        score=float(doc.get("score") or score_fallback),
        plan_type=str(doc.get("plan_type") or ""),
        section_type=str(doc.get("section_type") or ""),
        day_of_week=doc.get("day_of_week"),
        heading=str(doc.get("heading") or ""),
        text=str(doc.get("chunk_text") or doc.get("embedded_text") or ""),
        metadata={
            "plan_id": str(doc.get("plan_id") or ""),
            "plan_version": doc.get("plan_version"),
            "plan_status": doc.get("plan_status"),
        },
    )


def _memory_doc_to_chunk(doc: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        source="user_memory",
        score=float(doc.get("score") or 0.0),
        section_type=str(doc.get("memory_type") or ""),
        heading=f"Profile · {doc.get('domain', '')}".strip(" ·"),
        text=str(doc.get("content") or ""),
        metadata={"domain": doc.get("domain")},
    )
