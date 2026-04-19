#!/usr/bin/env python3
"""
fine_tuning/mine_plan_training_data.py
───────────────────────────────────────
Mine MongoDB confirmed plans and emit OpenAI fine-tuning JSONL.

For each confirmed plan in ``diet_plans`` / ``workout_plans``:
  - system   = domain system prompt (few_shot)
  - user     = rendered profile_snapshot (same format the tool sends)
  - assistant = plan_markdown

Rows whose plans fail ``check_hard_constraints`` are skipped — we only
train on gold that already passes the evaluator.

Tasks supported (``--task``):
  plan    — (profile → plan markdown) pairs mined from confirmed plans
  intent  — (query + workflow context → user_intent JSON) mined from
            sessions.workflow history; falls back to synthetic teacher
            distillation when no trace logs exist
  qa      — (plan + question → answer) pairs mined from logged follow-ups

Run:
    python -m fine_tuning.mine_plan_training_data --domain diet --task plan
    python -m fine_tuning.mine_plan_training_data --domain workout --task plan --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.db.mongo import get_db
from agent.prompts.diet_prompts import DIET_PROMPTS
from agent.prompts.workout_prompts import WORKOUT_PROMPTS
from agent.shared.plan_evaluator import check_hard_constraints

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

VAL_SPLIT = 0.10  # 10% val


# ── Helpers ────────────────────────────────────────────────────────────

def _system_prompt(domain: str) -> str:
    if domain == "diet":
        return DIET_PROMPTS["few_shot"]
    if domain == "workout":
        return WORKOUT_PROMPTS["few_shot"]
    raise ValueError(f"Unknown domain: {domain}")


def _render_user_message(domain: str, profile: dict[str, Any], user_request: str = "") -> str:
    """Render the user message in the same format the tool uses at inference.

    Mirrors the prompt shape in ``llm_helpers.generate_plan`` (no existing_plan
    branch — training is for fresh generation).
    """
    req = user_request or f"Create a personalized {domain} plan."
    return (
        f"Create a personalized {domain} plan using this profile:\n"
        f"{json.dumps(profile, indent=2)}\n\n"
        f'User\'s original request: "{req}"\n\n'
        "IMPORTANT: Follow the Output Contract from your system prompt "
        "EXACTLY. Include ALL required sections in order."
    )


def _split_train_val(examples: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Deterministic split by hash of the example payload."""
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_SPLIT)) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stable_id(doc: dict) -> str:
    """Fingerprint a doc for deterministic split ordering."""
    key = str(doc.get("_id", "")) or json.dumps(doc, default=str, sort_keys=True)
    return hashlib.md5(key.encode("utf-8")).hexdigest()


# ── Task: plan generation ──────────────────────────────────────────────

def mine_plan_examples(domain: str, *, min_plans: int = 0) -> list[dict]:
    """Mine (profile → plan) pairs from confirmed plans."""
    collection_name = f"{domain}_plans"
    db = get_db()
    cursor = db[collection_name].find({"status": "confirmed"})

    system = _system_prompt(domain)
    kept: list[dict] = []
    dropped_empty = 0
    dropped_hard = 0

    docs = sorted(list(cursor), key=_stable_id)

    for doc in docs:
        plan_markdown: str = doc.get("plan_markdown", "") or ""
        profile: dict = doc.get("profile_snapshot", {}) or {}

        if not plan_markdown.strip() or not profile:
            dropped_empty += 1
            continue

        hard = check_hard_constraints(
            plan_markdown, domain=domain, profile=profile,
        )
        if not hard.passed:
            dropped_hard += 1
            continue

        user_msg = _render_user_message(domain, profile)
        kept.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": plan_markdown},
            ],
        })

    print(
        f"  [plan/{domain}] kept={len(kept)} "
        f"dropped_empty={dropped_empty} dropped_hard_fail={dropped_hard}"
    )

    if min_plans and len(kept) < min_plans:
        print(
            f"  ⚠️  Only {len(kept)} examples found (min={min_plans}). "
            "Consider running the synthetic bootstrap or lowering the floor."
        )
    return kept


# ── Task: intent classification ────────────────────────────────────────

def mine_intent_examples(domain: str) -> list[dict]:
    """Mine (user query + workflow context → intent JSON) pairs.

    Source: the ``sessions`` collection logs workflow transitions in
    ``workflow.history`` when FITGEN_TRACE_INTENT=1 is set. If no traces
    exist, returns an empty list (bootstrap script handles the cold-start
    case separately).
    """
    db = get_db()
    cursor = db.sessions.find({
        f"workflow.history.domain": domain,
    })

    system = (
        "You are an intent classifier for FITGEN.AI. Given a user message "
        "and the current workflow state, return JSON: "
        '{"user_intent": "<one of valid_intents>", "reason": "<one sentence>"}.'
    )

    kept: list[dict] = []
    for sess in cursor:
        hist = (sess.get("workflow") or {}).get("history") or []
        for evt in hist:
            if evt.get("domain") != domain:
                continue
            query = evt.get("query")
            classified = evt.get("classified_intent")
            reason = evt.get("reason", "")
            if not query or not classified:
                continue
            user_msg = (
                f"Domain: {domain}\n"
                f"Step: {evt.get('step_completed') or '(none)'}\n"
                f"Has plan: {evt.get('has_plan', False)}\n"
                f"User: {query}"
            )
            assistant = json.dumps({"user_intent": classified, "reason": reason})
            kept.append({
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant},
                ],
            })

    print(f"  [intent/{domain}] kept={len(kept)}")
    return kept


# ── Task: follow-up Q&A ────────────────────────────────────────────────

def mine_qa_examples(domain: str) -> list[dict]:
    """Mine (plan + question → answer) pairs from the llm_traces collection.

    Requires FITGEN_TRACE_LLM=1 during production use so answer_plan_question
    / answer_followup_question calls are persisted. Missing collection
    simply yields zero examples.
    """
    db = get_db()
    if "llm_traces" not in db.list_collection_names():
        print(f"  [qa/{domain}] llm_traces collection missing — 0 examples")
        return []

    cursor = db.llm_traces.find({
        "domain": domain,
        "purpose": {"$in": ["qa", "plan_qa", "followup"]},
    })

    kept: list[dict] = []
    for row in cursor:
        question = row.get("question") or row.get("query")
        plan_text = row.get("plan_text") or ""
        answer = row.get("completion") or row.get("answer")
        if not question or not plan_text or not answer:
            continue
        system = (
            f"You are a helpful fitness and nutrition assistant. "
            f"Answer the user's question using ONLY the {domain} plan below."
        )
        user_msg = f"## Plan:\n{plan_text}\n\n## Question:\n{question}"
        kept.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": answer},
            ],
        })

    print(f"  [qa/{domain}] kept={len(kept)}")
    return kept


# ── Orchestrator ───────────────────────────────────────────────────────

_TASK_FNS = {
    "plan": mine_plan_examples,
    "intent": lambda domain, **_: mine_intent_examples(domain),
    "qa": lambda domain, **_: mine_qa_examples(domain),
}


def main() -> None:
    p = argparse.ArgumentParser(description="Mine FITGEN.AI fine-tuning data from MongoDB")
    p.add_argument("--domain", choices=["diet", "workout"], required=True)
    p.add_argument("--task", choices=["plan", "intent", "qa"], required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="Print stats + 1 sample example, do not write files")
    p.add_argument("--min-plans", type=int, default=0,
                   help="Warn if fewer than this many examples are mined (plan task only)")
    args = p.parse_args()

    print(f"\n▶ Mining {args.task} examples for domain={args.domain}")
    fn = _TASK_FNS[args.task]
    examples = fn(args.domain, min_plans=args.min_plans) if args.task == "plan" else fn(args.domain)

    if not examples:
        print("  ⚠️  No examples produced. Skipping file write.")
        return

    train, val = _split_train_val(examples)
    print(f"  split: train={len(train)} val={len(val)}")

    if args.dry_run:
        print("\n── Sample example (train[0]) ──")
        sample = train[0] if train else val[0]
        preview = {
            "system": sample["messages"][0]["content"][:160] + "...",
            "user": sample["messages"][1]["content"][:200] + "...",
            "assistant": sample["messages"][2]["content"][:200] + "...",
        }
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        print("\n✅ Dry run complete — files not written.")
        return

    train_path = DATA_DIR / f"{args.domain}_{args.task}_train.jsonl"
    val_path = DATA_DIR / f"{args.domain}_{args.task}_val.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(val_path, val)
    print(f"\n✅ Wrote {train_path} ({len(train)} rows)")
    print(f"✅ Wrote {val_path}   ({len(val)} rows)")


if __name__ == "__main__":
    main()
