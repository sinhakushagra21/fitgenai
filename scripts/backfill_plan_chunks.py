"""
scripts/backfill_plan_chunks.py
───────────────────────────────
One-shot Personal-RAG backfill for the ``plan_chunks`` collection.

Iterates every diet + workout plan in Mongo, calls the indexer, and
prints a summary. Safe to re-run — ``PlanChunksRepository.upsert_many``
is idempotent via ``(plan_id, source_content_hash)``.

Usage::

    python -m scripts.backfill_plan_chunks
    python -m scripts.backfill_plan_chunks --plan-type diet
    python -m scripts.backfill_plan_chunks --user-id 651f...

Env:
    MONGO_URI, MONGO_DB_NAME, OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # noqa: E402

from agent.db.mongo import get_db  # noqa: E402
from agent.rag.personal.indexer import index_plan  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("backfill_plan_chunks")


def _iter_plans(*, plan_type: str | None, user_id: str | None) -> list[dict[str, Any]]:
    db = get_db()
    q: dict[str, Any] = {}
    if user_id:
        from bson import ObjectId
        q["user_id"] = ObjectId(user_id)
    plans: list[dict[str, Any]] = []
    if plan_type in (None, "diet"):
        plans.extend([{**p, "_plan_type": "diet"} for p in db.diet_plans.find(q)])
    if plan_type in (None, "workout"):
        plans.extend([{**p, "_plan_type": "workout"} for p in db.workout_plans.find(q)])
    return plans


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-type", choices=["diet", "workout"], default=None)
    ap.add_argument("--user-id", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of plans processed (0 = all)")
    args = ap.parse_args()

    plans = _iter_plans(plan_type=args.plan_type, user_id=args.user_id)
    if args.limit:
        plans = plans[: args.limit]

    log.info("Found %d plans to backfill", len(plans))

    total_ok = 0
    total_chunks = 0
    for i, plan in enumerate(plans, 1):
        plan_id = str(plan["_id"])
        plan_type = plan["_plan_type"]
        try:
            n = index_plan(plan_id, plan_type=plan_type)
            total_ok += 1 if n > 0 else 0
            total_chunks += n
            log.info("[%d/%d] %s %s → %d chunks",
                     i, len(plans), plan_type, plan_id, n)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%d/%d] %s %s FAILED: %s",
                        i, len(plans), plan_type, plan_id, exc)

    log.info("Done. plans_ok=%d / %d   chunks_written=%d",
             total_ok, len(plans), total_chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
