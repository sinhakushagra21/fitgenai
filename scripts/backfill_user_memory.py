"""
scripts/backfill_user_memory.py
───────────────────────────────
One-shot Personal-RAG backfill for the ``user_memory`` collection.

Iterates every user and re-indexes their profile groups
(base / diet / workout).

Usage::

    python -m scripts.backfill_user_memory
    python -m scripts.backfill_user_memory --email foo@bar.com
"""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

load_dotenv()  # noqa: E402

from agent.db.mongo import get_db  # noqa: E402
from agent.rag.personal.indexer import index_profile  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("backfill_user_memory")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    db = get_db()
    q = {"email": args.email} if args.email else {}
    users = list(db.users.find(q, {"email": 1}))
    if args.limit:
        users = users[: args.limit]

    log.info("Found %d users to backfill", len(users))

    ok = 0
    written = 0
    for i, u in enumerate(users, 1):
        email = u.get("email")
        if not email:
            continue
        try:
            n = index_profile(email)
            written += n
            ok += 1 if n > 0 else 0
            log.info("[%d/%d] %s → %d groups", i, len(users), email, n)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%d/%d] %s FAILED: %s", i, len(users), email, exc)

    log.info("Done. users_ok=%d / %d   groups_written=%d",
             ok, len(users), written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
