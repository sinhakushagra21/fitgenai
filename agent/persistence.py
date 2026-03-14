"""
agent/persistence.py
────────────────────
SQLite persistence helpers for FITGEN.AI multi-turn workflows.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[1] / "fitgen.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_records (
                state_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                plan_text TEXT,
                calendar_sync INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context_states (
                context_id TEXT PRIMARY KEY,
                user_email TEXT,
                user_profile_json TEXT NOT NULL,
                workflow_json TEXT NOT NULL,
                calendar_sync INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(context_states)").fetchall()
        }
        if "user_email" not in columns:
            conn.execute("ALTER TABLE context_states ADD COLUMN user_email TEXT")
        conn.commit()


def get_record(state_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT state_id, domain, profile_json, plan_text, calendar_sync, updated_at FROM user_records WHERE state_id = ?",
            (state_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "state_id": row["state_id"],
        "domain": row["domain"],
        "profile": json.loads(row["profile_json"]),
        "plan_text": row["plan_text"] or "",
        "calendar_sync": bool(row["calendar_sync"]),
        "updated_at": row["updated_at"],
    }


def upsert_record(
    *,
    state_id: str,
    domain: str,
    profile: dict[str, Any],
    plan_text: str,
    calendar_sync: bool = False,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_records (state_id, domain, profile_json, plan_text, calendar_sync, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(state_id) DO UPDATE SET
                domain = excluded.domain,
                profile_json = excluded.profile_json,
                plan_text = excluded.plan_text,
                calendar_sync = excluded.calendar_sync,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                state_id,
                domain,
                json.dumps(profile, ensure_ascii=False),
                plan_text,
                int(calendar_sync),
            ),
        )
        conn.commit()


def update_calendar_sync(state_id: str, enabled: bool) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE user_records SET calendar_sync = ?, updated_at = CURRENT_TIMESTAMP WHERE state_id = ?",
            (int(enabled), state_id),
        )
        conn.commit()


def delete_record(state_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM user_records WHERE state_id = ?", (state_id,))
        conn.commit()
        return cur.rowcount > 0


def get_context_state(context_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT context_id, user_email, user_profile_json, workflow_json, calendar_sync, updated_at FROM context_states WHERE context_id = ?",
            (context_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "context_id": row["context_id"],
        "state_id": row["context_id"],
        "user_email": row["user_email"] or "",
        "user_profile": json.loads(row["user_profile_json"] or "{}"),
        "workflow": json.loads(row["workflow_json"] or "{}"),
        "calendar_sync_requested": bool(row["calendar_sync"]),
        "updated_at": row["updated_at"],
    }


def get_latest_context_state_by_email(user_email: str) -> dict[str, Any] | None:
    if not user_email:
        return None
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT context_id, user_email, user_profile_json, workflow_json, calendar_sync, updated_at
            FROM context_states
            WHERE user_email = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (user_email,),
        ).fetchone()
    if not row:
        return None
    return {
        "context_id": row["context_id"],
        "state_id": row["context_id"],
        "user_email": row["user_email"] or "",
        "user_profile": json.loads(row["user_profile_json"] or "{}"),
        "workflow": json.loads(row["workflow_json"] or "{}"),
        "calendar_sync_requested": bool(row["calendar_sync"]),
        "updated_at": row["updated_at"],
    }


def upsert_context_state(
    *,
    context_id: str,
    user_email: str,
    user_profile: dict[str, Any],
    workflow: dict[str, Any],
    calendar_sync_requested: bool,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO context_states (context_id, user_email, user_profile_json, workflow_json, calendar_sync, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(context_id) DO UPDATE SET
                user_email = excluded.user_email,
                user_profile_json = excluded.user_profile_json,
                workflow_json = excluded.workflow_json,
                calendar_sync = excluded.calendar_sync,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                context_id,
                user_email,
                json.dumps(user_profile, ensure_ascii=False),
                json.dumps(workflow, ensure_ascii=False),
                int(calendar_sync_requested),
            ),
        )
        conn.commit()


def delete_context_state(context_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM context_states WHERE context_id = ?", (context_id,))
        conn.commit()
        return cur.rowcount > 0
