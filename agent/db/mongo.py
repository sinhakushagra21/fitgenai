"""
agent/db/mongo.py
─────────────────
MongoDB connection singleton for FITGEN.AI.

Uses pymongo with connection pooling.  The MongoClient is created once
and reused across all requests.  Environment variables:

    MONGO_URI       – Full MongoDB connection string
                      (default: mongodb://localhost:27017)
    MONGO_DB_NAME   – Database name (default: fitgen_ai)
"""

from __future__ import annotations

import logging
import os

from pymongo import MongoClient
from pymongo.database import Database

logger = logging.getLogger("fitgen.db")

# ── Configuration ────────────────────────────────────────────────

MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "fitgen_ai")

# ── Singleton client ─────────────────────────────────────────────

_client: MongoClient | None = None


def _get_client() -> MongoClient:
    """Return (or lazily create) the singleton MongoClient."""
    global _client
    if _client is None:
        _client = MongoClient(
            MONGO_URI,
            maxPoolSize=20,
            minPoolSize=2,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            retryWrites=True,
        )
        logger.info("MongoDB client created — uri=%s db=%s", MONGO_URI, MONGO_DB_NAME)
    return _client


def get_db() -> Database:
    """Return the FITGEN.AI database handle."""
    return _get_client()[MONGO_DB_NAME]


def close_client() -> None:
    """Gracefully close the MongoClient (call on app shutdown)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB client closed.")


# ── Index creation ───────────────────────────────────────────────

def init_indexes() -> None:
    """Create all required indexes.  Safe to call on every startup
    — ``create_index`` is idempotent.
    """
    db = get_db()

    # users
    db.users.create_index("email", unique=True, sparse=True)

    # diet_plans
    db.diet_plans.create_index([("user_id", 1), ("status", 1)])
    db.diet_plans.create_index("session_id")

    # workout_plans
    db.workout_plans.create_index([("user_id", 1), ("status", 1)])
    db.workout_plans.create_index("session_id")

    # sessions — TTL auto-deletes after 30 days
    db.sessions.create_index("session_id", unique=True)
    db.sessions.create_index("user_id")
    db.sessions.create_index("expires_at", expireAfterSeconds=0)

    # feedback
    db.feedback.create_index("session_id")
    db.feedback.create_index([("user_id", 1), ("created_at", -1)])

    logger.info("MongoDB indexes ensured.")
