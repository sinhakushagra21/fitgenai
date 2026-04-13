"""
agent/cache/redis_client.py
───────────────────────────
Redis connection singleton and YouTube video cache helpers.

Environment variables:

    REDIS_URL   – Full Redis connection URL
                  (default: redis://localhost:6379/0)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import redis

logger = logging.getLogger("fitgen.cache")

# ── Configuration ────────────────────────────────────────────────

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_YOUTUBE_CACHE_TTL_SECONDS: int = int(os.getenv("YOUTUBE_CACHE_TTL_DAYS", "30")) * 86400
_YOUTUBE_KEY_PREFIX = "youtube:video:"

# ── Singleton client ─────────────────────────────────────────────

_pool: redis.ConnectionPool | None = None


def _get_pool() -> redis.ConnectionPool:
    """Return (or lazily create) the connection pool."""
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool.from_url(
            REDIS_URL,
            max_connections=20,
            decode_responses=True,
        )
        logger.info("Redis connection pool created — url=%s", REDIS_URL)
    return _pool


def get_redis() -> redis.Redis:
    """Return a Redis client from the shared pool."""
    return redis.Redis(connection_pool=_get_pool())


def close_redis() -> None:
    """Disconnect the Redis pool (call on app shutdown)."""
    global _pool
    if _pool is not None:
        _pool.disconnect()
        _pool = None
        logger.info("Redis connection pool closed.")


# ── YouTube video cache ──────────────────────────────────────────


def youtube_cache_get(exercise: str) -> dict[str, str] | None:
    """Retrieve a cached YouTube video for an exercise.

    Returns ``{"title": ..., "url": ..., "channel": ...}`` or None.
    """
    try:
        r = get_redis()
        key = f"{_YOUTUBE_KEY_PREFIX}{exercise.lower()}"
        raw = r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except (redis.RedisError, json.JSONDecodeError) as exc:
        logger.warning("Redis youtube_cache_get failed for %r: %s", exercise, exc)
        return None


def youtube_cache_set(
    exercise: str,
    title: str,
    url: str,
    channel: str = "",
) -> None:
    """Cache a YouTube video for an exercise with TTL."""
    try:
        r = get_redis()
        key = f"{_YOUTUBE_KEY_PREFIX}{exercise.lower()}"
        payload = json.dumps({"title": title, "url": url, "channel": channel})
        r.setex(key, _YOUTUBE_CACHE_TTL_SECONDS, payload)
    except redis.RedisError as exc:
        logger.warning("Redis youtube_cache_set failed for %r: %s", exercise, exc)
