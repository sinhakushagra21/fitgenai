"""
agent/cache/__init__.py
───────────────────────
Redis caching layer for FITGEN.AI.
"""

from agent.cache.redis_client import get_redis, youtube_cache_get, youtube_cache_set

__all__ = ["get_redis", "youtube_cache_get", "youtube_cache_set"]
