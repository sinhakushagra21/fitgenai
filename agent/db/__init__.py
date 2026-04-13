"""
agent/db/__init__.py
────────────────────
MongoDB database layer for FITGEN.AI.

Exports the singleton database accessor and index initializer.
"""

from agent.db.mongo import get_db, init_indexes

__all__ = ["get_db", "init_indexes"]
