"""
agent/rag/personal
──────────────────
Personal RAG — retrieval over a user's OWN plans and profile.

Public API:
    from agent.rag.personal import (
        PersonalRAG,       # retrieval
        index_plan,        # index a diet/workout plan on save
        index_profile,     # index a user's profile on update
        chunk_diet_plan,
        chunk_workout_plan,
    )
"""

from agent.rag.personal.chunker import (
    SectionType,
    chunk_diet_plan,
    chunk_workout_plan,
)
from agent.rag.personal.indexer import index_plan, index_profile
from agent.rag.personal.retriever import PersonalRAG, RetrievedChunk

__all__ = [
    "PersonalRAG",
    "RetrievedChunk",
    "SectionType",
    "chunk_diet_plan",
    "chunk_workout_plan",
    "index_plan",
    "index_profile",
]
