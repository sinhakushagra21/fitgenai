"""
agent/db/repositories/__init__.py
─────────────────────────────────
Convenience re-exports for all repository classes.
"""

from agent.db.repositories.user_repo import UserRepository
from agent.db.repositories.diet_plan_repo import DietPlanRepository
from agent.db.repositories.workout_plan_repo import WorkoutPlanRepository
from agent.db.repositories.session_repo import SessionRepository
from agent.db.repositories.feedback_repo import FeedbackRepository

__all__ = [
    "UserRepository",
    "DietPlanRepository",
    "WorkoutPlanRepository",
    "SessionRepository",
    "FeedbackRepository",
]
