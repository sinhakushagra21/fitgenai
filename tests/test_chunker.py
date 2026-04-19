"""
tests/test_chunker.py
─────────────────────
Unit tests for the section-aware markdown chunker used by Personal RAG.
"""

from __future__ import annotations

from agent.rag.personal.chunker import chunk_diet_plan, chunk_workout_plan
from agent.rag.personal.schema import SectionType


_DIET_MD = """\
# Diet Plan

Intro line.

## Calorie Calculation
Your TDEE is 2400 kcal. Target 1900 kcal for fat loss.

## Macro Targets
Protein 160g / Carbs 180g / Fat 70g.

## 7-Day Meal Plan

### Monday
Breakfast: oats.
Lunch: chicken salad.
Dinner: rice + tofu.

### Tuesday
Breakfast: eggs.

## Snack Swaps
- crisps → rice cakes

## Hydration
3L water per day.
"""


_WORKOUT_MD = """\
# Workout Plan

## Weekly Split Overview
PPL x 2, rest Sunday.

## Monday — Push
### Main Lifts
- Bench press 4x6
### Accessories
- Tricep rope pushdown

## Tuesday — Pull
### Main Lifts
- Barbell row 4x6
"""


class TestDietChunker:
    def test_produces_chunks(self):
        chunks = chunk_diet_plan(
            _DIET_MD, user_id="u1", plan_id="p1", plan_version=2,
            plan_status="draft",
        )
        assert len(chunks) >= 4

    def test_section_types(self):
        chunks = chunk_diet_plan(
            _DIET_MD, user_id="u1", plan_id="p1",
        )
        types = {c.section_type for c in chunks}
        assert SectionType.CALORIE_CALC.value in types
        assert SectionType.MACROS.value in types
        assert SectionType.MEAL_DAY.value in types
        assert SectionType.SNACK_SWAPS.value in types
        assert SectionType.HYDRATION.value in types

    def test_day_of_week_extracted(self):
        chunks = chunk_diet_plan(
            _DIET_MD, user_id="u1", plan_id="p1",
        )
        meal_days = [c for c in chunks
                     if c.section_type == SectionType.MEAL_DAY.value]
        days = {c.day_of_week for c in meal_days}
        assert "monday" in days
        assert "tuesday" in days

    def test_content_hash_deterministic(self):
        a = chunk_diet_plan(_DIET_MD, user_id="u1", plan_id="p1")
        b = chunk_diet_plan(_DIET_MD, user_id="u1", plan_id="p1")
        assert [c.source_content_hash for c in a] == \
               [c.source_content_hash for c in b]

    def test_embedded_text_contains_tags(self):
        chunks = chunk_diet_plan(_DIET_MD, user_id="u1", plan_id="p1")
        for c in chunks:
            assert "plan_type=diet" in c.embedded_text
            assert f"section={c.section_type}" in c.embedded_text

    def test_empty_markdown(self):
        assert chunk_diet_plan("", user_id="u", plan_id="p") == []
        assert chunk_diet_plan("   \n\n  ", user_id="u", plan_id="p") == []

    def test_to_mongo_conversion(self):
        chunks = chunk_diet_plan(
            _DIET_MD, user_id="507f1f77bcf86cd799439011",
            plan_id="507f1f77bcf86cd799439012",
        )
        doc = chunks[0].to_mongo()
        # ObjectId cast (falls back to str if invalid, but these are valid)
        assert "user_id" in doc and "plan_id" in doc
        assert doc["plan_type"] == "diet"
        assert isinstance(doc["embedding"], list)


class TestWorkoutChunker:
    def test_produces_chunks(self):
        chunks = chunk_workout_plan(_WORKOUT_MD, user_id="u1", plan_id="p1")
        assert len(chunks) >= 3

    def test_split_overview_classified(self):
        chunks = chunk_workout_plan(_WORKOUT_MD, user_id="u1", plan_id="p1")
        types = {c.section_type for c in chunks}
        assert SectionType.SPLIT_OVERVIEW.value in types

    def test_workout_day_extracted(self):
        chunks = chunk_workout_plan(_WORKOUT_MD, user_id="u1", plan_id="p1")
        day_chunks = [c for c in chunks
                      if c.section_type == SectionType.WORKOUT_DAY.value
                      or c.day_of_week]
        days = {c.day_of_week for c in day_chunks if c.day_of_week}
        assert "monday" in days or "tuesday" in days
