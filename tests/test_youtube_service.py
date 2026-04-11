"""
tests/test_youtube_service.py
─────────────────────────────
Unit tests for YouTube video enrichment service.
No API calls — tests deterministic helpers only.
"""

import pytest

from agent.tools.youtube_service import (
    _is_exercise_name,
    extract_exercise_names,
    _fallback_search_url,
    enrich_plan_with_videos,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _is_exercise_name
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestIsExerciseName:
    @pytest.mark.parametrize("name", [
        "Barbell Bench Press",
        "Incline Dumbbell Press",
        "Romanian Deadlift",
        "Cable Flyes",
        "Tricep Pushdown",
        "Walking Lunges",
        "Pull-Ups",
        "Face Pulls",
        "Hip Thrust",
        "Lat Pulldown",
        # Known short single-word exercises
        "Plank",
        "Crunch",
        "Dips",
        "Shrugs",
        "Squat",
        "Lunge",
    ])
    def test_valid_exercises(self, name):
        assert _is_exercise_name(name) is True

    @pytest.mark.parametrize("text", [
        "",
        "ab",                 # Too short
        "90s",                # Rest duration
        "60s",
        "120s",
        "3×10",               # Sets/reps
        "4×8",
        "18 sets",
        "3 reps",
        "Day 1",              # Day headers
        "Rest",
        "Exercise",           # Table headers
        "Sets × Reps",
        "Monday",
        "Upper Push",         # Body part / split labels
        "Lower Body",
        "Focus",
        "Volume",
        "## Day 4",           # Markdown headers
        "---",                # Separator
        "123",                # Pure numbers
        "Week 1",
        "Phase 2",
        "Total",
    ])
    def test_rejects_non_exercises(self, text):
        assert _is_exercise_name(text) is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# extract_exercise_names
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestExtractExerciseNames:
    SAMPLE_PLAN = """
## Day 1 — Upper Push

| Exercise | Sets × Reps | Rest |
|----------|-------------|------|
| Barbell Bench Press | 4×8 | 90s |
| Incline Dumbbell Press | 3×10 | 75s |
| Overhead Press | 3×8 | 90s |

## Day 2 — Lower Body

| Exercise | Sets × Reps | Rest |
|----------|-------------|------|
| Barbell Squat | 4×6 | 120s |
| Romanian Deadlift | 3×10 | 90s |
"""

    def test_extracts_all_exercises(self):
        exercises = extract_exercise_names(self.SAMPLE_PLAN)
        assert "Barbell Bench Press" in exercises
        assert "Incline Dumbbell Press" in exercises
        assert "Overhead Press" in exercises
        assert "Barbell Squat" in exercises
        assert "Romanian Deadlift" in exercises

    def test_no_false_positives(self):
        exercises = extract_exercise_names(self.SAMPLE_PLAN)
        for ex in exercises:
            assert ex not in ("90s", "75s", "120s", "Rest", "Exercise", "Upper Push", "Lower Body")

    def test_no_duplicates(self):
        plan_with_dupes = """
| Exercise | Sets |
|----------|------|
| Bench Press | 3×10 |
| Bench Press | 3×8 |
"""
        exercises = extract_exercise_names(plan_with_dupes)
        assert exercises.count("Bench Press") == 1

    def test_empty_plan(self):
        assert extract_exercise_names("") == []

    def test_plan_without_tables(self):
        result = extract_exercise_names("This is a random text with no workout info.")
        assert result == []

    def test_respects_max_cap(self):
        # Build a plan with 55 exercises to exceed the 50 cap
        rows = "\n".join(f"| Exercise {i} Name | 3×10 | 60s |" for i in range(55))
        plan = f"| Exercise | Sets | Rest |\n|---|---|---|\n{rows}"
        exercises = extract_exercise_names(plan)
        assert len(exercises) <= 50  # _MAX_EXERCISES

    def test_parenthetical_variants_deduplicated(self):
        plan = """| Exercise | Sets |
|----------|------|
| Face Pull (light) | 3×12 |
| Face Pull (external rotation focus) | 3×15 |"""
        exercises = extract_exercise_names(plan)
        assert exercises.count("Face Pull") == 1
        assert len(exercises) == 1

    def test_large_plan_all_extracted(self):
        """A 25-exercise plan should have all exercises extracted."""
        exercise_names = [
            "Barbell Bench Press", "Seated Cable Row", "Machine Shoulder Press",
            "Lat Pulldown", "Tricep Pushdown", "Leg Press", "Romanian Deadlift",
            "Walking Lunge", "Seated Leg Curl", "Calf Raise", "Incline Dumbbell Press",
            "Cable Face Pull", "Dumbbell Lateral Raise", "Barbell Back Squat",
            "Hip Thrust", "Leg Extension", "Cable Crunch", "Dead Bug",
            "Goblet Squat", "DB Hammer Curl", "Cable Low Row",
            "Chest Supported Row", "DB Bulgarian Split Squat",
            "Hanging Knee Raise", "Bird Dog",
        ]
        rows = "\n".join(f"| {name} | 3×10 | 60s |" for name in exercise_names)
        plan = f"| Exercise | Sets | Rest |\n|---|---|---|\n{rows}"
        exercises = extract_exercise_names(plan)
        assert len(exercises) == 25


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _fallback_search_url
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestFallbackSearchUrl:
    def test_generates_valid_url(self):
        result = _fallback_search_url("Barbell Squat")
        assert "youtube.com/results?search_query=" in result["url"]
        assert "Barbell+Squat" in result["url"]

    def test_has_title(self):
        result = _fallback_search_url("Deadlift")
        assert "Deadlift" in result["title"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# enrich_plan_with_videos (integration, no API key)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestEnrichPlanWithVideos:
    def test_injects_tutorial_column(self):
        plan = """| Exercise | Sets |
|----------|------|
| Bench Press | 3×10 |
| Squat Jump | 3×8 |"""
        enriched = enrich_plan_with_videos(plan)
        assert "| Tutorial |" in enriched
        assert "youtube.com" in enriched
        assert "▶️ Tutorial" in enriched

    def test_tutorial_header_added(self):
        plan = """| Exercise | Sets |
|----------|------|
| Deadlift | 3×5 |"""
        enriched = enrich_plan_with_videos(plan)
        lines = enriched.strip().split("\n")
        assert lines[0].endswith("Tutorial |")

    def test_returns_unchanged_if_no_exercises(self):
        plan = "This is just text, no exercises here."
        assert enrich_plan_with_videos(plan) == plan

    def test_overview_table_not_modified(self):
        plan = """| Parameter | Value |
|-----------|-------|
| Goal | Fat loss |

| Exercise | Sets |
|----------|------|
| Deadlift | 3×5 |"""
        enriched = enrich_plan_with_videos(plan)
        lines = enriched.strip().split("\n")
        assert "Tutorial" not in lines[0]
        assert any("Tutorial" in line for line in lines)

    def test_each_exercise_row_gets_link(self):
        plan = """| Exercise | Sets |
|----------|------|
| Bench Press | 3×10 |
| Squat Jump | 3×8 |"""
        enriched = enrich_plan_with_videos(plan)
        data_rows = [l for l in enriched.split("\n") if "youtube.com" in l]
        assert len(data_rows) == 2

    def test_plank_gets_tutorial_link(self):
        """Single-word exercise 'Plank' should get a tutorial link."""
        plan = """| Exercise | Sets |
|----------|------|
| Plank | 3×30s |"""
        enriched = enrich_plan_with_videos(plan)
        assert "youtube.com" in enriched
        assert "▶️ Tutorial" in enriched

    def test_parenthetical_exercises_get_links(self):
        """Exercises with parentheticals should match after stripping."""
        plan = """| Exercise | Sets |
|----------|------|
| Face Pull (light, external rotation) | 3×15 |
| Calf Raise (machine) | 4×15 |"""
        enriched = enrich_plan_with_videos(plan)
        data_rows = [l for l in enriched.split("\n") if "youtube.com" in l]
        assert len(data_rows) == 2
