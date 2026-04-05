"""
agent/visualizations.py
───────────────────────
Visual plan outputs for FITGEN.AI fitness and nutrition plans.

Provides 3 visualization functions:
  1. Weekly schedule heatmap (workout days / muscle groups)
  2. Macro distribution pie chart (diet plans)
  3. Progressive overload timeline (strength progression)
"""

from __future__ import annotations

import re
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Weekly Schedule Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def create_weekly_schedule(
    workout_days: int = 5,
    plan_text: str = "",
    profile: dict[str, Any] | None = None,
) -> plt.Figure:
    """Create a weekly workout schedule heatmap.

    Args:
        workout_days: Number of training days per week.
        plan_text: Generated plan text to extract muscle groups from.
        profile: User profile dict (optional).

    Returns:
        matplotlib Figure.
    """
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    categories = ["Strength", "Cardio", "Flexibility", "Rest"]

    # Build a simple schedule grid based on workout_days
    data = np.zeros((len(categories), 7))

    # Distribute workout days across the week
    if workout_days >= 6:
        train_days = [0, 1, 2, 3, 4, 5]  # Mon-Sat
    elif workout_days == 5:
        train_days = [0, 1, 2, 3, 4]  # Mon-Fri
    elif workout_days == 4:
        train_days = [0, 1, 3, 4]  # Mon, Tue, Thu, Fri
    elif workout_days == 3:
        train_days = [0, 2, 4]  # Mon, Wed, Fri
    elif workout_days == 2:
        train_days = [0, 3]  # Mon, Thu
    else:
        train_days = [0]

    rest_days = [d for d in range(7) if d not in train_days]

    # Assign intensities
    for i, day in enumerate(train_days):
        data[0, day] = 0.8 + 0.2 * (i % 2)  # Strength: high intensity
        data[1, day] = 0.3 + 0.1 * (i % 3)  # Cardio: moderate
        data[2, day] = 0.4                    # Flexibility: moderate

    for day in rest_days:
        data[3, day] = 1.0  # Rest days
        data[2, day] = 0.2  # Light stretching on rest days

    fig, ax = plt.subplots(figsize=(10, 4))

    cmap = plt.cm.YlOrRd
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(7))
    ax.set_xticklabels(days, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=11)

    # Add text annotations
    for i in range(len(categories)):
        for j in range(7):
            value = data[i, j]
            if value > 0:
                text_color = "white" if value > 0.6 else "black"
                label = f"{value:.0%}" if value > 0 else ""
                ax.text(j, i, label, ha="center", va="center",
                        color=text_color, fontsize=9, fontweight="bold")

    goal = profile.get("goal", "fitness") if profile else "fitness"
    ax.set_title(
        f"Weekly Training Schedule — {workout_days} Days/Week ({goal.title()})",
        fontsize=13, fontweight="bold", pad=12,
    )

    fig.colorbar(im, ax=ax, label="Intensity", shrink=0.8)
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Macro Distribution Pie Chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def create_macro_pie_chart(
    plan_text: str = "",
    profile: dict[str, Any] | None = None,
) -> plt.Figure:
    """Create a macro nutrient distribution pie chart.

    Extracts protein/carbs/fat percentages from plan text or estimates
    based on the user's goal.

    Returns:
        matplotlib Figure.
    """
    # Try to extract macros from plan text
    protein_pct, carbs_pct, fat_pct = _extract_macros(plan_text)

    # Fallback: estimate based on goal
    if protein_pct == 0 and carbs_pct == 0 and fat_pct == 0:
        goal = (profile or {}).get("goal", "maintenance")
        protein_pct, carbs_pct, fat_pct = _goal_macros(goal)

    sizes = [protein_pct, carbs_pct, fat_pct]
    labels = [
        f"Protein\n{protein_pct}%",
        f"Carbs\n{carbs_pct}%",
        f"Fat\n{fat_pct}%",
    ]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    explode = (0.05, 0.05, 0.05)

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 12, "fontweight": "bold"},
        pctdistance=0.75,
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(11)

    goal = (profile or {}).get("goal", "fitness")
    ax.set_title(
        f"Daily Macro Distribution — {goal.title()} Goal",
        fontsize=14, fontweight="bold", pad=15,
    )

    # Add calorie estimate legend
    weight = (profile or {}).get("weight_kg", 70)
    est_calories = _estimate_calories(weight, goal)
    legend_text = (
        f"Est. Daily Calories: ~{est_calories} kcal\n"
        f"Protein: ~{int(est_calories * protein_pct / 400)}g | "
        f"Carbs: ~{int(est_calories * carbs_pct / 400)}g | "
        f"Fat: ~{int(est_calories * fat_pct / 900)}g"
    )
    ax.text(
        0, -1.3, legend_text,
        ha="center", va="center", fontsize=10,
        style="italic", color="#555",
        transform=ax.transAxes,
    )

    fig.tight_layout()
    return fig


def _extract_macros(text: str) -> tuple[int, int, int]:
    """Try to extract protein/carbs/fat percentages from plan text."""
    patterns = [
        r"protein[:\s]*(\d+)\s*%",
        r"carb(?:ohydrate)?s?[:\s]*(\d+)\s*%",
        r"fat[:\s]*(\d+)\s*%",
    ]
    values = []
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        values.append(int(match.group(1)) if match else 0)

    if sum(values) > 0 and sum(values) <= 100:
        return tuple(values)  # type: ignore
    return (0, 0, 0)


def _goal_macros(goal: str) -> tuple[int, int, int]:
    """Default macro split based on fitness goal."""
    macros = {
        "fat loss": (40, 30, 30),
        "muscle gain": (30, 45, 25),
        "maintenance": (30, 40, 30),
        "performance": (25, 50, 25),
    }
    return macros.get(goal, (30, 40, 30))


def _estimate_calories(weight_kg: float, goal: str) -> int:
    """Rough daily calorie estimate."""
    base = weight_kg * 30  # maintenance
    adjustments = {
        "fat loss": -500,
        "muscle gain": 300,
        "maintenance": 0,
        "performance": 200,
    }
    return int(base + adjustments.get(goal, 0))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Progressive Overload Timeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def create_progress_timeline(
    weeks: int = 12,
    profile: dict[str, Any] | None = None,
) -> plt.Figure:
    """Create a progressive overload projection timeline.

    Shows projected strength and volume increases over a training period.

    Returns:
        matplotlib Figure.
    """
    fitness_level = (profile or {}).get("fitness_level", "beginner")

    # Weekly progression rates
    rates = {
        "beginner": {"strength": 0.05, "volume": 0.04},
        "intermediate": {"strength": 0.025, "volume": 0.03},
        "advanced": {"strength": 0.01, "volume": 0.02},
    }
    rate = rates.get(fitness_level, rates["beginner"])

    week_nums = np.arange(1, weeks + 1)
    strength = 100 * (1 + rate["strength"]) ** (week_nums - 1)
    volume = 100 * (1 + rate["volume"]) ** (week_nums - 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = "#2ecc71"
    color2 = "#3498db"

    ax1.fill_between(week_nums, 100, strength, alpha=0.15, color=color1)
    ax1.plot(week_nums, strength, color=color1, linewidth=2.5, marker="o",
             markersize=5, label="Strength (% of baseline)")
    ax1.set_xlabel("Week", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Strength (% of baseline)", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.fill_between(week_nums, 100, volume, alpha=0.15, color=color2)
    ax2.plot(week_nums, volume, color=color2, linewidth=2.5, marker="s",
             markersize=5, linestyle="--", label="Volume (% of baseline)")
    ax2.set_ylabel("Training Volume (% of baseline)", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Annotations
    final_str = strength[-1]
    final_vol = volume[-1]
    ax1.annotate(
        f"+{final_str - 100:.0f}%",
        xy=(weeks, final_str), xytext=(weeks - 2, final_str + 5),
        arrowprops=dict(arrowstyle="->", color=color1),
        fontsize=11, fontweight="bold", color=color1,
    )

    ax1.set_title(
        f"Projected Progress — {weeks}-Week Plan ({fitness_level.title()} Level)",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax1.set_xticks(week_nums)
    ax1.grid(axis="x", alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    fig.tight_layout()
    return fig
