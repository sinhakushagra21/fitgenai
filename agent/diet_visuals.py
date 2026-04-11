"""
agent/diet_visuals.py
─────────────────────
Rich visual components for diet plan output in FITGEN.AI.

Provides:
  1. Macro extraction from plan markdown
  2. Macro donut chart (inline next to Macro Summary table)

All functions accept the raw plan text and user profile, extract
relevant data via regex, and render via matplotlib.
"""

from __future__ import annotations

import re
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers — data extraction from plan markdown
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_macros_from_plan(text: str) -> dict[str, float]:
    """Extract protein_g, carbs_g, fat_g, total_kcal from the Macro Summary table."""
    result: dict[str, float] = {}

    # Try to find gram values from a markdown table
    # Pattern: | Protein | 148 g | 592 kcal | ...
    protein_match = re.search(r"Protein\s*\|\s*(\d+)\s*g\s*\|\s*(\d+)\s*kcal", text, re.IGNORECASE)
    carbs_match = re.search(r"Carbs?\s*\|\s*(\d+)\s*g\s*\|\s*(\d+)\s*kcal", text, re.IGNORECASE)
    fat_match = re.search(r"Fat(?:s)?\s*\|\s*(\d+)\s*g\s*\|\s*(\d+)\s*kcal", text, re.IGNORECASE)
    total_match = re.search(r"Total\s*\|\s*[—–-]*\s*\|\s*~?(\d[,\d]*)\s*kcal", text, re.IGNORECASE)

    if protein_match:
        result["protein_g"] = float(protein_match.group(1))
        result["protein_kcal"] = float(protein_match.group(2))
    if carbs_match:
        result["carbs_g"] = float(carbs_match.group(1))
        result["carbs_kcal"] = float(carbs_match.group(2))
    if fat_match:
        result["fat_g"] = float(fat_match.group(1))
        result["fat_kcal"] = float(fat_match.group(2))
    if total_match:
        result["total_kcal"] = float(total_match.group(1).replace(",", ""))

    # Fallback: try "148g protein" or "protein: 148g" patterns
    if "protein_g" not in result:
        m = re.search(r"(\d+)\s*g\s*protein", text, re.IGNORECASE)
        if not m:
            m = re.search(r"protein[:\s]+(\d+)\s*g", text, re.IGNORECASE)
        if m:
            result["protein_g"] = float(m.group(1))

    if "carbs_g" not in result:
        m = re.search(r"(\d+)\s*g\s*carb", text, re.IGNORECASE)
        if not m:
            m = re.search(r"carb(?:ohydrate)?s?[:\s]+(\d+)\s*g", text, re.IGNORECASE)
        if m:
            result["carbs_g"] = float(m.group(1))

    if "fat_g" not in result:
        m = re.search(r"(\d+)\s*g\s*fat", text, re.IGNORECASE)
        if not m:
            m = re.search(r"fat[:\s]+(\d+)\s*g", text, re.IGNORECASE)
        if m:
            result["fat_g"] = float(m.group(1))

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Macro Donut Chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_macro_donut_chart(
    protein_g: float,
    carbs_g: float,
    fat_g: float,
    total_kcal: float = 0,
) -> plt.Figure:
    """Create a dark-themed donut chart for macro distribution.

    Uses gram values directly (more accurate than percentage extraction).
    """
    protein_kcal = protein_g * 4
    carbs_kcal = carbs_g * 4
    fat_kcal = fat_g * 9
    computed_total = protein_kcal + carbs_kcal + fat_kcal
    display_total = total_kcal if total_kcal > 0 else computed_total

    sizes = [protein_kcal, carbs_kcal, fat_kcal]
    pcts = [v / computed_total * 100 for v in sizes] if computed_total > 0 else [33, 34, 33]

    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    wedges, texts = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="#0a0a0a", linewidth=2),
        textprops={"color": "white", "fontsize": 10, "fontweight": "bold"},
    )

    # Centre text
    ax.text(0, 0.08, f"{display_total:.0f}", ha="center", va="center",
            fontsize=28, fontweight="900", color="white", fontfamily="sans-serif")
    ax.text(0, -0.15, "kcal/day", ha="center", va="center",
            fontsize=10, color="#888888", fontfamily="sans-serif")

    # Legend below
    legend_labels = [
        f"Protein  {protein_g:.0f}g  ({pcts[0]:.0f}%)",
        f"Carbs     {carbs_g:.0f}g  ({pcts[1]:.0f}%)",
        f"Fat         {fat_g:.0f}g  ({pcts[2]:.0f}%)",
    ]
    patches = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in colors]
    ax.legend(
        patches, legend_labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.12),
        fontsize=9, frameon=False, ncol=1,
        labelcolor="white",
    )

    fig.tight_layout()
    return fig
