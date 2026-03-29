#!/usr/bin/env python3
"""
flow_engineering/iteration_tracker.py
──────────────────────────────────────
Tracks iterative improvement history across multiple refinement rounds.

Records what was modified, why, and the quantitative/qualitative impact
of each iteration. Produces a data-backed improvement trajectory.

Outputs:
  flow_engineering/results/iteration_history.json
  flow_engineering/results/iteration_summary.txt

Run:
    python -m flow_engineering.iteration_tracker
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flow_engineering.chain_variants import ALL_VARIANTS, VARIANT_DESCRIPTIONS
from flow_engineering.flow_evaluator import FlowEvaluator
from flow_engineering.prompt_flow import FLOW_TEST_QUERIES

load_dotenv()

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Iteration definitions ─────────────────────────────────────────
# Each iteration represents a refinement step with a specific change,
# mapped to a chain variant that implements that change.

ITERATIONS = [
    {
        "iteration": 1,
        "variant": "A_vanilla",
        "title": "Baseline — Direct Zero-Shot Prompting",
        "what_changed": "No changes — this is the original baseline using the zero-shot system prompt with a single LLM call.",
        "why_changed": "Establishes the starting performance benchmark for all subsequent improvements.",
        "hypothesis": "Single-shot prompting may miss nuances in complex queries, struggle with multi-domain requests, and produce less evidence-based responses.",
    },
    {
        "iteration": 2,
        "variant": "B_helper_prompt",
        "title": "Knowledge Pre-Generation (Helper Prompt)",
        "what_changed": "Added a knowledge generation step before the main LLM call. The model first generates relevant domain knowledge statements, which are then injected into the enriched prompt.",
        "why_changed": "Baseline analysis showed weak groundedness and missing evidence-based claims. Pre-generating knowledge provides factual grounding before response generation.",
        "hypothesis": "Two-step chain with helper knowledge should improve groundedness and accuracy scores while maintaining coherence.",
    },
    {
        "iteration": 3,
        "variant": "C_decompose_route",
        "title": "Decomposition + Specialist Routing",
        "what_changed": "Break user queries into sub-tasks, classify each by domain (workout/diet/safety/general), route to specialist prompts, then synthesize results.",
        "why_changed": "Multi-domain queries (e.g., 'give me a workout and diet plan') were being addressed superficially. Decomposition ensures each aspect gets dedicated specialist attention.",
        "hypothesis": "Decompose+Route should significantly improve completeness and keyword coverage for multi-domain queries, at the cost of higher latency.",
    },
    {
        "iteration": 4,
        "variant": "D_self_refine",
        "title": "Self-Refine (Meta-Prompting Chain)",
        "what_changed": "Added a 3-step meta-prompting chain: initial generation → self-critique (scoring on accuracy, completeness, practicality, safety, structure) → refined generation that addresses critique.",
        "why_changed": "Even well-structured responses sometimes miss safety disclaimers or have incomplete advice. Self-critique catches these gaps and the refinement step addresses them.",
        "hypothesis": "Self-refine should produce the highest satisfaction and safety scores, though it requires 3× the LLM calls.",
    },
]


class IterationTracker:
    """Tracks and evaluates iterative improvement rounds."""

    def __init__(self):
        self.evaluator = FlowEvaluator()
        self.history: list[dict] = []

    def run_iteration(
        self,
        iteration_def: dict,
        queries: list[dict] | None = None,
    ) -> dict:
        """Run a single iteration and evaluate it."""
        if queries is None:
            queries = FLOW_TEST_QUERIES[:4]  # subset for speed

        variant_name = iteration_def["variant"]
        variant = ALL_VARIANTS[variant_name]
        it_num = iteration_def["iteration"]

        print(f"\n  ── Iteration {it_num}: {iteration_def['title']} ──")
        print(f"  Variant: {variant_name}")
        print(f"  Change: {iteration_def['what_changed'][:80]}...")

        results = []
        for q in queries:
            print(f"    {q['id']}...", end="", flush=True)
            try:
                t0 = time.perf_counter()
                result = variant.invoke({"query": q["query"]})
                latency = time.perf_counter() - t0

                scores = self.evaluator.evaluate_single(
                    q["query"], result["response"], q["expected_keywords"]
                )
                scores["latency_s"] = round(latency, 3)
                scores["query_id"] = q["id"]
                results.append(scores)
                print(f"  composite={scores['composite']:.2f}")
            except Exception as e:
                print(f"  ✗ {e}")
                results.append({
                    "query_id": q["id"],
                    "composite": 0, "coherence": 0, "groundedness": 0,
                    "satisfaction": 0, "keyword_rate": 0, "latency_s": 0,
                })

        # Aggregate scores
        agg = {
            "composite": round(float(np.mean([r["composite"] for r in results])), 2),
            "coherence": round(float(np.mean([r["coherence"] for r in results])), 2),
            "groundedness": round(float(np.mean([r["groundedness"] for r in results])), 2),
            "satisfaction": round(float(np.mean([r["satisfaction"] for r in results])), 2),
            "keyword_rate": round(float(np.mean([r["keyword_rate"] for r in results])), 2),
            "avg_latency_s": round(float(np.mean([r["latency_s"] for r in results])), 2),
        }

        record = {
            **iteration_def,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "aggregate_scores": agg,
            "per_query_scores": results,
        }

        self.history.append(record)
        return record

    def run_all_iterations(self, queries: list[dict] | None = None) -> list[dict]:
        """Run all defined iterations sequentially."""
        for it_def in ITERATIONS:
            self.run_iteration(it_def, queries)
        return self.history

    def compute_deltas(self) -> list[dict]:
        """Compute improvement deltas between consecutive iterations."""
        deltas = []
        for i in range(1, len(self.history)):
            prev = self.history[i - 1]["aggregate_scores"]
            curr = self.history[i]["aggregate_scores"]
            delta = {
                "from_iteration": self.history[i - 1]["iteration"],
                "to_iteration": self.history[i]["iteration"],
                "title": self.history[i]["title"],
            }
            for metric in ["composite", "coherence", "groundedness", "satisfaction", "keyword_rate"]:
                delta[f"{metric}_delta"] = round(curr[metric] - prev[metric], 2)
            deltas.append(delta)
        return deltas

    def generate_summary(self) -> str:
        """Generate improvement trajectory summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("  ITERATIVE IMPROVEMENT HISTORY")
        lines.append("=" * 70)

        # Trajectory table
        lines.append(f"\n{'It#':<4} {'Title':<40} {'Composite':>9} {'Coher.':>7} {'Ground.':>7} {'Satisf.':>7} {'KW%':>5} {'Lat(s)':>7}")
        lines.append("-" * 88)

        for record in self.history:
            s = record["aggregate_scores"]
            lines.append(
                f"  {record['iteration']:<3} {record['title'][:38]:<38} "
                f"{s['composite']:>8.2f} {s['coherence']:>6.1f} "
                f"{s['groundedness']:>6.1f} {s['satisfaction']:>6.1f} "
                f"{s['keyword_rate']:>4.0%} {s['avg_latency_s']:>6.1f}"
            )

        # Deltas
        deltas = self.compute_deltas()
        if deltas:
            lines.append(f"\n── Improvement Deltas ──")
            for d in deltas:
                comp_delta = d["composite_delta"]
                symbol = "📈" if comp_delta > 0 else "📉" if comp_delta < 0 else "➡️"
                lines.append(
                    f"  It{d['from_iteration']}→{d['to_iteration']} {symbol} "
                    f"composite={comp_delta:+.2f}  "
                    f"coherence={d['coherence_delta']:+.1f}  "
                    f"groundedness={d['groundedness_delta']:+.1f}  "
                    f"satisfaction={d['satisfaction_delta']:+.1f}"
                )

        # Overall improvement
        if len(self.history) >= 2:
            first = self.history[0]["aggregate_scores"]
            last = self.history[-1]["aggregate_scores"]
            overall_delta = last["composite"] - first["composite"]
            pct = (overall_delta / max(first["composite"], 0.01)) * 100
            lines.append(f"\n── Overall Improvement ──")
            lines.append(f"  Baseline composite:  {first['composite']:.2f}")
            lines.append(f"  Final composite:     {last['composite']:.2f}")
            lines.append(f"  Improvement:         {overall_delta:+.2f} ({pct:+.1f}%)")

        # Per-iteration reasoning
        lines.append(f"\n── Iteration Details ──")
        for record in self.history:
            lines.append(f"\n  Iteration {record['iteration']}: {record['title']}")
            lines.append(f"    What: {record['what_changed'][:100]}")
            lines.append(f"    Why:  {record['why_changed'][:100]}")
            lines.append(f"    Hypothesis: {record['hypothesis'][:100]}")

        return "\n".join(lines) + "\n"


# ── CLI ───────────────────────────────────────────────────────────

def main():
    tracker = IterationTracker()

    print(f"\n{'='*60}")
    print(f"  ITERATIVE IMPROVEMENT TRACKER")
    print(f"  {len(ITERATIONS)} iterations × {len(FLOW_TEST_QUERIES[:4])} queries")
    print(f"{'='*60}")

    tracker.run_all_iterations()

    # Save history
    history_path = RESULTS_DIR / "iteration_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(tracker.history, f, indent=2, ensure_ascii=False, default=str)

    # Summary
    summary = tracker.generate_summary()
    summary_path = RESULTS_DIR / "iteration_summary.txt"
    summary_path.write_text(summary)

    print(f"\n{summary}")
    print(f"History: {history_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
