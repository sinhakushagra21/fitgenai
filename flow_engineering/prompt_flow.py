#!/usr/bin/env python3
"""
flow_engineering/prompt_flow.py
────────────────────────────────
LangChain-based Prompt Flow Runner for FITGEN.AI.

Defines three high-level flow architectures and runs them across
test queries to compare performance:

  1. Single-Chain Flow  — direct prompt → response (baseline)
  2. Chain-of-Chains    — decomposition → specialist calls → synthesis
  3. Router Flow        — intent classification → conditional routing

Uses LangChain's Runnable interface for composable, production-grade
pipeline design.

Run:
    python -m flow_engineering.prompt_flow
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.prompts.base_prompts import BASE_PROMPTS, BASE_ZERO_SHOT, BASE_COT
from flow_engineering.chain_variants import ALL_VARIANTS

load_dotenv()

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4o-mini"

# ── Test queries for flow comparison ──────────────────────────────

FLOW_TEST_QUERIES = [
    {
        "id": "beginner_plan",
        "query": "Create a beginner-friendly 3-day workout plan for someone with dumbbells only.",
        "expected_keywords": ["day 1", "day 2", "day 3", "dumbbell", "sets", "reps"],
        "domain": "workout",
    },
    {
        "id": "vegan_macros",
        "query": "I'm vegan and weigh 70kg. Calculate my daily macros for muscle gain.",
        "expected_keywords": ["protein", "carbs", "fat", "grams", "vegan", "calori"],
        "domain": "diet",
    },
    {
        "id": "multi_domain",
        "query": "I want to lose 5kg in 2 months. Give me a workout schedule and meal plan.",
        "expected_keywords": ["deficit", "workout", "meal", "cardio", "protein"],
        "domain": "both",
    },
    {
        "id": "plateau_advice",
        "query": "I've been stuck at 80kg bench press for 3 weeks. How do I break through?",
        "expected_keywords": ["progressive overload", "deload", "form", "sets", "volume"],
        "domain": "workout",
    },
    {
        "id": "safety_case",
        "query": "I have lower back pain but want to strengthen my core. What exercises are safe?",
        "expected_keywords": ["doctor", "plank", "bird dog", "avoid", "pain"],
        "domain": "workout",
    },
    {
        "id": "supplement_question",
        "query": "What supplements should I take for better gym performance and recovery?",
        "expected_keywords": ["creatine", "protein", "sleep", "evidence"],
        "domain": "diet",
    },
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Flow Architectures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class FlowRunner:
    """Orchestrates running multiple flow architectures across queries."""

    def __init__(self, model: str = MODEL):
        self.model = model
        self.variants = ALL_VARIANTS

    def run_single_query(self, query: str, variant_name: str) -> dict[str, Any]:
        """Run a single query through a specific variant."""
        variant = self.variants[variant_name]
        t0 = time.perf_counter()
        result = variant.invoke({"query": query})
        elapsed = time.perf_counter() - t0
        result["latency_s"] = round(elapsed, 3)
        return result

    def run_all_queries(self, queries: list[dict] | None = None) -> list[dict]:
        """Run all test queries through all variants."""
        if queries is None:
            queries = FLOW_TEST_QUERIES

        all_results = []
        total = len(queries) * len(self.variants)
        count = 0

        for q in queries:
            for variant_name in self.variants:
                count += 1
                print(
                    f"  [{count:3d}/{total}] {variant_name:22s} | {q['query'][:50]}...",
                    end="",
                    flush=True,
                )

                try:
                    result = self.run_single_query(q["query"], variant_name)
                    result["query_id"] = q["id"]
                    result["query"] = q["query"]
                    result["expected_keywords"] = q["expected_keywords"]
                    result["domain"] = q["domain"]
                    all_results.append(result)
                    print(f"  ✓ {result['latency_s']}s ({len(result['response'])} chars)")
                except Exception as e:
                    print(f"  ✗ {e}")
                    all_results.append({
                        "variant": variant_name,
                        "query_id": q["id"],
                        "query": q["query"],
                        "response": f"ERROR: {e}",
                        "latency_s": 0,
                        "chain_steps": [],
                        "expected_keywords": q["expected_keywords"],
                        "domain": q["domain"],
                    })

        return all_results

    def compare_flows(self, results: list[dict]) -> dict:
        """Compare flow performance across variants."""
        import numpy as np

        comparison = {}
        for variant_name in self.variants:
            variant_results = [r for r in results if r["variant"] == variant_name]
            if not variant_results:
                continue

            latencies = [r["latency_s"] for r in variant_results]
            lengths = [len(r["response"]) for r in variant_results if r["response"] and not r["response"].startswith("ERROR")]

            # Keyword match rate
            kw_rates = []
            for r in variant_results:
                keywords = r.get("expected_keywords", [])
                if keywords and r["response"] and not r["response"].startswith("ERROR"):
                    text_lower = r["response"].lower()
                    rate = sum(1 for kw in keywords if kw.lower() in text_lower) / len(keywords)
                    kw_rates.append(rate)

            comparison[variant_name] = {
                "n_queries": len(variant_results),
                "avg_latency_s": round(float(np.mean(latencies)), 2) if latencies else 0,
                "avg_length": round(float(np.mean(lengths)), 0) if lengths else 0,
                "keyword_match_rate": round(float(np.mean(kw_rates)), 2) if kw_rates else 0,
                "errors": sum(1 for r in variant_results if r["response"].startswith("ERROR")),
            }

        return comparison


# ── CLI ───────────────────────────────────────────────────────────

def main():
    runner = FlowRunner()

    print(f"\n{'='*60}")
    print(f"  LANGCHAIN FLOW RUNNER")
    print(f"  {len(FLOW_TEST_QUERIES)} queries × {len(ALL_VARIANTS)} variants")
    print(f"{'='*60}\n")

    results = runner.run_all_queries()
    comparison = runner.compare_flows(results)

    # Save results
    results_path = RESULTS_DIR / "flow_run_results.json"
    # Truncate responses for storage
    save_results = []
    for r in results:
        save_r = dict(r)
        save_r["response"] = save_r.get("response", "")[:500]
        save_r.pop("intermediate", None)
        save_results.append(save_r)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    # Summary
    lines = []
    lines.append("=" * 60)
    lines.append("  FLOW COMPARISON — SUMMARY")
    lines.append("=" * 60)
    lines.append(f"\n{'Variant':<25} {'Latency':>8} {'Length':>8} {'KW Rate':>8} {'Errors':>7}")
    lines.append("-" * 58)
    for name, stats in comparison.items():
        lines.append(
            f"  {name:<23} {stats['avg_latency_s']:>7.1f}s {stats['avg_length']:>7.0f} "
            f"{stats['keyword_match_rate']:>7.0%} {stats['errors']:>6}"
        )

    summary_text = "\n".join(lines) + "\n"
    summary_path = RESULTS_DIR / "flow_run_summary.txt"
    summary_path.write_text(summary_text)

    print(f"\n{summary_text}")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
