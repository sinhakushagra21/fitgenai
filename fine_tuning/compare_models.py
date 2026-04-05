#!/usr/bin/env python3
"""
fine_tuning/compare_models.py
──────────────────────────────
Compare baseline (gpt-4o-mini) vs fine-tuned model on the test set.

Metrics:
  • Routing accuracy  — tool selection correctness
  • Latency           — average response time
  • Content quality   — keyword match rate from expected_response_contains

Outputs:
  fine_tuning/results/comparison_report.csv
  fine_tuning/results/comparison_summary.txt

Run:
    python -m fine_tuning.compare_models
    python -m fine_tuning.compare_models --finetuned-model ft:gpt-4o-mini:...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.prompts.base_prompts import BASE_ZERO_SHOT
from agent.tools import ALL_TOOLS

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = FT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_FILE = PROJECT_ROOT / "data" / "test.jsonl"
MODEL_ID_FILE = FT_DIR / "data" / "finetuned_model_id.txt"

from agent.config import DEFAULT_MODEL

BASELINE_MODEL = DEFAULT_MODEL

SYSTEM_PROMPT = BASE_ZERO_SHOT + """

You have access to two tools:
- workout_tool: for exercise, training, and workout queries
- diet_tool: for nutrition, diet, and meal planning queries

If the query is not fitness-related, respond politely without calling a tool.
"""


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _extract_tool(response) -> str:
    if hasattr(response, "tool_calls") and response.tool_calls:
        return response.tool_calls[0]["name"]
    return "none"


def _evaluate_model(model_name: str, test_data: list[dict]) -> list[dict]:
    """Run a model through the test set and collect metrics."""
    results = []
    llm = ChatOpenAI(model=model_name, temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    for i, example in enumerate(test_data):
        query = example["query"]
        expected_tool = example["expected_tool"]
        expected_contains = example.get("expected_response_contains", [])

        print(f"    [{i+1:2d}/{len(test_data)}] {query[:60]}...", end="", flush=True)

        t0 = time.perf_counter()
        try:
            response = llm_with_tools.invoke([
                SystemMessage(content=SYSTEM_PROMPT.strip()),
                HumanMessage(content=query),
            ])
            elapsed = time.perf_counter() - t0
            tool_called = _extract_tool(response)
            content = response.content or ""

            # Check keyword matches
            content_lower = content.lower()
            matches = sum(1 for kw in expected_contains if kw.lower() in content_lower)
            keyword_rate = matches / max(len(expected_contains), 1)

            results.append({
                "model": model_name,
                "query": query,
                "category": example.get("category", ""),
                "expected_tool": expected_tool,
                "tool_called": tool_called,
                "routing_correct": int(tool_called == expected_tool),
                "keyword_match_rate": round(keyword_rate, 2),
                "response_length": len(content),
                "latency_s": round(elapsed, 3),
            })
            routing_ok = "✓" if tool_called == expected_tool else "✗"
            print(f"  → {tool_called} ({routing_ok}) {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  ❌ {e}")
            results.append({
                "model": model_name,
                "query": query,
                "category": example.get("category", ""),
                "expected_tool": expected_tool,
                "tool_called": "error",
                "routing_correct": 0,
                "keyword_match_rate": 0,
                "response_length": 0,
                "latency_s": round(elapsed, 3),
            })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs fine-tuned model")
    parser.add_argument(
        "--finetuned-model",
        type=str,
        default=None,
        help="Fine-tuned model ID (reads from finetuned_model_id.txt if not provided)",
    )
    args = parser.parse_args()

    # Resolve fine-tuned model ID
    ft_model = args.finetuned_model
    if not ft_model and MODEL_ID_FILE.exists():
        ft_model = MODEL_ID_FILE.read_text().strip()
    if not ft_model:
        print("⚠️  No fine-tuned model found. Running baseline-only evaluation.")
        print("   (Run run_finetune.py first, or pass --finetuned-model)")

    # Load test data
    test_data = _load_jsonl(TEST_FILE)
    print(f"\nTest set: {len(test_data)} examples\n")

    # Evaluate baseline
    print(f"═══ Evaluating BASELINE: {BASELINE_MODEL} ═══")
    baseline_results = _evaluate_model(BASELINE_MODEL, test_data)

    # Evaluate fine-tuned (if available)
    ft_results = []
    if ft_model:
        print(f"\n═══ Evaluating FINE-TUNED: {ft_model} ═══")
        ft_results = _evaluate_model(ft_model, test_data)

    # ── Save raw results ──────────────────────────────────────────
    all_results = baseline_results + ft_results
    csv_path = RESULTS_DIR / "comparison_report.csv"
    fieldnames = [
        "model", "query", "category", "expected_tool", "tool_called",
        "routing_correct", "keyword_match_rate", "response_length", "latency_s",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # ── Generate summary ──────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append("  MODEL COMPARISON REPORT")
    lines.append("=" * 60)

    for model_name, results in [(BASELINE_MODEL, baseline_results)] + (
        [(ft_model, ft_results)] if ft_results else []
    ):
        lines.append(f"\n── {model_name} ──")
        n = len(results)
        routing_acc = sum(r["routing_correct"] for r in results) / max(n, 1)
        avg_latency = np.mean([r["latency_s"] for r in results])
        avg_kw_rate = np.mean([r["keyword_match_rate"] for r in results])

        lines.append(f"  Routing accuracy:    {routing_acc:.1%}")
        lines.append(f"  Avg latency:         {avg_latency:.2f}s")
        lines.append(f"  Avg keyword match:   {avg_kw_rate:.1%}")

        # Per-category breakdown
        for cat in ("typical", "edge", "adversarial"):
            cat_results = [r for r in results if r["category"] == cat]
            if cat_results:
                cat_acc = sum(r["routing_correct"] for r in cat_results) / len(cat_results)
                lines.append(f"  [{cat:12s}] accuracy: {cat_acc:.1%} ({len(cat_results)} examples)")

    if ft_results:
        bl_acc = sum(r["routing_correct"] for r in baseline_results) / max(len(baseline_results), 1)
        ft_acc = sum(r["routing_correct"] for r in ft_results) / max(len(ft_results), 1)
        delta = ft_acc - bl_acc
        lines.append(f"\n── Improvement ──")
        lines.append(f"  Routing accuracy delta: {delta:+.1%}")
        lines.append(f"  Fine-tuning {'improved' if delta > 0 else 'did not improve'} routing.")

    summary_text = "\n".join(lines) + "\n"
    summary_path = RESULTS_DIR / "comparison_summary.txt"
    summary_path.write_text(summary_text)

    print(f"\n{summary_text}")
    print(f"Results saved to:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
