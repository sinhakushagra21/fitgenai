#!/usr/bin/env python3
"""
fine_tuning/compare_models.py
──────────────────────────────
Compare baseline vs fine-tuned model.

Two evaluation modes:

  --task routing  (default, legacy)
      Tool-routing accuracy on data/test.jsonl + keyword matches.

  --task plan --domain {diet|workout}  (new)
      Reuses agent/shared/plan_evaluator to score plan generation on the
      held-out validation set mined from MongoDB:
        - hard-constraint pass rate
        - light rubric score (mean)
        - combined score (mean)

Outputs:
  fine_tuning/results/comparison_report.csv
  fine_tuning/results/comparison_summary.txt

Run:
    python -m fine_tuning.compare_models
    python -m fine_tuning.compare_models --task plan --domain diet
    python -m fine_tuning.compare_models --task plan --domain diet \
        --finetuned-model ft:gpt-4.1-mini:...
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
LEGACY_MODEL_ID_FILE = FT_DIR / "data" / "finetuned_model_id.txt"
MODEL_ID_REGISTRY = FT_DIR / "data" / "finetuned_model_ids.json"

from agent.config import DEFAULT_MODEL, FAST_MODEL

BASELINE_MODEL_ROUTING = DEFAULT_MODEL
BASELINE_MODEL_PLAN = DEFAULT_MODEL

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


def _resolve_finetuned_model(task: str, domain: str | None, cli_override: str | None) -> str | None:
    if cli_override:
        return cli_override
    if MODEL_ID_REGISTRY.exists():
        try:
            reg = json.loads(MODEL_ID_REGISTRY.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            reg = {}
        if task == "routing":
            model = reg.get("routing")
            if model:
                return model
        elif domain:
            model = (reg.get(domain) or {}).get(task)
            if model:
                return model
    # Legacy fallback for routing
    if task == "routing" and LEGACY_MODEL_ID_FILE.exists():
        return LEGACY_MODEL_ID_FILE.read_text().strip() or None
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Routing eval (legacy path)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_tool(response) -> str:
    if hasattr(response, "tool_calls") and response.tool_calls:
        return response.tool_calls[0]["name"]
    return "none"


def _evaluate_routing(model_name: str, test_data: list[dict]) -> list[dict]:
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Plan eval (new path — reuses agent/shared/plan_evaluator)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _evaluate_plan_generation(
    model_name: str,
    domain: str,
    val_rows: list[dict],
) -> list[dict]:
    """Generate a plan per val row and score it with the production evaluator."""
    from agent.shared.plan_evaluator import evaluate_plan  # lazy import

    results = []
    llm = ChatOpenAI(model=model_name, temperature=0.5)

    for i, row in enumerate(val_rows):
        msgs = row["messages"]
        system = msgs[0]["content"]
        user = msgs[1]["content"]

        # Recover profile dict from the user message (best-effort: the
        # miner serializes profile_snapshot as JSON in the prompt).
        profile = _extract_profile_from_user_msg(user)
        user_request = f"Create a personalized {domain} plan."

        print(f"    [{i+1:2d}/{len(val_rows)}] generating...", end="", flush=True)
        t0 = time.perf_counter()
        try:
            resp = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            plan_md = resp.content or ""
            elapsed = time.perf_counter() - t0

            eval_result = evaluate_plan(
                plan_md,
                domain=domain,
                profile=profile,
                user_request=user_request,
            )
            results.append({
                "model": model_name,
                "domain": domain,
                "hard_passed": int(eval_result.hard.passed),
                "light_score": round(eval_result.light.score, 4),
                "combined_score": round(eval_result.combined_score, 4),
                "plan_length": len(plan_md),
                "latency_s": round(elapsed, 3),
                "hard_reasons": "; ".join(eval_result.hard.reasons)[:300],
            })
            ok = "✓" if eval_result.hard.passed else "✗"
            print(f"  hard={ok} light={eval_result.light.score:.2f} {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  ❌ {e}")
            results.append({
                "model": model_name,
                "domain": domain,
                "hard_passed": 0,
                "light_score": 0.0,
                "combined_score": 0.0,
                "plan_length": 0,
                "latency_s": round(elapsed, 3),
                "hard_reasons": f"error: {e}"[:300],
            })

    return results


def _extract_profile_from_user_msg(user_msg: str) -> dict:
    """Best-effort parse of the JSON profile embedded in the prompt."""
    start = user_msg.find("{")
    end = user_msg.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(user_msg[start:end + 1])
    except json.JSONDecodeError:
        return {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs fine-tuned model")
    parser.add_argument(
        "--task",
        choices=["routing", "plan"],
        default="routing",
        help="Which task to evaluate (default: routing).",
    )
    parser.add_argument("--domain", choices=["diet", "workout"], default=None,
                        help="Required when --task=plan.")
    parser.add_argument(
        "--finetuned-model",
        type=str,
        default=None,
        help="Fine-tuned model ID (reads from finetuned_model_ids.json if unset).",
    )
    args = parser.parse_args()

    ft_model = _resolve_finetuned_model(args.task, args.domain, args.finetuned_model)
    if not ft_model:
        print("⚠️  No fine-tuned model found. Running baseline-only evaluation.")

    if args.task == "routing":
        baseline_model = BASELINE_MODEL_ROUTING
        test_data = _load_jsonl(TEST_FILE)
        print(f"\nTest set: {len(test_data)} examples\n")

        print(f"═══ Evaluating BASELINE: {baseline_model} ═══")
        baseline_results = _evaluate_routing(baseline_model, test_data)

        ft_results = []
        if ft_model:
            print(f"\n═══ Evaluating FINE-TUNED: {ft_model} ═══")
            ft_results = _evaluate_routing(ft_model, test_data)

        _write_routing_report(baseline_model, baseline_results, ft_model, ft_results)
        return

    # ── task == "plan" ──
    if not args.domain:
        raise SystemExit("--domain is required when --task=plan")

    val_path = FT_DIR / "data" / f"{args.domain}_plan_val.jsonl"
    if not val_path.exists():
        raise SystemExit(
            f"❌ {val_path} not found. Run: python -m fine_tuning.mine_plan_training_data "
            f"--domain {args.domain} --task plan"
        )
    val_rows = _load_jsonl(val_path)
    print(f"\nValidation set: {len(val_rows)} plans (domain={args.domain})\n")

    baseline_model = BASELINE_MODEL_PLAN
    print(f"═══ Evaluating BASELINE: {baseline_model} ═══")
    baseline_results = _evaluate_plan_generation(baseline_model, args.domain, val_rows)

    ft_results = []
    if ft_model:
        print(f"\n═══ Evaluating FINE-TUNED: {ft_model} ═══")
        ft_results = _evaluate_plan_generation(ft_model, args.domain, val_rows)

    _write_plan_report(baseline_model, baseline_results, ft_model, ft_results, args.domain)


def _write_routing_report(baseline_model, baseline_results, ft_model, ft_results):
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

    lines = ["=" * 60, "  MODEL COMPARISON REPORT (routing)", "=" * 60]

    for model_name, results in [(baseline_model, baseline_results)] + (
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

    if ft_results:
        bl_acc = sum(r["routing_correct"] for r in baseline_results) / max(len(baseline_results), 1)
        ft_acc = sum(r["routing_correct"] for r in ft_results) / max(len(ft_results), 1)
        lines.append(f"\n  Delta: {(ft_acc - bl_acc):+.1%}")

    summary_text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "comparison_summary.txt").write_text(summary_text)
    print(f"\n{summary_text}\nSaved to: {csv_path}")


def _write_plan_report(baseline_model, baseline_results, ft_model, ft_results, domain):
    all_results = baseline_results + ft_results
    csv_path = RESULTS_DIR / f"plan_comparison_{domain}.csv"
    fieldnames = [
        "model", "domain", "hard_passed", "light_score", "combined_score",
        "plan_length", "latency_s", "hard_reasons",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    lines = ["=" * 60, f"  PLAN COMPARISON REPORT (domain={domain})", "=" * 60]
    for model_name, results in [(baseline_model, baseline_results)] + (
        [(ft_model, ft_results)] if ft_results else []
    ):
        if not results:
            continue
        lines.append(f"\n── {model_name} ──")
        n = len(results)
        hard_rate = sum(r["hard_passed"] for r in results) / max(n, 1)
        avg_light = float(np.mean([r["light_score"] for r in results]))
        avg_comb = float(np.mean([r["combined_score"] for r in results]))
        avg_latency = float(np.mean([r["latency_s"] for r in results]))
        lines.append(f"  Hard pass rate:      {hard_rate:.1%}")
        lines.append(f"  Avg light score:     {avg_light:.3f}")
        lines.append(f"  Avg combined score:  {avg_comb:.3f}")
        lines.append(f"  Avg latency:         {avg_latency:.2f}s")

    if ft_results:
        bl_hard = sum(r["hard_passed"] for r in baseline_results) / max(len(baseline_results), 1)
        ft_hard = sum(r["hard_passed"] for r in ft_results) / max(len(ft_results), 1)
        bl_light = float(np.mean([r["light_score"] for r in baseline_results]))
        ft_light = float(np.mean([r["light_score"] for r in ft_results]))
        lines.append(f"\n  Hard-pass delta:  {(ft_hard - bl_hard):+.1%}")
        lines.append(f"  Light-score delta: {(ft_light - bl_light):+.3f}")

    summary_text = "\n".join(lines) + "\n"
    (RESULTS_DIR / f"plan_summary_{domain}.txt").write_text(summary_text)
    print(f"\n{summary_text}\nSaved to: {csv_path}")


if __name__ == "__main__":
    main()
