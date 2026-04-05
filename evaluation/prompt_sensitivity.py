#!/usr/bin/env python3
"""
evaluation/prompt_sensitivity.py
─────────────────────────────────
Step 1 — Prompt Sensitivity Optimization.

Tests how sensitive model responses are to:
  1. Phrasings of the SAME query  (paraphrase stability)
  2. Temperature settings          (stochastic stability)
  3. Different prompting techniques (technique variance)

Metrics:
  • Routing accuracy       — did the model pick the right tool?
  • Cosine similarity      — how consistent are responses across variants?
  • Response length delta  — stability of output length

Outputs:
  evaluation/results/sensitivity_report.csv
  evaluation/results/sensitivity_summary.txt

Run:
    python -m evaluation.prompt_sensitivity          (small sample, ~50 calls)
    python -m evaluation.prompt_sensitivity --full   (full sweep,  ~300 calls)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from itertools import product

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Project imports ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.prompts.base_prompts import BASE_PROMPTS
from agent.prompts.techniques import TECHNIQUE_KEYS

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from agent.config import DEFAULT_MODEL

MODEL = DEFAULT_MODEL
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]

# Representative queries with paraphrase variants
QUERY_VARIANTS: list[dict] = [
    {
        "id": "workout_plan",
        "expected_tool": "workout_tool",
        "variants": [
            "Give me a 4-day upper/lower split for hypertrophy.",
            "Can you design a four-day upper lower workout split focused on building muscle?",
            "I need a hypertrophy-oriented training program using an upper/lower split, 4 days per week.",
        ],
    },
    {
        "id": "diet_plan",
        "expected_tool": "diet_tool",
        "variants": [
            "Create a 2000-calorie meal plan for fat loss.",
            "I want a meal plan with 2000 calories per day aimed at losing fat.",
            "Design a fat-loss diet with a daily target of two thousand calories.",
        ],
    },
    {
        "id": "protein_intake",
        "expected_tool": "diet_tool",
        "variants": [
            "How much protein should I eat to build muscle?",
            "What's the ideal daily protein intake for muscle growth?",
            "How many grams of protein per day do I need for gaining muscle mass?",
        ],
    },
    {
        "id": "warm_up",
        "expected_tool": "workout_tool",
        "variants": [
            "What's the best warm-up routine before heavy squats?",
            "How should I warm up before doing heavy barbell squats?",
            "Give me a warm-up protocol specifically for heavy squat sessions.",
        ],
    },
    {
        "id": "off_topic",
        "expected_tool": "none",
        "variants": [
            "What stocks should I invest in?",
            "Can you recommend some good stock market investments?",
            "I need financial advice on stock picks.",
        ],
    },
]

# ── Helpers ───────────────────────────────────────────────────────

_TOOL_RELAY = """
When responding, if you decide to use a tool, respond with the tool call.
If the query is not fitness-related, politely redirect to fitness topics.
"""


def _extract_tool_call(response) -> str:
    """Return tool name from response or 'none'."""
    if hasattr(response, "tool_calls") and response.tool_calls:
        return response.tool_calls[0]["name"]
    return "none"


def _call_llm(
    technique: str,
    query: str,
    temperature: float,
) -> dict:
    """Call the LLM with a specific technique prompt and return metadata."""
    from agent.tools import ALL_TOOLS

    prompt_text = BASE_PROMPTS[technique] + _TOOL_RELAY
    llm = ChatOpenAI(model=MODEL, temperature=temperature)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    t0 = time.perf_counter()
    response = llm_with_tools.invoke(
        [SystemMessage(content=prompt_text), HumanMessage(content=query)]
    )
    elapsed = time.perf_counter() - t0

    tool_called = _extract_tool_call(response)
    content = response.content or ""

    return {
        "technique": technique,
        "query": query,
        "temperature": temperature,
        "tool_called": tool_called,
        "response_text": content,
        "response_length": len(content),
        "latency_s": round(elapsed, 3),
    }


def _pairwise_cosine(texts: list[str]) -> float:
    """Average pairwise cosine similarity among texts (TF-IDF based)."""
    if len(texts) < 2:
        return 1.0
    # Filter out empty strings
    non_empty = [t for t in texts if t.strip()]
    if len(non_empty) < 2:
        return 1.0
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(non_empty)
    sim_matrix = cosine_similarity(tfidf)
    # Get upper-triangle (exclude diagonal)
    n = sim_matrix.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(sim_matrix[i, j])
    return round(float(np.mean(pairs)), 4) if pairs else 1.0


# ── Main Sweep ────────────────────────────────────────────────────

def run_sensitivity(full: bool = False) -> Path:
    """Execute the prompt sensitivity sweep and save results."""

    # In --full mode: all techniques × all temps × all queries
    # In default mode: subset for quick testing
    techniques = TECHNIQUE_KEYS if full else ["zero_shot", "cot", "few_shot"]
    temps = TEMPERATURES if full else [0.0, 0.7]
    queries = QUERY_VARIANTS if full else QUERY_VARIANTS[:3]

    total_calls = sum(
        len(q["variants"]) * len(techniques) * len(temps) for q in queries
    )
    print(f"\n{'='*60}")
    print(f"  PROMPT SENSITIVITY SWEEP")
    print(f"  Techniques: {techniques}")
    print(f"  Temperatures: {temps}")
    print(f"  Query groups: {len(queries)} ({sum(len(q['variants']) for q in queries)} variants)")
    print(f"  Total LLM calls: {total_calls}")
    print(f"{'='*60}\n")

    all_rows: list[dict] = []
    call_count = 0

    for q_group in queries:
        q_id = q_group["id"]
        expected = q_group["expected_tool"]

        for technique in techniques:
            for temp in temps:
                for variant in q_group["variants"]:
                    call_count += 1
                    print(f"  [{call_count:3d}/{total_calls}] {technique:22s} T={temp}  {variant[:50]}...", end="")

                    try:
                        result = _call_llm(technique, variant, temp)
                    except Exception as e:
                        print(f"  ❌ {e}")
                        result = {
                            "technique": technique,
                            "query": variant,
                            "temperature": temp,
                            "tool_called": "error",
                            "response_text": str(e),
                            "response_length": 0,
                            "latency_s": 0,
                        }

                    result["query_group"] = q_id
                    result["expected_tool"] = expected
                    result["routing_correct"] = int(result["tool_called"] == expected)
                    all_rows.append(result)
                    print(f"  → {result['tool_called']}  ({'✓' if result['routing_correct'] else '✗'})  {result['latency_s']}s")

    # ── Save raw results ──────────────────────────────────────────
    csv_path = RESULTS_DIR / "sensitivity_report.csv"
    fieldnames = [
        "query_group", "technique", "temperature", "query",
        "expected_tool", "tool_called", "routing_correct",
        "response_length", "latency_s", "response_text",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # ── Compute aggregated metrics ────────────────────────────────
    summary_lines: list[str] = []
    summary_lines.append("=" * 60)
    summary_lines.append("  PROMPT SENSITIVITY — SUMMARY REPORT")
    summary_lines.append("=" * 60)

    # 1. Overall routing accuracy
    correct_total = sum(r["routing_correct"] for r in all_rows)
    total = len(all_rows)
    summary_lines.append(f"\nOverall Routing Accuracy: {correct_total}/{total} = {correct_total/total:.1%}")

    # 2. Routing accuracy by technique
    summary_lines.append("\n── Routing Accuracy by Technique ──")
    for tech in techniques:
        rows_t = [r for r in all_rows if r["technique"] == tech]
        correct = sum(r["routing_correct"] for r in rows_t)
        summary_lines.append(f"  {tech:22s}  {correct}/{len(rows_t)} = {correct/len(rows_t):.1%}")

    # 3. Routing accuracy by temperature
    summary_lines.append("\n── Routing Accuracy by Temperature ──")
    for temp in temps:
        rows_t = [r for r in all_rows if r["temperature"] == temp]
        correct = sum(r["routing_correct"] for r in rows_t)
        summary_lines.append(f"  T={temp:<4}  {correct}/{len(rows_t)} = {correct/len(rows_t):.1%}")

    # 4. Paraphrase consistency (cosine similarity within each group)
    summary_lines.append("\n── Paraphrase Consistency (Cosine Sim) ──")
    for q_group in queries:
        q_id = q_group["id"]
        for tech in techniques:
            for temp in temps:
                texts = [
                    r["response_text"]
                    for r in all_rows
                    if r["query_group"] == q_id
                    and r["technique"] == tech
                    and r["temperature"] == temp
                ]
                sim = _pairwise_cosine(texts)
                summary_lines.append(f"  {q_id:16s} {tech:22s} T={temp:<4}  sim={sim:.4f}")

    # 5. Response length variability
    summary_lines.append("\n── Response Length Stats by Technique ──")
    for tech in techniques:
        lengths = [r["response_length"] for r in all_rows if r["technique"] == tech]
        if lengths:
            summary_lines.append(
                f"  {tech:22s}  mean={np.mean(lengths):.0f}  "
                f"std={np.std(lengths):.0f}  "
                f"min={min(lengths)}  max={max(lengths)}"
            )

    # 6. Average latency
    summary_lines.append("\n── Average Latency by Technique ──")
    for tech in techniques:
        lats = [r["latency_s"] for r in all_rows if r["technique"] == tech]
        if lats:
            summary_lines.append(f"  {tech:22s}  {np.mean(lats):.2f}s avg")

    # 7. Identify the most robust technique
    summary_lines.append("\n── Recommendation ──")
    best_tech = None
    best_score = -1
    for tech in techniques:
        rows_t = [r for r in all_rows if r["technique"] == tech]
        acc = sum(r["routing_correct"] for r in rows_t) / max(len(rows_t), 1)
        if acc > best_score:
            best_score = acc
            best_tech = tech
    summary_lines.append(f"  Most robust technique: {best_tech} ({best_score:.1%} routing accuracy)")

    summary_text = "\n".join(summary_lines) + "\n"
    summary_path = RESULTS_DIR / "sensitivity_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n{summary_text}")
    print(f"Results saved to:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")

    return csv_path


# ── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FITGEN.AI Prompt Sensitivity Analysis")
    parser.add_argument("--full", action="store_true", help="Run full sweep (all techniques × temps × queries)")
    args = parser.parse_args()
    run_sensitivity(full=args.full)


if __name__ == "__main__":
    main()
