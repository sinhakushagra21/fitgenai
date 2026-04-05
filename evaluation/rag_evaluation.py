#!/usr/bin/env python3
"""
evaluation/rag_evaluation.py
──────────────────────────────
Evaluate the RAG pipeline against the baseline (non-RAG) responses.

Compares:
  1. Factual grounding — does the response cite sources?
  2. Keyword coverage  — does it contain expected keywords?
  3. Response quality   — length, structure, citations

Outputs:
  evaluation/results/rag_comparison.csv
  evaluation/results/rag_summary.txt

Run:
    python -m evaluation.rag_evaluation
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.rag.retriever import retrieve, format_context
from agent.prompts.base_prompts import BASE_ZERO_SHOT

load_dotenv()

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from agent.config import DEFAULT_MODEL

MODEL = DEFAULT_MODEL

# Queries specifically designed to test RAG benefit
RAG_TEST_QUERIES = [
    {
        "query": "How much protein should I eat per day to build muscle?",
        "expected_keywords": ["1.6", "2.2", "g/kg", "protein synthesis", "ISSN"],
        "topic": "diet",
    },
    {
        "query": "What are the best rep ranges for hypertrophy?",
        "expected_keywords": ["6-12", "reps", "failure", "volume", "sets"],
        "topic": "workout",
    },
    {
        "query": "How long should I rest between sets for strength training?",
        "expected_keywords": ["3-5 minutes", "rest", "compound"],
        "topic": "workout",
    },
    {
        "query": "Is creatine safe and how much should I take?",
        "expected_keywords": ["3-5 g", "creatine monohydrate", "safe", "ISSN"],
        "topic": "diet",
    },
    {
        "query": "What is progressive overload and how do I implement it?",
        "expected_keywords": ["progressive overload", "weight", "reps", "sets"],
        "topic": "workout",
    },
    {
        "query": "How do I calculate my caloric needs for fat loss?",
        "expected_keywords": ["TDEE", "Mifflin", "deficit", "500"],
        "topic": "diet",
    },
    {
        "query": "Should I do HIIT or steady-state cardio for fat loss?",
        "expected_keywords": ["HIIT", "fat loss", "steady-state"],
        "topic": "workout",
    },
    {
        "query": "What should I eat before and after a workout?",
        "expected_keywords": ["protein", "carbs", "pre-workout", "post-workout"],
        "topic": "diet",
    },
    {
        "query": "How important is sleep for muscle recovery?",
        "expected_keywords": ["7-9 hours", "sleep", "recovery", "cortisol"],
        "topic": "general",
    },
    {
        "query": "What are good protein sources for vegans?",
        "expected_keywords": ["soy", "lentils", "tofu", "B12"],
        "topic": "diet",
    },
]


def _count_citations(text: str) -> int:
    """Count citation-like references [1], [2], etc."""
    return len(re.findall(r"\[\d+\]", text))


def _keyword_match_rate(text: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found in text."""
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches / max(len(keywords), 1)


def _call_baseline(query: str) -> dict:
    """Call the baseline model without RAG context."""
    llm = ChatOpenAI(model=MODEL, temperature=0.3)
    t0 = time.perf_counter()
    response = llm.invoke([
        SystemMessage(content=BASE_ZERO_SHOT),
        HumanMessage(content=query),
    ])
    elapsed = time.perf_counter() - t0
    return {
        "text": response.content,
        "latency_s": round(elapsed, 3),
    }


def _call_rag(query: str) -> dict:
    """Call the model with RAG-augmented context."""
    docs = retrieve(query, k=3)
    context = format_context(docs)
    rag_prompt = (
        BASE_ZERO_SHOT
        + "\n\nUse the following retrieved evidence to ground your response. "
        "Cite sources using [1], [2], etc.\n\n"
        + context
    )
    llm = ChatOpenAI(model=MODEL, temperature=0.3)
    t0 = time.perf_counter()
    response = llm.invoke([
        SystemMessage(content=rag_prompt),
        HumanMessage(content=query),
    ])
    elapsed = time.perf_counter() - t0
    return {
        "text": response.content,
        "latency_s": round(elapsed, 3),
        "sources": [{"title": d["title"], "score": d["score"]} for d in docs],
    }


def main() -> None:
    print(f"\n{'='*60}")
    print(f"  RAG vs BASELINE EVALUATION")
    print(f"  {len(RAG_TEST_QUERIES)} queries × 2 modes")
    print(f"{'='*60}\n")

    rows = []

    for i, q in enumerate(RAG_TEST_QUERIES, 1):
        query = q["query"]
        keywords = q["expected_keywords"]
        print(f"[{i:2d}/{len(RAG_TEST_QUERIES)}] {query[:60]}...")

        # Baseline
        print(f"  Baseline...", end="", flush=True)
        baseline = _call_baseline(query)
        bl_kw = _keyword_match_rate(baseline["text"], keywords)
        bl_cit = _count_citations(baseline["text"])
        print(f"  ✓ ({baseline['latency_s']}s, kw={bl_kw:.0%}, cit={bl_cit})")

        # RAG
        print(f"  RAG...", end="", flush=True)
        rag = _call_rag(query)
        rag_kw = _keyword_match_rate(rag["text"], keywords)
        rag_cit = _count_citations(rag["text"])
        print(f"  ✓ ({rag['latency_s']}s, kw={rag_kw:.0%}, cit={rag_cit})")

        rows.append({
            "query": query,
            "topic": q["topic"],
            "baseline_keyword_rate": round(bl_kw, 2),
            "rag_keyword_rate": round(rag_kw, 2),
            "baseline_citations": bl_cit,
            "rag_citations": rag_cit,
            "baseline_length": len(baseline["text"]),
            "rag_length": len(rag["text"]),
            "baseline_latency_s": baseline["latency_s"],
            "rag_latency_s": rag["latency_s"],
            "keyword_improvement": round(rag_kw - bl_kw, 2),
        })

    # Save CSV
    csv_path = RESULTS_DIR / "rag_comparison.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    lines = []
    lines.append("=" * 60)
    lines.append("  RAG vs BASELINE — SUMMARY")
    lines.append("=" * 60)

    bl_kw_avg = np.mean([r["baseline_keyword_rate"] for r in rows])
    rag_kw_avg = np.mean([r["rag_keyword_rate"] for r in rows])
    bl_cit_avg = np.mean([r["baseline_citations"] for r in rows])
    rag_cit_avg = np.mean([r["rag_citations"] for r in rows])
    bl_lat_avg = np.mean([r["baseline_latency_s"] for r in rows])
    rag_lat_avg = np.mean([r["rag_latency_s"] for r in rows])

    lines.append(f"\n{'Metric':<30} {'Baseline':>10} {'RAG':>10} {'Delta':>10}")
    lines.append("-" * 62)
    lines.append(f"{'Keyword match rate':<30} {bl_kw_avg:>10.1%} {rag_kw_avg:>10.1%} {rag_kw_avg-bl_kw_avg:>+10.1%}")
    lines.append(f"{'Avg citations per response':<30} {bl_cit_avg:>10.1f} {rag_cit_avg:>10.1f} {rag_cit_avg-bl_cit_avg:>+10.1f}")
    lines.append(f"{'Avg latency (s)':<30} {bl_lat_avg:>10.2f} {rag_lat_avg:>10.2f} {rag_lat_avg-bl_lat_avg:>+10.2f}")

    improved = sum(1 for r in rows if r["keyword_improvement"] > 0)
    same = sum(1 for r in rows if r["keyword_improvement"] == 0)
    worse = sum(1 for r in rows if r["keyword_improvement"] < 0)
    lines.append(f"\nKeyword improvement: {improved} better, {same} same, {worse} worse")

    summary_text = "\n".join(lines) + "\n"
    summary_path = RESULTS_DIR / "rag_summary.txt"
    summary_path.write_text(summary_text)

    print(f"\n{summary_text}")
    print(f"Results: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
