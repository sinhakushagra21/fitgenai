#!/usr/bin/env python3
"""
flow_engineering/flow_evaluator.py
───────────────────────────────────
Automated evaluation pipeline for flow engineering experiments.

Runs all flow variants across test queries and measures:
  - Accuracy (keyword match rate vs expected content)
  - Coherence (LLM-as-judge 1-5 score)
  - Groundedness (citation count, evidence-based claims)
  - Latency (time per flow execution)
  - User satisfaction proxy (composite score)

Compares refined flows against the original baseline.

Outputs:
  flow_engineering/results/evaluation_report.csv
  flow_engineering/results/evaluation_summary.txt

Run:
    python -m flow_engineering.flow_evaluator
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flow_engineering.chain_variants import ALL_VARIANTS, VARIANT_DESCRIPTIONS
from flow_engineering.prompt_flow import FLOW_TEST_QUERIES

load_dotenv()

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4o-mini"

# ── LLM-as-Judge prompts ─────────────────────────────────────────

COHERENCE_JUDGE = """\
Rate the COHERENCE of this fitness AI response on a 1-5 scale.

1 = Disjointed, hard to follow, contradictory
2 = Some structure but unclear in places
3 = Reasonably coherent and organized
4 = Well-structured with clear logical flow
5 = Excellent structure, crystal clear, flows naturally

QUERY: {query}
RESPONSE: {response}

Return ONLY a JSON object: {{"score": <1-5>, "reason": "<brief reason>"}}
"""

GROUNDEDNESS_JUDGE = """\
Rate the GROUNDEDNESS of this fitness AI response on a 1-5 scale.

1 = No evidence, pure opinion or speculation
2 = Vague references to science but no specifics
3 = Some evidence-based claims with general principles
4 = Well-grounded with specific numbers, principles, and guidelines
5 = Excellent — cites studies, uses precise evidence, fully grounded

QUERY: {query}
RESPONSE: {response}

Return ONLY a JSON object: {{"score": <1-5>, "reason": "<brief reason>"}}
"""

SATISFACTION_JUDGE = """\
Rate the OVERALL USER SATISFACTION for this fitness AI response on a 1-5 scale.
Imagine you are a real user who asked this fitness question.

1 = Unhelpful, would not use this advice
2 = Somewhat useful but missing key information
3 = Decent — addresses the question adequately
4 = Good — practical, structured, and informative
5 = Excellent — would highly recommend, exceeds expectations

QUERY: {query}
RESPONSE: {response}

Return ONLY a JSON object: {{"score": <1-5>, "reason": "<brief reason>"}}
"""


class FlowEvaluator:
    """Evaluates flow variants with multi-dimensional metrics."""

    def __init__(self, model: str = MODEL):
        self.model = model
        self.judge_llm = ChatOpenAI(model=model, temperature=0)

    def _judge(self, prompt_template: str, query: str, response: str) -> dict:
        """Call LLM-as-judge and parse JSON score."""
        prompt = prompt_template.format(query=query, response=response)
        try:
            result = self.judge_llm.invoke([HumanMessage(content=prompt)])
            match = re.search(r"\{.*\}", result.content, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return {"score": 0, "reason": "judge_error"}

    def _keyword_match_rate(self, text: str, keywords: list[str]) -> float:
        """Fraction of expected keywords found in response."""
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        return round(matches / max(len(keywords), 1), 2)

    def _count_citations(self, text: str) -> int:
        """Count citation-like references."""
        return len(re.findall(r"\[\d+\]", text))

    def evaluate_single(self, query: str, response: str, keywords: list[str]) -> dict:
        """Evaluate a single response across all dimensions."""
        keyword_rate = self._keyword_match_rate(response, keywords)
        citations = self._count_citations(response)

        coherence = self._judge(COHERENCE_JUDGE, query, response)
        groundedness = self._judge(GROUNDEDNESS_JUDGE, query, response)
        satisfaction = self._judge(SATISFACTION_JUDGE, query, response)

        scores = {
            "keyword_rate": keyword_rate,
            "citations": citations,
            "coherence": coherence.get("score", 0),
            "groundedness": groundedness.get("score", 0),
            "satisfaction": satisfaction.get("score", 0),
            "response_length": len(response),
        }

        # Composite score (weighted average)
        weights = {"keyword_rate": 2, "coherence": 3, "groundedness": 3, "satisfaction": 4}
        weighted_sum = (
            scores["keyword_rate"] * 5 * weights["keyword_rate"]
            + scores["coherence"] * weights["coherence"]
            + scores["groundedness"] * weights["groundedness"]
            + scores["satisfaction"] * weights["satisfaction"]
        )
        total_weight = sum(weights.values())
        scores["composite"] = round(weighted_sum / total_weight, 2)

        return scores

    def evaluate_all(self, queries: list[dict] | None = None) -> list[dict]:
        """Run full evaluation across all variants and queries."""
        if queries is None:
            queries = FLOW_TEST_QUERIES

        all_rows = []
        total = len(queries) * len(ALL_VARIANTS)
        count = 0

        for q in queries:
            for variant_name, variant in ALL_VARIANTS.items():
                count += 1
                print(
                    f"  [{count:3d}/{total}] {variant_name:22s} | {q['query'][:45]}...",
                    end="",
                    flush=True,
                )

                try:
                    # Run variant
                    t0 = time.perf_counter()
                    result = variant.invoke({"query": q["query"]})
                    latency = time.perf_counter() - t0

                    # Evaluate
                    scores = self.evaluate_single(
                        q["query"], result["response"], q["expected_keywords"]
                    )
                    scores["latency_s"] = round(latency, 3)

                    row = {
                        "query_id": q["id"],
                        "query": q["query"],
                        "variant": variant_name,
                        "domain": q["domain"],
                        **scores,
                    }
                    all_rows.append(row)

                    print(
                        f"  comp={scores['composite']:.1f}  "
                        f"coh={scores['coherence']}  "
                        f"gnd={scores['groundedness']}  "
                        f"sat={scores['satisfaction']}  "
                        f"{latency:.1f}s"
                    )

                except Exception as e:
                    print(f"  ✗ {e}")
                    all_rows.append({
                        "query_id": q["id"],
                        "query": q["query"],
                        "variant": variant_name,
                        "domain": q["domain"],
                        "keyword_rate": 0,
                        "citations": 0,
                        "coherence": 0,
                        "groundedness": 0,
                        "satisfaction": 0,
                        "composite": 0,
                        "latency_s": 0,
                        "response_length": 0,
                    })

        return all_rows

    def generate_report(self, rows: list[dict]) -> str:
        """Generate summary report from evaluation results."""
        lines = []
        lines.append("=" * 70)
        lines.append("  FLOW ENGINEERING EVALUATION — FINAL REPORT")
        lines.append("=" * 70)

        # Per-variant aggregation
        lines.append(f"\n{'Variant':<25} {'Composite':>9} {'Coherence':>9} {'Grounded':>9} {'Satisf.':>9} {'KW Rate':>8} {'Latency':>8}")
        lines.append("-" * 80)

        variant_stats = {}
        for variant_name in ALL_VARIANTS:
            v_rows = [r for r in rows if r["variant"] == variant_name]
            if not v_rows:
                continue

            stats = {
                "composite": np.mean([r["composite"] for r in v_rows]),
                "coherence": np.mean([r["coherence"] for r in v_rows]),
                "groundedness": np.mean([r["groundedness"] for r in v_rows]),
                "satisfaction": np.mean([r["satisfaction"] for r in v_rows]),
                "keyword_rate": np.mean([r["keyword_rate"] for r in v_rows]),
                "latency_s": np.mean([r["latency_s"] for r in v_rows]),
            }
            variant_stats[variant_name] = stats

            lines.append(
                f"  {variant_name:<23} {stats['composite']:>8.2f} "
                f"{stats['coherence']:>8.1f} {stats['groundedness']:>8.1f} "
                f"{stats['satisfaction']:>8.1f} {stats['keyword_rate']:>7.0%} "
                f"{stats['latency_s']:>7.1f}s"
            )

        # Best variant
        if variant_stats:
            best = max(variant_stats, key=lambda k: variant_stats[k]["composite"])
            baseline_comp = variant_stats.get("A_vanilla", {}).get("composite", 0)
            best_comp = variant_stats[best]["composite"]
            improvement = ((best_comp - baseline_comp) / max(baseline_comp, 0.01)) * 100

            lines.append(f"\n── Best Performing Variant ──")
            lines.append(f"  {best}: composite={best_comp:.2f}")
            lines.append(f"  vs baseline (A_vanilla): {improvement:+.1f}% improvement")
            lines.append(f"  Description: {VARIANT_DESCRIPTIONS.get(best, '')}")

        # Per-domain breakdown
        lines.append(f"\n── Per Domain ──")
        for domain in ("workout", "diet", "both"):
            d_rows = [r for r in rows if r["domain"] == domain]
            if d_rows:
                avg_comp = np.mean([r["composite"] for r in d_rows])
                lines.append(f"  {domain:10s}  avg composite={avg_comp:.2f}  n={len(d_rows)}")

        return "\n".join(lines) + "\n"


# ── CLI ───────────────────────────────────────────────────────────

def main():
    evaluator = FlowEvaluator()

    print(f"\n{'='*60}")
    print(f"  FLOW ENGINEERING EVALUATION")
    print(f"  {len(FLOW_TEST_QUERIES)} queries × {len(ALL_VARIANTS)} variants")
    print(f"{'='*60}\n")

    rows = evaluator.evaluate_all()

    # Save CSV
    csv_path = RESULTS_DIR / "evaluation_report.csv"
    fieldnames = [
        "query_id", "variant", "domain", "composite", "coherence",
        "groundedness", "satisfaction", "keyword_rate", "citations",
        "latency_s", "response_length", "query",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    report = evaluator.generate_report(rows)
    summary_path = RESULTS_DIR / "evaluation_summary.txt"
    summary_path.write_text(report)

    print(f"\n{report}")
    print(f"Report CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
