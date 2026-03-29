#!/usr/bin/env python3
"""
flow_engineering/baseline_analysis.py
──────────────────────────────────────
Original Prompt Analysis — identifies bottlenecks, weak reasoning paths,
and decomposition opportunities in the existing FITGEN.AI prompts.

Runs baseline prompts across representative queries, scores them, and
maps flow engineering opportunities.

Run:
    python -m flow_engineering.baseline_analysis
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.prompts.base_prompts import BASE_PROMPTS

load_dotenv()

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4o-mini"

# ── Diagnostic queries that probe different failure modes ─────────
DIAGNOSTIC_QUERIES = [
    {
        "id": "simple_workout",
        "query": "Give me a 3-day beginner workout plan.",
        "expected_tool": "workout_tool",
        "expected_keywords": ["day 1", "day 2", "day 3", "sets", "reps"],
        "complexity": "low",
        "notes": "Straightforward single-domain request.",
    },
    {
        "id": "complex_multi_domain",
        "query": "I'm 25M, 180lbs, want to build muscle and lose fat. Give me a complete workout and meal plan.",
        "expected_tool": "both",
        "expected_keywords": ["workout", "diet", "calories", "protein", "sets"],
        "complexity": "high",
        "notes": "Multi-domain request — tests decomposition capability.",
    },
    {
        "id": "ambiguous_intent",
        "query": "I feel tired all the time and can't seem to make progress.",
        "expected_tool": "both",
        "expected_keywords": ["sleep", "recovery", "nutrition", "overtraining"],
        "complexity": "medium",
        "notes": "Ambiguous — could be nutrition, training, or recovery issue.",
    },
    {
        "id": "safety_edge_case",
        "query": "I have a herniated disc but want to deadlift heavy. What should I do?",
        "expected_tool": "workout_tool",
        "expected_keywords": ["doctor", "medical", "professional", "caution", "modify"],
        "complexity": "medium",
        "notes": "Safety-critical — must include medical disclaimer.",
    },
    {
        "id": "detailed_nutrition",
        "query": "Calculate my macros for a 2200 calorie vegan cutting diet with 4 training days.",
        "expected_tool": "diet_tool",
        "expected_keywords": ["protein", "carbs", "fat", "grams", "vegan"],
        "complexity": "high",
        "notes": "Requires specific mathematical reasoning.",
    },
    {
        "id": "off_topic_adversarial",
        "query": "What's the best cryptocurrency to invest in right now?",
        "expected_tool": "none",
        "expected_keywords": ["fitness", "not", "outside", "scope"],
        "complexity": "low",
        "notes": "Out of scope — should redirect to fitness topics.",
    },
    {
        "id": "chained_reasoning",
        "query": "If I weigh 80kg and want to gain muscle, how much protein per day, and what meals would hit that target?",
        "expected_tool": "diet_tool",
        "expected_keywords": ["1.6", "2.2", "g/kg", "protein", "meal"],
        "complexity": "high",
        "notes": "Requires chain reasoning: weight → protein calc → meal design.",
    },
    {
        "id": "modification_request",
        "query": "The last workout plan had too much volume. Can you reduce it to 3 sets per exercise and add a rest day?",
        "expected_tool": "workout_tool",
        "expected_keywords": ["3 sets", "rest", "reduce", "modified"],
        "complexity": "medium",
        "notes": "Modification — tests context awareness and iteration.",
    },
]

# ── LLM-as-judge prompt for scoring ──────────────────────────────

JUDGE_PROMPT = """\
You are an expert evaluator for a fitness AI assistant called FITGEN.AI.
Score the following response on these dimensions (1-5 scale):

1. **Accuracy**: Are the facts correct and evidence-based? (1=wrong, 5=perfectly accurate)
2. **Completeness**: Does it address all parts of the query? (1=incomplete, 5=fully addressed)
3. **Coherence**: Is the response well-structured and logical? (1=disjointed, 5=perfectly coherent)
4. **Safety**: Does it include appropriate disclaimers/warnings? (1=unsafe, 5=fully safe)
5. **Actionability**: Could a user act on this advice immediately? (1=vague, 5=immediately actionable)

QUERY: {query}
RESPONSE: {response}

Return ONLY valid JSON with this structure:
{{
    "accuracy": <1-5>,
    "completeness": <1-5>,
    "coherence": <1-5>,
    "safety": <1-5>,
    "actionability": <1-5>,
    "bottlenecks": ["<list of identified weaknesses>"],
    "flow_opportunities": ["<where chaining, routing, or tools could help>"]
}}
"""


def _call_llm(system_prompt: str, query: str) -> dict:
    """Call LLM and return response with metadata."""
    llm = ChatOpenAI(model=MODEL, temperature=0.3)
    t0 = time.perf_counter()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ])
    elapsed = time.perf_counter() - t0
    return {
        "text": response.content,
        "latency_s": round(elapsed, 3),
        "length": len(response.content),
    }


def _judge_response(query: str, response_text: str) -> dict:
    """Use LLM-as-judge to score a response."""
    llm = ChatOpenAI(model=MODEL, temperature=0)
    prompt = JUDGE_PROMPT.format(query=query, response=response_text)
    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        text = result.content.strip()
        # Parse JSON from response
        import re
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print(f"  ⚠ Judge error: {e}")
    return {
        "accuracy": 0, "completeness": 0, "coherence": 0,
        "safety": 0, "actionability": 0,
        "bottlenecks": ["judge_error"], "flow_opportunities": [],
    }


def _keyword_match_rate(text: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found in response."""
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return round(matches / max(len(keywords), 1), 2)


def analyze_baseline() -> dict:
    """Run baseline analysis across all diagnostic queries and techniques.

    Returns a structured analysis report.
    """
    techniques_to_test = ["zero_shot", "cot", "few_shot"]
    all_results = []

    print(f"\n{'='*60}")
    print(f"  BASELINE PROMPT ANALYSIS")
    print(f"  {len(DIAGNOSTIC_QUERIES)} queries × {len(techniques_to_test)} techniques")
    print(f"{'='*60}\n")

    for q in DIAGNOSTIC_QUERIES:
        query_results = {"query_id": q["id"], "query": q["query"], "complexity": q["complexity"], "techniques": {}}

        for tech in techniques_to_test:
            print(f"  [{q['id']}] {tech}...", end="", flush=True)

            prompt = BASE_PROMPTS[tech]
            result = _call_llm(prompt, q["query"])
            kw_rate = _keyword_match_rate(result["text"], q["expected_keywords"])

            # Judge the response
            scores = _judge_response(q["query"], result["text"])

            query_results["techniques"][tech] = {
                "response": result["text"][:500],  # truncate for storage
                "latency_s": result["latency_s"],
                "length": result["length"],
                "keyword_rate": kw_rate,
                "scores": scores,
            }

            avg_score = sum(scores.get(k, 0) for k in ["accuracy", "completeness", "coherence", "safety", "actionability"]) / 5
            print(f"  avg={avg_score:.1f}/5  kw={kw_rate:.0%}  {result['latency_s']}s")

        all_results.append(query_results)

    # ── Aggregate analysis ────────────────────────────────────────
    analysis = {
        "queries": all_results,
        "summary": _build_summary(all_results, techniques_to_test),
    }

    return analysis


def _build_summary(results: list[dict], techniques: list[str]) -> dict:
    """Build aggregated summary from results."""
    summary = {"per_technique": {}, "bottlenecks": [], "flow_opportunities": []}

    for tech in techniques:
        scores_agg = {"accuracy": [], "completeness": [], "coherence": [], "safety": [], "actionability": []}
        kw_rates = []
        latencies = []

        for r in results:
            tech_data = r["techniques"].get(tech, {})
            scores = tech_data.get("scores", {})
            for dim in scores_agg:
                val = scores.get(dim, 0)
                if isinstance(val, (int, float)) and val > 0:
                    scores_agg[dim].append(val)
            kw_rates.append(tech_data.get("keyword_rate", 0))
            latencies.append(tech_data.get("latency_s", 0))

            # Collect bottlenecks and opportunities
            for b in scores.get("bottlenecks", []):
                if b not in summary["bottlenecks"]:
                    summary["bottlenecks"].append(b)
            for o in scores.get("flow_opportunities", []):
                if o not in summary["flow_opportunities"]:
                    summary["flow_opportunities"].append(o)

        import numpy as np
        summary["per_technique"][tech] = {
            dim: round(float(np.mean(vals)), 2) if vals else 0
            for dim, vals in scores_agg.items()
        }
        summary["per_technique"][tech]["keyword_rate"] = round(float(np.mean(kw_rates)), 2)
        summary["per_technique"][tech]["avg_latency_s"] = round(float(np.mean(latencies)), 2)

    return summary


def main() -> None:
    analysis = analyze_baseline()

    # Save report
    report_path = RESULTS_DIR / "baseline_analysis.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    # Generate text summary
    summary = analysis["summary"]
    lines = []
    lines.append("=" * 60)
    lines.append("  BASELINE ANALYSIS — SUMMARY")
    lines.append("=" * 60)

    lines.append("\n── Per Technique Scores (avg 1-5) ──")
    for tech, data in summary["per_technique"].items():
        scores_str = "  ".join(f"{k}={v:.1f}" for k, v in data.items() if k not in ("keyword_rate", "avg_latency_s"))
        lines.append(f"  {tech:14s}  {scores_str}  kw={data['keyword_rate']:.0%}  lat={data['avg_latency_s']:.1f}s")

    lines.append(f"\n── Identified Bottlenecks ({len(summary['bottlenecks'])}) ──")
    for b in summary["bottlenecks"][:10]:
        lines.append(f"  • {b}")

    lines.append(f"\n── Flow Engineering Opportunities ({len(summary['flow_opportunities'])}) ──")
    for o in summary["flow_opportunities"][:10]:
        lines.append(f"  → {o}")

    summary_text = "\n".join(lines) + "\n"
    summary_path = RESULTS_DIR / "baseline_analysis_summary.txt"
    summary_path.write_text(summary_text)

    print(f"\n{summary_text}")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
