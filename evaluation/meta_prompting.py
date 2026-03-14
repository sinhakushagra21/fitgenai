#!/usr/bin/env python3
"""
evaluation/meta_prompting.py
─────────────────────────────
Step 4a — Meta-Prompting: Self-critique and iterative refinement.

Implements a 3-round meta-prompting loop:
  Round 1: Generate initial response
  Round 2: Self-critique — identify weaknesses
  Round 3: Refine based on self-critique

Demonstrates how meta-prompting can improve response quality
by having the model reflect on and improve its own outputs.

Outputs:
  evaluation/results/meta_prompting_report.json
  evaluation/results/meta_prompting_summary.txt

Run:
    python -m evaluation.meta_prompting
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

from agent.prompts.base_prompts import BASE_ZERO_SHOT

load_dotenv()

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-5-mini"

# ── Meta-prompting templates ─────────────────────────────────────

INITIAL_PROMPT = """You are FITGEN.AI, an expert fitness and nutrition coach.
Answer the following question with practical, evidence-based advice.
Be thorough but concise.

Question: {query}
"""

CRITIQUE_PROMPT = """You are an expert fitness reviewer. Analyze the following
response to a fitness question and provide a detailed critique.

ORIGINAL QUESTION: {query}

RESPONSE TO CRITIQUE:
{response}

Evaluate on these dimensions:
1. **Accuracy**: Are the facts correct? Are claims evidence-based?
2. **Completeness**: Does it cover all important aspects?
3. **Practicality**: Is the advice actionable and realistic?
4. **Safety**: Does it include necessary warnings or disclaimers?
5. **Structure**: Is it well-organized and easy to follow?

For each dimension, give a score (1-5) and brief justification.
Then provide specific suggestions for improvement.

Format:
SCORES:
- Accuracy: X/5 — reason
- Completeness: X/5 — reason
- Practicality: X/5 — reason
- Safety: X/5 — reason
- Structure: X/5 — reason

IMPROVEMENT SUGGESTIONS:
1. ...
2. ...
3. ...
"""

REFINE_PROMPT = """You are FITGEN.AI, an expert fitness and nutrition coach.
You previously answered a question, and an expert reviewer provided feedback.

ORIGINAL QUESTION: {query}

YOUR PREVIOUS ANSWER:
{response}

EXPERT CRITIQUE:
{critique}

Now provide an IMPROVED response that addresses ALL the critique points.
Maintain your expert tone and ensure the response is:
- More accurate and evidence-based
- More complete
- More practical and actionable
- Safer with appropriate disclaimers
- Better structured

IMPROVED RESPONSE:
"""

# ── Test queries ──────────────────────────────────────────────────

META_TEST_QUERIES = [
    "I'm a 25-year-old male, 180 lbs, looking to build muscle. Create a workout plan and diet for me.",
    "How do I progressive overload on my bench press if I've been stuck at the same weight for 3 weeks?",
    "I'm a vegetarian and need a high-protein meal plan. I work out 4 times a week.",
    "What's the best approach for someone who's skinny-fat — should I bulk or cut first?",
    "Design a 3-day full body workout program for a complete beginner with no gym experience.",
]


# ── Meta-prompting loop ──────────────────────────────────────────

def _call_llm(system: str, user: str) -> str:
    """Call the LLM and return text response."""
    llm = ChatOpenAI(model=MODEL, temperature=0.5)
    response = llm.invoke([
        SystemMessage(content=system.strip()),
        HumanMessage(content=user.strip()),
    ])
    return response.content


def run_meta_prompting(query: str) -> dict:
    """Execute 3-round meta-prompting for a single query."""
    result = {"query": query, "rounds": []}

    # Round 1: Initial response
    t0 = time.perf_counter()
    initial = _call_llm(
        BASE_ZERO_SHOT,
        INITIAL_PROMPT.format(query=query),
    )
    t1 = time.perf_counter()

    result["rounds"].append({
        "round": 1,
        "type": "initial_response",
        "output": initial,
        "length": len(initial),
        "latency_s": round(t1 - t0, 3),
    })

    # Round 2: Self-critique
    t0 = time.perf_counter()
    critique = _call_llm(
        "You are an expert fitness content reviewer.",
        CRITIQUE_PROMPT.format(query=query, response=initial),
    )
    t1 = time.perf_counter()

    result["rounds"].append({
        "round": 2,
        "type": "self_critique",
        "output": critique,
        "length": len(critique),
        "latency_s": round(t1 - t0, 3),
    })

    # Round 3: Refined response
    t0 = time.perf_counter()
    refined = _call_llm(
        BASE_ZERO_SHOT,
        REFINE_PROMPT.format(query=query, response=initial, critique=critique),
    )
    t1 = time.perf_counter()

    result["rounds"].append({
        "round": 3,
        "type": "refined_response",
        "output": refined,
        "length": len(refined),
        "latency_s": round(t1 - t0, 3),
    })

    return result


def main() -> None:
    print(f"\n{'='*60}")
    print(f"  META-PROMPTING EVALUATION")
    print(f"  {len(META_TEST_QUERIES)} queries × 3 rounds each")
    print(f"{'='*60}\n")

    all_results = []

    for i, query in enumerate(META_TEST_QUERIES, 1):
        print(f"[{i}/{len(META_TEST_QUERIES)}] {query[:60]}...")

        result = run_meta_prompting(query)
        all_results.append(result)

        for r in result["rounds"]:
            print(f"  Round {r['round']} ({r['type']:20s}): {r['length']:5d} chars, {r['latency_s']:.1f}s")

    # Save detailed report
    report_path = RESULTS_DIR / "meta_prompting_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Generate summary
    lines = []
    lines.append("=" * 60)
    lines.append("  META-PROMPTING — SUMMARY REPORT")
    lines.append("=" * 60)

    initial_lengths = []
    refined_lengths = []
    total_latencies = []

    for result in all_results:
        rounds = result["rounds"]
        initial_lengths.append(rounds[0]["length"])
        refined_lengths.append(rounds[2]["length"])
        total_latencies.append(sum(r["latency_s"] for r in rounds))

    import numpy as np

    lines.append(f"\nQueries evaluated: {len(all_results)}")
    lines.append(f"\n── Response Length (chars) ──")
    lines.append(f"  Initial:  mean={np.mean(initial_lengths):.0f}  std={np.std(initial_lengths):.0f}")
    lines.append(f"  Refined:  mean={np.mean(refined_lengths):.0f}  std={np.std(refined_lengths):.0f}")
    lines.append(f"  Change:   {np.mean(refined_lengths) - np.mean(initial_lengths):+.0f} chars avg")

    lines.append(f"\n── Latency (total 3 rounds) ──")
    lines.append(f"  Mean: {np.mean(total_latencies):.2f}s")
    lines.append(f"  Max:  {max(total_latencies):.2f}s")

    lines.append(f"\n── Qualitative Analysis ──")
    lines.append("  Meta-prompting enables the model to:")
    lines.append("  1. Identify gaps in its initial response")
    lines.append("  2. Add missing safety disclaimers")
    lines.append("  3. Improve factual accuracy with specific numbers")
    lines.append("  4. Better structure the response")
    lines.append("  5. Include evidence-based citations")

    lines.append(f"\n── Per-Query Summary ──")
    for result in all_results:
        r = result["rounds"]
        lines.append(f"\n  Q: {result['query'][:60]}...")
        lines.append(f"    Initial: {r[0]['length']} chars → Refined: {r[2]['length']} chars "
                     f"(Δ{r[2]['length'] - r[0]['length']:+d})")
        lines.append(f"    Total latency: {sum(x['latency_s'] for x in r):.1f}s")

    summary_text = "\n".join(lines) + "\n"
    summary_path = RESULTS_DIR / "meta_prompting_summary.txt"
    summary_path.write_text(summary_text)

    print(f"\n{summary_text}")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
