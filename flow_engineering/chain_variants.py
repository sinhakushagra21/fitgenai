#!/usr/bin/env python3
"""
flow_engineering/chain_variants.py
───────────────────────────────────
Defines four LangChain chain variants for structured experimentation.

Each variant is a composable Runnable that can be plugged into the
FlowRunner for side-by-side comparison.

Variants:
  A — Vanilla:        Direct system prompt → LLM → response
  B — Helper Prompt:  Generate domain knowledge → enrich prompt → LLM
  C — Decompose+Route: Break query → classify sub-tasks → route → merge
  D — Self-Refine:    Generate → self-critique → refine (meta-prompting chain)

Run:
    python -m flow_engineering.chain_variants
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.prompts.base_prompts import (
    BASE_COT,
    BASE_GENERATE_KNOWLEDGE,
    BASE_ZERO_SHOT,
    BASE_DECOMPOSITION,
)

load_dotenv()

MODEL = "gpt-4o-mini"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Variant A — Vanilla (baseline single-chain)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VARIANT_A_DESCRIPTION = (
    "Vanilla: Direct system prompt → LLM → response. "
    "No chaining, routing, or auxiliary steps."
)


def _variant_a_fn(inputs: dict[str, Any]) -> dict[str, Any]:
    """Variant A: Single LLM call with zero-shot prompt."""
    query = inputs["query"]
    llm = ChatOpenAI(model=MODEL, temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=BASE_ZERO_SHOT),
        HumanMessage(content=query),
    ])
    return {
        "variant": "A_vanilla",
        "response": response.content,
        "chain_steps": ["direct_llm_call"],
        "description": VARIANT_A_DESCRIPTION,
    }


variant_a = RunnableLambda(_variant_a_fn).with_config({"run_name": "Variant_A_Vanilla"})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Variant B — Helper Prompt (knowledge generation → enriched prompt)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VARIANT_B_DESCRIPTION = (
    "Helper Prompt: Pre-generate domain knowledge → inject into main prompt → "
    "LLM response. Two-step sequential chain."
)

_KNOWLEDGE_GENERATION_PROMPT = """\
You are a fitness and nutrition knowledge assistant.
Given the user's query, generate 3-5 relevant, evidence-based knowledge
statements that a fitness coach should consider when answering.

Return ONLY the knowledge statements, one per line, prefixed with "K1:", "K2:", etc.

User query: {query}
"""


def _variant_b_fn(inputs: dict[str, Any]) -> dict[str, Any]:
    """Variant B: Two-step chain — knowledge generation → enriched response."""
    query = inputs["query"]
    llm = ChatOpenAI(model=MODEL, temperature=0.3)

    # Step 1: Generate helper knowledge
    knowledge_resp = llm.invoke([
        SystemMessage(content="You are a fitness science knowledge assistant."),
        HumanMessage(content=_KNOWLEDGE_GENERATION_PROMPT.format(query=query)),
    ])
    knowledge = knowledge_resp.content

    # Step 2: Use knowledge to generate enriched response
    enriched_prompt = (
        BASE_GENERATE_KNOWLEDGE
        + f"\n\n<pre_generated_knowledge>\n{knowledge}\n</pre_generated_knowledge>\n\n"
        "Use the above knowledge statements to ground your response."
    )
    response = llm.invoke([
        SystemMessage(content=enriched_prompt),
        HumanMessage(content=query),
    ])

    return {
        "variant": "B_helper_prompt",
        "response": response.content,
        "chain_steps": ["knowledge_generation", "enriched_llm_call"],
        "intermediate": {"generated_knowledge": knowledge},
        "description": VARIANT_B_DESCRIPTION,
    }


variant_b = RunnableLambda(_variant_b_fn).with_config({"run_name": "Variant_B_Helper"})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Variant C — Decompose + Route
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VARIANT_C_DESCRIPTION = (
    "Decompose+Route: Break query into sub-tasks → classify each → "
    "route to specialist prompts → merge results."
)

_DECOMPOSE_PROMPT = """\
You are a query decomposition assistant for a fitness AI.
Break the following user query into distinct sub-tasks.

For each sub-task, specify:
- task_id: a short identifier
- description: what needs to be answered
- domain: one of "workout", "diet", "safety", "general"

Return ONLY valid JSON array. Example:
[
    {{"task_id": "1", "description": "Design upper body workout", "domain": "workout"}},
    {{"task_id": "2", "description": "Calculate protein needs", "domain": "diet"}}
]

User query: {query}
"""

_SPECIALIST_PROMPTS = {
    "workout": "You are an expert strength & conditioning coach. Answer ONLY the workout-related sub-task below with specific sets, reps, and exercise selections.",
    "diet": "You are an expert sports nutritionist. Answer ONLY the nutrition-related sub-task below with specific macros, calories, and food recommendations.",
    "safety": "You are a fitness safety advisor. Address the safety concern below with appropriate medical disclaimers and modifications.",
    "general": "You are a friendly fitness assistant. Answer the general question below concisely.",
}

_SYNTHESIS_PROMPT = """\
You are FITGEN.AI. Synthesize the following specialist responses into one
coherent, well-structured answer. Do NOT repeat information. Maintain a
warm, coaching tone.

Original question: {query}

Specialist responses:
{responses}

Provide a unified response:
"""


def _variant_c_fn(inputs: dict[str, Any]) -> dict[str, Any]:
    """Variant C: Decompose → Route → Specialist calls → Synthesize."""
    query = inputs["query"]
    llm = ChatOpenAI(model=MODEL, temperature=0.3)
    import re

    # Step 1: Decompose
    decompose_resp = llm.invoke([
        SystemMessage(content="You are a query decomposition assistant."),
        HumanMessage(content=_DECOMPOSE_PROMPT.format(query=query)),
    ])
    try:
        match = re.search(r"\[.*\]", decompose_resp.content, flags=re.DOTALL)
        sub_tasks = json.loads(match.group(0)) if match else [{"task_id": "1", "description": query, "domain": "general"}]
    except (json.JSONDecodeError, AttributeError):
        sub_tasks = [{"task_id": "1", "description": query, "domain": "general"}]

    # Step 2: Route each sub-task to specialist
    specialist_responses = []
    for task in sub_tasks:
        domain = task.get("domain", "general")
        specialist_prompt = _SPECIALIST_PROMPTS.get(domain, _SPECIALIST_PROMPTS["general"])
        resp = llm.invoke([
            SystemMessage(content=specialist_prompt),
            HumanMessage(content=f"Sub-task: {task['description']}"),
        ])
        specialist_responses.append({
            "task_id": task["task_id"],
            "domain": domain,
            "response": resp.content,
        })

    # Step 3: Synthesize
    responses_text = "\n\n".join(
        f"[{r['domain'].upper()}] {r['response']}" for r in specialist_responses
    )
    synthesis_resp = llm.invoke([
        SystemMessage(content="You are FITGEN.AI, a fitness coaching assistant."),
        HumanMessage(content=_SYNTHESIS_PROMPT.format(query=query, responses=responses_text)),
    ])

    return {
        "variant": "C_decompose_route",
        "response": synthesis_resp.content,
        "chain_steps": ["decompose", "route_specialists", "synthesize"],
        "intermediate": {
            "sub_tasks": sub_tasks,
            "specialist_responses": [
                {"task_id": r["task_id"], "domain": r["domain"], "response": r["response"][:200]}
                for r in specialist_responses
            ],
        },
        "description": VARIANT_C_DESCRIPTION,
    }


variant_c = RunnableLambda(_variant_c_fn).with_config({"run_name": "Variant_C_Decompose"})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Variant D — Self-Refine (generate → critique → refine)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VARIANT_D_DESCRIPTION = (
    "Self-Refine: Generate initial response → self-critique with scoring → "
    "produce refined response. Three-step meta-prompting chain."
)

_CRITIQUE_PROMPT = """\
Analyze the following fitness AI response and provide a critique.

QUERY: {query}
RESPONSE: {response}

Score each dimension 1-5:
1. Accuracy — factual correctness
2. Completeness — covers all aspects
3. Practicality — actionable advice
4. Safety — appropriate disclaimers
5. Structure — well-organized

Then list 2-3 specific improvements. Be concise.
"""

_REFINE_PROMPT = """\
You are FITGEN.AI. Improve your previous response based on expert feedback.

ORIGINAL QUERY: {query}
YOUR PREVIOUS RESPONSE: {initial_response}
EXPERT FEEDBACK: {critique}

Write an IMPROVED response that addresses ALL feedback points.
Keep it practical, well-structured, and evidence-based.
"""


def _variant_d_fn(inputs: dict[str, Any]) -> dict[str, Any]:
    """Variant D: Generate → Critique → Refine."""
    query = inputs["query"]
    llm = ChatOpenAI(model=MODEL, temperature=0.3)

    # Step 1: Initial generation
    initial = llm.invoke([
        SystemMessage(content=BASE_COT),
        HumanMessage(content=query),
    ])
    initial_text = initial.content

    # Step 2: Self-critique
    critique = llm.invoke([
        SystemMessage(content="You are an expert fitness content reviewer."),
        HumanMessage(content=_CRITIQUE_PROMPT.format(query=query, response=initial_text)),
    ])
    critique_text = critique.content

    # Step 3: Refine
    refined = llm.invoke([
        SystemMessage(content=BASE_COT),
        HumanMessage(content=_REFINE_PROMPT.format(
            query=query, initial_response=initial_text, critique=critique_text,
        )),
    ])

    return {
        "variant": "D_self_refine",
        "response": refined.content,
        "chain_steps": ["initial_generation", "self_critique", "refinement"],
        "intermediate": {
            "initial_response": initial_text[:300],
            "critique": critique_text[:300],
        },
        "description": VARIANT_D_DESCRIPTION,
    }


variant_d = RunnableLambda(_variant_d_fn).with_config({"run_name": "Variant_D_SelfRefine"})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALL_VARIANTS = {
    "A_vanilla": variant_a,
    "B_helper_prompt": variant_b,
    "C_decompose_route": variant_c,
    "D_self_refine": variant_d,
}

VARIANT_DESCRIPTIONS = {
    "A_vanilla": VARIANT_A_DESCRIPTION,
    "B_helper_prompt": VARIANT_B_DESCRIPTION,
    "C_decompose_route": VARIANT_C_DESCRIPTION,
    "D_self_refine": VARIANT_D_DESCRIPTION,
}


def build_all_variants() -> dict[str, Any]:
    """Return the registry of all chain variants."""
    return dict(ALL_VARIANTS)


# ── CLI ───────────────────────────────────────────────────────────

def main():
    """Quick smoke-test: run each variant on a sample query."""
    test_query = "Give me a 3-day beginner workout plan with meal suggestions."
    print(f"\n{'='*60}")
    print(f"  CHAIN VARIANTS — SMOKE TEST")
    print(f"  Query: {test_query}")
    print(f"{'='*60}\n")

    import time
    for name, variant in ALL_VARIANTS.items():
        print(f"Running {name}...", end="", flush=True)
        t0 = time.perf_counter()
        result = variant.invoke({"query": test_query})
        elapsed = time.perf_counter() - t0
        print(f"  ✓ ({elapsed:.1f}s, {len(result['response'])} chars)")
        print(f"  Steps: {result['chain_steps']}")
        print(f"  Preview: {result['response'][:120]}...\n")


if __name__ == "__main__":
    main()
