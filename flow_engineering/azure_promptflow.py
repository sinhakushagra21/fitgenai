#!/usr/bin/env python3
"""
flow_engineering/azure_promptflow.py
─────────────────────────────────────
Azure Prompt Flow integration for FITGEN.AI.

Provides:
  1. YAML flow definition generation for Azure Prompt Flow
  2. Programmatic flow execution using the promptflow SDK
  3. Batch evaluation mode for running flows across datasets
  4. Flow validation utilities

Azure Prompt Flow connects LLM calls, tools, and evaluation into
visual DAG workflows that can be deployed, versioned, and monitored
in Azure AI Studio.

Note: Requires `promptflow` and `promptflow-tools` packages.
      Falls back gracefully if not installed (generates YAML only).

Run:
    python -m flow_engineering.azure_promptflow
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()

FLOWS_DIR = Path(__file__).resolve().parent / "flows"
FLOWS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. YAML Flow Definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLASSIFY_AND_ROUTE_FLOW = """\
# Azure Prompt Flow: Classify and Route
# ─────────────────────────────────────
# This flow classifies user intent and routes to the appropriate
# specialist tool (workout_tool or diet_tool).
#
# DAG: input → classify_intent → route_decision → specialist_call → output

$schema: https://azuremlschemas.azureedge.net/promptflow/latest/Flow.schema.json

inputs:
  query:
    type: string
    description: "User's fitness-related question or request"
  prompting_technique:
    type: string
    default: "zero_shot"
    description: "Which prompting technique to use (zero_shot, few_shot, cot, analogical, generate_knowledge, decomposition)"

outputs:
  response:
    type: string
    reference: ${synthesize_response.output}
  tool_used:
    type: string
    reference: ${classify_intent.output.tool}
  reasoning:
    type: string
    reference: ${classify_intent.output.reasoning}

nodes:
  - name: classify_intent
    type: llm
    source:
      type: code
      path: classify_intent.py
    inputs:
      query: ${inputs.query}
      technique: ${inputs.prompting_technique}
    connection: open_ai_connection
    api: chat
    provider: AzureOpenAI
    model: gpt-4o-mini

  - name: route_decision
    type: python
    source:
      type: code
      path: route_decision.py
    inputs:
      classification: ${classify_intent.output}

  - name: specialist_call
    type: llm
    source:
      type: code
      path: specialist_call.py
    inputs:
      query: ${inputs.query}
      tool: ${route_decision.output.tool}
      specialist_prompt: ${route_decision.output.specialist_prompt}
    connection: open_ai_connection
    api: chat
    provider: AzureOpenAI
    model: gpt-4o-mini

  - name: synthesize_response
    type: python
    source:
      type: code
      path: synthesize_response.py
    inputs:
      specialist_output: ${specialist_call.output}
      tool_used: ${route_decision.output.tool}

environment:
  python_requirements_txt: requirements.txt
"""

GENERATE_WITH_RAG_FLOW = """\
# Azure Prompt Flow: RAG-Augmented Generation
# ─────────────────────────────────────────────
# Retrieves relevant fitness knowledge, augments the prompt,
# and generates a grounded response.
#
# DAG: input → retrieve_context → augment_prompt → generate → output

$schema: https://azuremlschemas.azureedge.net/promptflow/latest/Flow.schema.json

inputs:
  query:
    type: string
    description: "User's fitness question"
  top_k:
    type: int
    default: 3
    description: "Number of knowledge documents to retrieve"

outputs:
  response:
    type: string
    reference: ${generate_response.output}
  sources:
    type: string
    reference: ${retrieve_context.output.sources}
  retrieval_scores:
    type: string
    reference: ${retrieve_context.output.scores}

nodes:
  - name: retrieve_context
    type: python
    source:
      type: code
      path: retrieve_context.py
    inputs:
      query: ${inputs.query}
      top_k: ${inputs.top_k}

  - name: augment_prompt
    type: python
    source:
      type: code
      path: augment_prompt.py
    inputs:
      query: ${inputs.query}
      context: ${retrieve_context.output.context}

  - name: generate_response
    type: llm
    source:
      type: code
      path: generate_response.py
    inputs:
      augmented_prompt: ${augment_prompt.output}
    connection: open_ai_connection
    api: chat
    provider: AzureOpenAI
    model: gpt-4o-mini

environment:
  python_requirements_txt: requirements.txt
"""

SELF_REFINE_FLOW = """\
# Azure Prompt Flow: Self-Refine (Meta-Prompting)
# ─────────────────────────────────────────────────
# Three-step meta-prompting chain:
#   1. Generate initial response
#   2. Self-critique with scoring
#   3. Produce refined response
#
# DAG: input → initial_generate → self_critique → refine → output

$schema: https://azuremlschemas.azureedge.net/promptflow/latest/Flow.schema.json

inputs:
  query:
    type: string
    description: "User's fitness question"

outputs:
  refined_response:
    type: string
    reference: ${refine_response.output}
  initial_response:
    type: string
    reference: ${initial_generate.output}
  critique:
    type: string
    reference: ${self_critique.output}

nodes:
  - name: initial_generate
    type: llm
    source:
      type: code
      path: initial_generate.py
    inputs:
      query: ${inputs.query}
    connection: open_ai_connection
    api: chat
    provider: AzureOpenAI
    model: gpt-4o-mini

  - name: self_critique
    type: llm
    source:
      type: code
      path: self_critique.py
    inputs:
      query: ${inputs.query}
      response: ${initial_generate.output}
    connection: open_ai_connection
    api: chat
    provider: AzureOpenAI
    model: gpt-4o-mini

  - name: refine_response
    type: llm
    source:
      type: code
      path: refine_response.py
    inputs:
      query: ${inputs.query}
      initial_response: ${initial_generate.output}
      critique: ${self_critique.output}
    connection: open_ai_connection
    api: chat
    provider: AzureOpenAI
    model: gpt-4o-mini

environment:
  python_requirements_txt: requirements.txt
"""

EVALUATION_FLOW = """\
# Azure Prompt Flow: Evaluation Flow
# ────────────────────────────────────
# Evaluates model responses against ground truth using multiple metrics.
#
# DAG: input → compute_metrics → aggregate → output

$schema: https://azuremlschemas.azureedge.net/promptflow/latest/Flow.schema.json

inputs:
  query:
    type: string
  response:
    type: string
  expected_keywords:
    type: list
  ground_truth_tool:
    type: string

outputs:
  evaluation_result:
    type: string
    reference: ${aggregate_scores.output}

nodes:
  - name: keyword_match
    type: python
    source:
      type: code
      path: keyword_match.py
    inputs:
      response: ${inputs.response}
      expected_keywords: ${inputs.expected_keywords}

  - name: coherence_judge
    type: llm
    source:
      type: code
      path: coherence_judge.py
    inputs:
      query: ${inputs.query}
      response: ${inputs.response}
    connection: open_ai_connection
    api: chat
    provider: AzureOpenAI
    model: gpt-4o-mini

  - name: groundedness_judge
    type: llm
    source:
      type: code
      path: groundedness_judge.py
    inputs:
      query: ${inputs.query}
      response: ${inputs.response}
    connection: open_ai_connection
    api: chat
    provider: AzureOpenAI
    model: gpt-4o-mini

  - name: aggregate_scores
    type: python
    source:
      type: code
      path: aggregate_scores.py
    inputs:
      keyword_score: ${keyword_match.output}
      coherence_score: ${coherence_judge.output}
      groundedness_score: ${groundedness_judge.output}

environment:
  python_requirements_txt: requirements.txt
"""

# ── Flow definitions registry ────────────────────────────────────

FLOW_DEFINITIONS = {
    "classify_and_route": CLASSIFY_AND_ROUTE_FLOW,
    "generate_with_rag": GENERATE_WITH_RAG_FLOW,
    "self_refine": SELF_REFINE_FLOW,
    "evaluation": EVALUATION_FLOW,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Flow File Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_flow_files() -> list[Path]:
    """Write YAML flow definitions to disk."""
    generated = []
    for name, yaml_content in FLOW_DEFINITIONS.items():
        path = FLOWS_DIR / f"{name}.yaml"
        path.write_text(yaml_content)
        generated.append(path)
        print(f"  ✓ Generated {path}")
    return generated


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Flow Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_flows() -> dict[str, bool]:
    """Validate that all YAML flow definitions parse correctly."""
    import yaml

    results = {}
    for name, yaml_content in FLOW_DEFINITIONS.items():
        try:
            parsed = yaml.safe_load(yaml_content)
            # Check required fields
            has_inputs = "inputs" in parsed
            has_outputs = "outputs" in parsed
            has_nodes = "nodes" in parsed
            valid = has_inputs and has_outputs and has_nodes
            results[name] = valid
            status = "✓" if valid else "✗"
            node_count = len(parsed.get("nodes", []))
            print(f"  {status} {name}: {node_count} nodes, valid={valid}")
        except Exception as e:
            results[name] = False
            print(f"  ✗ {name}: parse error — {e}")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Programmatic Flow Execution (via promptflow SDK if available)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _check_promptflow_available() -> bool:
    """Check if promptflow SDK is installed."""
    try:
        import promptflow  # noqa: F401
        return True
    except ImportError:
        return False


def run_flow_programmatic(
    flow_name: str,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Execute a flow programmatically.

    If promptflow SDK is available, uses it directly.
    Otherwise, falls back to the LangChain-based equivalent.
    """
    if _check_promptflow_available():
        return _run_with_promptflow(flow_name, inputs)
    else:
        return _run_with_langchain_fallback(flow_name, inputs)


def _run_with_promptflow(flow_name: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Execute using Azure Prompt Flow SDK."""
    try:
        from promptflow.client import PFClient

        client = PFClient()
        flow_path = FLOWS_DIR / f"{flow_name}.yaml"

        if not flow_path.exists():
            generate_flow_files()

        result = client.test(flow=str(flow_path), inputs=inputs)
        return {"source": "promptflow_sdk", "result": result}
    except Exception as e:
        print(f"  ⚠ Prompt Flow SDK execution failed: {e}")
        return _run_with_langchain_fallback(flow_name, inputs)


def _run_with_langchain_fallback(flow_name: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Execute using LangChain chain variants as fallback."""
    from flow_engineering.chain_variants import ALL_VARIANTS

    # Map flow names to chain variants
    flow_to_variant = {
        "classify_and_route": "C_decompose_route",
        "generate_with_rag": "B_helper_prompt",
        "self_refine": "D_self_refine",
        "evaluation": "A_vanilla",
    }

    variant_name = flow_to_variant.get(flow_name, "A_vanilla")
    variant = ALL_VARIANTS[variant_name]

    result = variant.invoke({"query": inputs.get("query", inputs.get("question", ""))})
    return {
        "source": "langchain_fallback",
        "variant_used": variant_name,
        "result": result,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Batch Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_batch_evaluation(
    flow_name: str,
    dataset: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run a flow across a dataset in batch mode."""
    import time

    results = []
    for i, item in enumerate(dataset, 1):
        print(f"  [{i}/{len(dataset)}] {item.get('query', '')[:50]}...", end="", flush=True)
        t0 = time.perf_counter()
        try:
            result = run_flow_programmatic(flow_name, item)
            elapsed = time.perf_counter() - t0
            result["latency_s"] = round(elapsed, 3)
            result["input"] = item
            results.append(result)
            print(f"  ✓ {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  ✗ {e}")
            results.append({
                "source": "error",
                "error": str(e),
                "latency_s": round(elapsed, 3),
                "input": item,
            })

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Azure AI Studio Connection Info
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_azure_config() -> dict[str, str]:
    """Return Azure Prompt Flow configuration from environment."""
    return {
        "azure_openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "(not set)"),
        "azure_openai_key": "***" if os.getenv("AZURE_OPENAI_KEY") else "(not set)",
        "azure_openai_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        "azure_subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID", "(not set)"),
        "azure_resource_group": os.getenv("AZURE_RESOURCE_GROUP", "(not set)"),
        "azure_workspace_name": os.getenv("AZURE_ML_WORKSPACE", "(not set)"),
        "promptflow_installed": str(_check_promptflow_available()),
    }


# ── CLI ───────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  AZURE PROMPT FLOW — SETUP & VALIDATION")
    print(f"{'='*60}\n")

    # 1. Generate YAML files
    print("1. Generating YAML flow definitions...")
    files = generate_flow_files()

    # 2. Validate
    print("\n2. Validating flow definitions...")
    validation = validate_flows()

    # 3. Check Azure config
    print("\n3. Azure configuration:")
    config = get_azure_config()
    for k, v in config.items():
        print(f"  {k}: {v}")

    # 4. Test execution (LangChain fallback)
    print("\n4. Test flow execution (LangChain fallback)...")
    test_input = {"query": "Give me a beginner workout plan."}
    for flow_name in ["classify_and_route", "self_refine"]:
        print(f"\n  Testing {flow_name}...")
        try:
            result = run_flow_programmatic(flow_name, test_input)
            source = result.get("source", "unknown")
            print(f"  ✓ Source: {source}")
            if "result" in result and isinstance(result["result"], dict):
                response = result["result"].get("response", "")
                print(f"  Preview: {response[:100]}...")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    # 5. Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Flow YAML files: {len(files)} generated in {FLOWS_DIR}")
    valid_count = sum(1 for v in validation.values() if v)
    print(f"  Validation: {valid_count}/{len(validation)} flows valid")
    print(f"  SDK available: {config['promptflow_installed']}")
    if config["promptflow_installed"] == "False":
        print(f"  Note: Install promptflow for native Azure integration:")
        print(f"    pip install promptflow promptflow-tools")


if __name__ == "__main__":
    main()
