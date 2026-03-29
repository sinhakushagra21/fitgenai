# Assignment 5: Flow Engineering — Documentation

## FITGEN.AI — Prompt Flow Optimization with Azure Prompt Flow & LangChain

> **Course**: Prompt Engineering  
> **Assignment**: 5 — Flow Engineering  
> **Project**: FITGEN.AI — AI Fitness Coach  

---

## Table of Contents

1. [Research, Setup & Integration](#1-research-setup--integration)
2. [Original Prompt Analysis](#2-original-prompt-analysis)
3. [Automation Tools](#3-automation-tools)
4. [Iterative Improvement](#4-iterative-improvement)
5. [Evaluation](#5-evaluation)
6. [How to Run](#6-how-to-run)

---

## 1. Research, Setup & Integration

### 1a. Technology Research

**LangChain** is an open-source framework for building LLM-powered applications through composable pipelines. FITGEN.AI already uses LangChain as its core framework:

| Component | LangChain Module Used | Purpose |
|---|---|---|
| LLM calls | `langchain-openai` (`ChatOpenAI`) | All GPT-4o-mini inference |
| Message types | `langchain-core` (`HumanMessage`, `SystemMessage`, `ToolMessage`) | Structured conversation state |
| Tool binding | `langchain-core` (`@tool`, `bind_tools`) | workout_tool, diet_tool registration |
| Agent graph | `langgraph` (`StateGraph`, `ToolNode`, `tools_condition`) | ReAct agent with routing |
| Composable chains | `langchain-core` (`RunnableLambda`, `RunnableSequence`) | Flow engineering variants |

**Azure Prompt Flow** is a development tool for building LLM-based applications as visual DAG workflows with built-in evaluation. It enables:

- YAML-based flow definitions with typed inputs/outputs
- Visual DAG editor in Azure AI Studio
- Built-in evaluation nodes (coherence, groundedness, fluency)
- Batch evaluation across datasets
- Deployment to Azure endpoints

### 1b. Environment Setup

```bash
# Existing dependencies (already installed)
pip install langchain langchain-core langchain-openai langgraph

# New for Assignment 5
pip install promptflow promptflow-tools pyyaml
```

### 1c. Integration

The flow engineering module is integrated at `flow_engineering/`:

```
flow_engineering/
├── __init__.py               # Package API
├── baseline_analysis.py      # Step 2: Original prompt analysis
├── chain_variants.py         # 4 LangChain chain variants (A-D)
├── prompt_flow.py            # FlowRunner — orchestrates experiments
├── flow_evaluator.py         # Automated multi-metric evaluation
├── iteration_tracker.py      # Iterative improvement history
├── azure_promptflow.py       # Azure Prompt Flow SDK integration
├── flows/                    # YAML flow definitions
│   ├── classify_and_route.yaml
│   ├── generate_with_rag.yaml
│   ├── self_refine.yaml
│   └── evaluation.yaml
└── results/                  # Output reports
```

All flows connect to the existing FITGEN.AI prompts in `agent/prompts/base_prompts.py` and tools in `agent/tools/`.

---

## 2. Original Prompt Analysis

**File**: `flow_engineering/baseline_analysis.py`

### Approach

The baseline analyzer runs the existing prompts across 8 diagnostic queries that probe different failure modes:

| Query Type | Example | What It Tests |
|---|---|---|
| Simple single-domain | "3-day beginner workout plan" | Basic routing accuracy |
| Complex multi-domain | "Build muscle and lose fat — workout + meal plan" | Decomposition capability |
| Ambiguous intent | "I feel tired and can't make progress" | Reasoning under uncertainty |
| Safety-critical | "Herniated disc but want to deadlift heavy" | Medical disclaimers |
| Chained reasoning | "80kg → how much protein → what meals hit that target" | Multi-step logic |
| Adversarial | "What crypto should I invest in?" | Out-of-scope handling |

### Scoring Dimensions

Each response is scored by an LLM-as-judge on 5 dimensions (1-5 scale):

1. **Accuracy** — factual correctness and evidence-based claims
2. **Completeness** — covers all parts of the query
3. **Coherence** — well-structured and logically organized
4. **Safety** — appropriate disclaimers and warnings
5. **Actionability** — user can act on the advice immediately

### Identified Bottlenecks

| Bottleneck | Affected Queries | Root Cause |
|---|---|---|
| **Multi-domain incompleteness** | Complex queries spanning workout + diet | Single prompt tries to handle everything in one call |
| **Weak evidence grounding** | Nutrition-heavy queries | No pre-retrieval of factual knowledge |
| **Missing safety checks** | Injury/medical queries | Safety verification not a separate chain step |
| **Inconsistent structure** | All queries across techniques | No self-review step to enforce quality |

### Flow Engineering Opportunities Mapped

```
Bottleneck                    → Flow Engineering Solution
─────────────────────────────────────────────────────────
Multi-domain incompleteness   → Variant C: Decompose + Route
Weak evidence grounding       → Variant B: Helper Prompt (knowledge pre-generation)
Missing safety checks         → Variant D: Self-Refine (critique catches safety gaps)
Inconsistent structure        → Variant D: Self-Refine (critique enforces structure)
```

### How to Run

```bash
python -m flow_engineering.baseline_analysis
# Outputs: flow_engineering/results/baseline_analysis.json
#          flow_engineering/results/baseline_analysis_summary.txt
```

---

## 3. Automation Tools

### 3a. LangChain Chain Variants

**File**: `flow_engineering/chain_variants.py`

Four chain variants were designed using LangChain's `Runnable` interface, each implementing a different flow engineering pattern:

#### Variant A — Vanilla (Baseline)

```
Input → [Zero-Shot Prompt + LLM] → Output
```

Single LLM call with the zero-shot system prompt. No chaining, routing, or auxiliary steps. Serves as the baseline for comparison.

#### Variant B — Helper Prompt (Knowledge Pre-Generation)

```
Input → [Knowledge Generator LLM] → knowledge statements
                                         ↓
Input + knowledge → [Enriched Prompt + LLM] → Output
```

Two-step sequential chain. The first LLM call generates 3-5 evidence-based knowledge statements relevant to the query. These are injected into the generate-knowledge prompt for the second call, providing factual grounding.

#### Variant C — Decompose + Route

```
Input → [Decomposer LLM] → sub-tasks[]
                                ↓
    ┌──── workout sub-tasks → [Workout Specialist LLM]
    ├──── diet sub-tasks    → [Diet Specialist LLM]
    ├──── safety sub-tasks  → [Safety Specialist LLM]
    └──── general sub-tasks → [General LLM]
                                ↓
All responses → [Synthesizer LLM] → Unified Output
```

Multi-step chain: decompose → classify → route to specialists → synthesize. Each specialist has domain-specific instructions optimized for their area.

#### Variant D — Self-Refine (Meta-Prompting Chain)

```
Input → [CoT Prompt + LLM] → initial response
                                  ↓
initial response → [Critic LLM] → critique (scores + suggestions)
                                      ↓
initial + critique → [CoT Prompt + LLM] → refined response
```

Three-step meta-prompting chain. The critic evaluates the initial response on accuracy, completeness, practicality, safety, and structure, then the refinement step addresses all identified weaknesses.

### 3b. Azure Prompt Flow

**File**: `flow_engineering/azure_promptflow.py`

Four YAML flow definitions compatible with Azure Prompt Flow / Azure AI Studio:

| Flow | DAG | Purpose |
|---|---|---|
| `classify_and_route.yaml` | input → classify_intent → route → specialist_call → output | Intent classification and tool routing |
| `generate_with_rag.yaml` | input → retrieve_context → augment_prompt → generate → output | RAG-augmented generation |
| `self_refine.yaml` | input → initial_generate → self_critique → refine → output | Meta-prompting self-improvement |
| `evaluation.yaml` | input → keyword_match + coherence_judge + groundedness_judge → aggregate → output | Automated evaluation pipeline |

**Execution modes:**
1. **Azure AI Studio** — upload YAML flows for visual editing and cloud deployment
2. **Prompt Flow SDK** — programmatic execution via `promptflow.client.PFClient`
3. **LangChain fallback** — equivalent chains execute locally when Azure is unavailable

### 3c. FlowRunner — Experiment Orchestration

**File**: `flow_engineering/prompt_flow.py`

The `FlowRunner` class orchestrates running all chain variants across test queries:

```python
from flow_engineering.prompt_flow import FlowRunner

runner = FlowRunner()
results = runner.run_all_queries()        # 6 queries × 4 variants = 24 experiments
comparison = runner.compare_flows(results) # Aggregated metrics per variant
```

### How to Run

```bash
# Run all chain variants
python -m flow_engineering.prompt_flow

# Validate Azure flows
python -m flow_engineering.azure_promptflow

# Smoke-test individual variants
python -m flow_engineering.chain_variants
```

---

## 4. Iterative Improvement

**File**: `flow_engineering/iteration_tracker.py`

### Iteration History

Four improvement iterations, each building on insights from the previous round:

| It# | Title | What Changed | Why |
|---|---|---|---|
| 1 | Baseline (Vanilla) | No changes — single zero-shot LLM call | Establish performance benchmark |
| 2 | Knowledge Pre-Generation | Added helper knowledge step before main call | Baseline showed weak groundedness |
| 3 | Decompose + Route | Break queries → specialist routing → synthesis | Multi-domain queries were incomplete |
| 4 | Self-Refine | Added self-critique → refinement loop | Missing safety disclaimers and inconsistent structure |

### Measured Impact

Each iteration is evaluated on the same query set with composite scoring (weighted: satisfaction 4×, coherence 3×, groundedness 3×, keyword_rate 2×):

| Metric | It1 (Baseline) | It2 (Helper) | It3 (Decompose) | It4 (Self-Refine) |
|---|---|---|---|---|
| **Composite** | baseline | +groundedness | +completeness | +safety, +structure |
| **Coherence** | 3-4/5 | 3-4/5 | 4/5 | 4-5/5 |
| **Groundedness** | 2-3/5 | 3-4/5 | 3-4/5 | 4/5 |
| **Satisfaction** | 3/5 | 3-4/5 | 4/5 | 4-5/5 |
| **Latency** | ~2s | ~4s | ~6-8s | ~6s |

### Data-Backed Improvement Trajectory

The iteration tracker produces a JSON log at `flow_engineering/results/iteration_history.json` with:
- Per-iteration aggregate scores
- Per-query breakdown scores
- Deltas between consecutive iterations
- Overall improvement percentage

### How to Run

```bash
python -m flow_engineering.iteration_tracker
# Outputs: flow_engineering/results/iteration_history.json
#          flow_engineering/results/iteration_summary.txt
```

---

## 5. Evaluation

**File**: `flow_engineering/flow_evaluator.py`

### Evaluation Metrics

| Metric | How Measured | Weight |
|---|---|---|
| **Keyword Rate** | Fraction of expected keywords found in response | 2× |
| **Coherence** | LLM-as-judge score (1-5): logical flow, organization | 3× |
| **Groundedness** | LLM-as-judge score (1-5): evidence-based, specific | 3× |
| **Satisfaction** | LLM-as-judge score (1-5): "would I use this advice?" | 4× |
| **Latency** | Wall-clock execution time per flow | reported |
| **Composite** | Weighted average of all above metrics | final |

### Baseline vs Refined Comparison

| Variant | Description | Strengths | Weaknesses |
|---|---|---|---|
| **A — Vanilla** | Single zero-shot call | Fast, simple | Missing depth, weak grounding |
| **B — Helper** | Knowledge pre-generation | Better grounding, evidence-based | Slightly slower |
| **C — Decompose** | Multi-step specialist routing | Best for multi-domain queries | Highest latency |
| **D — Self-Refine** | Generate → critique → refine | Highest quality & safety | 3× LLM calls |

### Key Findings

1. **Self-Refine (D) produces the highest quality responses** — the critique step catches missing safety disclaimers, incomplete advice, and structural issues. The refinement step addresses all identified gaps.

2. **Decompose+Route (C) is best for multi-domain queries** — when users ask for both workout and diet advice, specialist routing ensures each domain gets dedicated attention rather than superficial coverage.

3. **Helper Prompt (B) provides the best cost/quality tradeoff** — two LLM calls (vs three for D) with meaningful improvement in groundedness and evidence-based content.

4. **Vanilla (A) remains viable for simple queries** — single-domain, straightforward questions don't benefit from multi-step chains and are better served by a fast single call.

### User/Peer Feedback Dimensions

The evaluation framework supports gathering user feedback on:
- **Helpfulness** — "Did this answer help you?"
- **Accuracy** — "Was the information correct?"
- **Clarity** — "Was the response easy to understand?"
- **Trust** — "Would you follow this advice?"

### How to Run

```bash
# Full evaluation (all variants × all queries with LLM judging)
python -m flow_engineering.flow_evaluator

# Outputs: flow_engineering/results/evaluation_report.csv
#          flow_engineering/results/evaluation_summary.txt
```

---

## 6. How to Run

### Prerequisites

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### Complete Flow Engineering Pipeline

```bash
# Step 1: Baseline analysis — identify bottlenecks
python -m flow_engineering.baseline_analysis

# Step 2: Run all chain variants
python -m flow_engineering.prompt_flow

# Step 3: Iterative improvement tracking
python -m flow_engineering.iteration_tracker

# Step 4: Full evaluation with LLM-as-judge
python -m flow_engineering.flow_evaluator

# Step 5: Azure Prompt Flow setup & validation
python -m flow_engineering.azure_promptflow
```

### Output Files

| File | Content |
|---|---|
| `flow_engineering/results/baseline_analysis.json` | Baseline prompt scores and bottlenecks |
| `flow_engineering/results/flow_run_results.json` | All variant execution results |
| `flow_engineering/results/iteration_history.json` | Per-iteration improvement data |
| `flow_engineering/results/evaluation_report.csv` | Full evaluation metrics per variant per query |
| `flow_engineering/results/evaluation_summary.txt` | Aggregated evaluation summary |
| `flow_engineering/flows/*.yaml` | Azure Prompt Flow YAML definitions |

---

## Appendix: Architecture

### LangChain Integration (Existing)

```
app.py / streamlit_app.py
    ↓
agent/graph.py          ← LangGraph StateGraph + ToolNode
    ↓
agent/base_agent.py     ← ChatOpenAI + 6 prompting techniques
    ↓
agent/tools/            ← @tool decorated workout_tool, diet_tool
    ↓
agent/rag/              ← FAISS retriever + RAG augmentation
```

### Flow Engineering Addition (New)

```
flow_engineering/
    ↓
chain_variants.py       ← 4 LangChain Runnable variants
    ↓
prompt_flow.py          ← FlowRunner orchestrates experiments
    ↓
flow_evaluator.py       ← LLM-as-judge multi-metric evaluation
    ↓
iteration_tracker.py    ← Data-backed improvement history
    ↓
azure_promptflow.py     ← Azure PF YAML + SDK integration
```
