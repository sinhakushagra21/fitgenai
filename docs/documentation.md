# Assignment 4: Fine-Tuning the Model — Documentation

## FITGEN.AI — Prompt Engineering & Model Optimization

> **Course**: Prompt Engineering  
> **Assignment**: 4 — Fine-Tuning the Model  
> **Project**: FITGEN.AI — AI Fitness Coach  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Step 1: Prompt Sensitivity Optimization](#2-step-1-prompt-sensitivity-optimization)
3. [Step 2: Dataset Curation](#3-step-2-dataset-curation)
4. [Step 3: Fine-Tuning & RAG Pipeline](#4-step-3-fine-tuning--rag-pipeline)
5. [Step 4: Meta-Prompting & Perplexity Evaluation](#5-step-4-meta-prompting--perplexity-evaluation)
6. [Results Summary](#6-results-summary)
7. [How to Run](#7-how-to-run)

---

## 1. Project Overview

FITGEN.AI is a conversational fitness coaching agent built with:

- **LangGraph** — ReAct agent architecture with tool routing
- **LangChain + OpenAI** — GPT-5-mini for inference
- **5 Prompting Techniques** — Zero-Shot, Few-Shot, Chain-of-Thought, Analogical, Generate-Knowledge
- **Streamlit** — Interactive UI with side-by-side technique comparison

### Architecture

```
START → base_agent (LLM + tools) → [tools_condition]
         ↓ has tool_calls?           ↓ no tool_calls
    ToolNode (workout_tool          END
              diet_tool)
         ↓
    base_agent (relay)
         ↓
        END
```

Each specialist tool runs all 5 prompting techniques in parallel and returns a JSON object with one response per technique.

---

## 2. Step 1: Prompt Sensitivity Optimization

**File**: `evaluation/prompt_sensitivity.py`

### Approach

Tests model robustness across three dimensions:

| Dimension       | What we vary                                  | Why it matters                                |
|-----------------|-----------------------------------------------|-----------------------------------------------|
| **Paraphrase**  | 3 phrasings per query (same intent)           | Users ask the same thing in different ways     |
| **Temperature** | 0.0, 0.3, 0.7, 1.0                           | Stochastic variance in generation             |
| **Technique**   | Zero-Shot, Few-Shot, CoT, Analogical, GenKnow | Different prompt structures affect routing     |

### Metrics

- **Routing Accuracy**: Does the model select the correct tool (workout_tool, diet_tool, or none)?
- **Cosine Similarity**: TF-IDF based pairwise similarity between paraphrase responses (consistency)
- **Response Length Delta**: Stability of output length across variants

### How to Run

```bash
# Quick mode (~50 API calls)
python -m evaluation.prompt_sensitivity

# Full sweep (~300 API calls)
python -m evaluation.prompt_sensitivity --full
```

### Key Findings

- Zero-Shot and CoT achieve the highest routing accuracy
- Temperature has minimal effect on routing (tool selection is deterministic)
- Paraphrase stability is highest at T=0.0 and decreases with temperature
- Few-Shot prompts show the most consistent response structure across variants

---

## 3. Step 2: Dataset Curation

**File**: `data/generate_dataset.py`

### Dataset Composition

| Category      | Count | Description                                    |
|---------------|-------|------------------------------------------------|
| Typical       | 60    | Clear workout/diet queries with obvious routing |
| Edge Cases    | 20    | Ambiguous, multi-domain, borderline queries    |
| Adversarial   | 20    | Jailbreaks, off-topic, harmful, unsafe         |
| **Total**     | **100** |                                              |

### Tool Distribution

| Expected Tool  | Count |
|----------------|-------|
| workout_tool   | 43    |
| diet_tool      | 40    |
| none           | 17    |

### Splits

| Split | Count | Ratio |
|-------|-------|-------|
| Train | 70    | 70%   |
| Dev   | 15    | 15%   |
| Test  | 15    | 15%   |

### Format (JSONL)

```json
{
    "query": "Give me a 4-day upper/lower split for hypertrophy.",
    "expected_tool": "workout_tool",
    "expected_response_contains": ["upper", "lower", "hypertrophy"],
    "category": "typical"
}
```

---

## 4. Step 3: Fine-Tuning & RAG Pipeline

### 4a. Fine-Tuning Pipeline

**Files**: `fine_tuning/prepare_finetune_data.py`, `fine_tuning/run_finetune.py`, `fine_tuning/compare_models.py`

#### Data Preparation

Converts our dataset to OpenAI's fine-tuning format:
- System prompt with tool definitions
- User query
- Assistant response with correct tool_call (or polite refusal)

```bash
# Prepare fine-tuning data
python -m fine_tuning.prepare_finetune_data

# Validate data (dry run)
python -m fine_tuning.run_finetune --dry-run

# Launch fine-tuning job
python -m fine_tuning.run_finetune

# Compare baseline vs fine-tuned
python -m fine_tuning.compare_models
```

#### Fine-Tuning Configuration

| Parameter      | Value                     |
|----------------|---------------------------|
| Base Model     | gpt-4o-mini-2024-07-18    |
| Training Set   | 70 examples               |
| Validation Set | 15 examples               |
| Epochs         | 3 (configurable)          |
| Suffix         | fitgen-router             |

#### Comparison Metrics

- **Routing accuracy** by category (typical, edge, adversarial)
- **Keyword match rate** — expected keywords in response
- **Latency** — response time differences

### 4b. RAG Pipeline

**Files**: `agent/rag/knowledge_base.py`, `agent/rag/retriever.py`, `agent/rag/rag_tool.py`

#### Knowledge Base

25 curated documents covering:
- 10 workout knowledge entries (hypertrophy, progressive overload, rest periods, etc.)
- 10 diet/nutrition entries (protein, calories, macros, supplements, etc.)
- 5 general/safety entries (sleep, steroids warning, minimum calories, pregnancy, recomp)

All documents include peer-reviewed citations (ISSN, NSCA, ACSM, etc.).

#### Vector Store

- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **Index**: FAISS IndexFlatIP (cosine similarity via normalized vectors)
- **Caching**: Index and docs cached to disk for fast reloads
- **Retrieval**: Top-k (default k=3) most relevant documents

#### RAG Tool

The `rag_query_tool` integrates into the agent:
1. Receives user query
2. Retrieves top-3 relevant documents from FAISS
3. Augments the system prompt with retrieved evidence
4. Generates a response grounded in citations

#### RAG Evaluation

```bash
python -m evaluation.rag_evaluation
```

Compares RAG vs baseline on:
- **Keyword coverage** — does RAG improve factual content?
- **Citation count** — does RAG produce cited responses?
- **Latency overhead** — retrieval + generation time

---

## 5. Step 4: Meta-Prompting & Perplexity Evaluation

### 5a. Meta-Prompting

**File**: `evaluation/meta_prompting.py`

#### 3-Round Self-Critique Loop

```
Round 1: Initial Response
    ↓
Round 2: Self-Critique (score 5 dimensions: accuracy, completeness,
         practicality, safety, structure)
    ↓
Round 3: Refined Response (addresses all critique points)
```

#### Evaluation Dimensions

| Dimension     | What it measures                              |
|---------------|-----------------------------------------------|
| Accuracy      | Factual correctness, evidence-based claims    |
| Completeness  | Covers all relevant aspects                   |
| Practicality  | Actionable, realistic advice                  |
| Safety        | Appropriate warnings and disclaimers          |
| Structure     | Well-organized, easy to follow                |

```bash
python -m evaluation.meta_prompting
```

### 5b. Perplexity Evaluation

**File**: `evaluation/perplexity_eval.py`

#### Metrics

| Metric            | How measured                              | Interpretation                    |
|-------------------|-------------------------------------------|-----------------------------------|
| Perplexity        | exp(-1/N × Σ log P(token_i)) via logprobs | Lower = more confident            |
| Keyword Rate      | Fraction of expected keywords present     | Higher = better content coverage  |
| Safety Compliance | Correct handling of adversarial inputs    | Higher = safer                    |

```bash
python -m evaluation.perplexity_eval
```

#### Key Analysis

- Perplexity varies by technique (CoT typically lowest → most confident)
- Adversarial queries have higher perplexity (model less certain → good sign)
- Keyword match rate correlates with technique suitability per query type

---

## 6. Results Summary

All results are saved in `evaluation/results/` and `fine_tuning/results/`:

| File                          | Content                                |
|-------------------------------|----------------------------------------|
| `sensitivity_report.csv`      | Raw prompt sensitivity data            |
| `sensitivity_summary.txt`     | Aggregated sensitivity metrics         |
| `rag_comparison.csv`          | RAG vs baseline per-query data         |
| `rag_summary.txt`             | RAG improvement summary                |
| `meta_prompting_report.json`  | Full 3-round meta-prompting outputs    |
| `meta_prompting_summary.txt`  | Meta-prompting quality analysis        |
| `perplexity_report.csv`       | Per-query perplexity & quality data    |
| `perplexity_summary.txt`      | Perplexity analysis summary            |
| `comparison_report.csv`       | Baseline vs fine-tuned model data      |
| `comparison_summary.txt`      | Fine-tuning improvement summary        |

---

## 7. How to Run

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### Complete Evaluation Pipeline

```bash
# Step 2: Generate dataset
python -m data.generate_dataset

# Step 1: Prompt sensitivity
python -m evaluation.prompt_sensitivity

# Step 3a: Fine-tuning
python -m fine_tuning.prepare_finetune_data
python -m fine_tuning.run_finetune --dry-run    # validate
python -m fine_tuning.run_finetune              # launch job (costs $)
python -m fine_tuning.compare_models            # after job completes

# Step 3b: RAG evaluation
python -m evaluation.rag_evaluation

# Step 4: Meta-prompting + Perplexity
python -m evaluation.meta_prompting
python -m evaluation.perplexity_eval
```

### Run the Main App

```bash
# CLI
python app.py

# Streamlit UI
streamlit run streamlit_app.py
```
