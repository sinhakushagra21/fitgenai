#!/usr/bin/env python3
"""
fine_tuning/prepare_finetune_data.py
─────────────────────────────────────
Convert the FITGEN.AI training dataset into OpenAI fine-tuning format.

OpenAI expects JSONL with:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]}
  ]
}

For routing fine-tuning we teach the model to produce the correct
tool_call (or a polite refusal) given a user query.

Outputs:
  fine_tuning/data/finetune_train.jsonl
  fine_tuning/data/finetune_val.jsonl

Run:
    python -m fine_tuning.prepare_finetune_data                # routing (default)
    python -m fine_tuning.prepare_finetune_data --source mongo \
        --domain diet --task plan                              # defers to miner
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.prompts.base_prompts import BASE_ZERO_SHOT

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SRC = PROJECT_ROOT / "data" / "train.jsonl"
DEV_SRC   = PROJECT_ROOT / "data" / "dev.jsonl"

# ── System prompt (trimmed for fine-tuning) ───────────────────────
SYSTEM_PROMPT = BASE_ZERO_SHOT + """

You have access to two tools:
- workout_tool: for exercise, training, and workout queries
- diet_tool: for nutrition, diet, and meal planning queries

If the query is not fitness-related, respond politely without calling a tool.
"""

# ── Tool schemas (matching real tool definitions) ─────────────────
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "workout_tool",
            "description": "Expert workout and training specialist for FITGEN.AI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's workout or training question.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diet_tool",
            "description": "Expert nutrition and diet specialist for FITGEN.AI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's nutrition or diet question.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# ── Refusal templates ─────────────────────────────────────────────
REFUSAL_RESPONSES = [
    "I appreciate your question, but I'm FITGEN.AI — a fitness and nutrition coach. I can only help with workout plans, diet advice, and health-related fitness topics. How can I help you with your fitness goals?",
    "That's outside my area of expertise! I specialize in fitness, nutrition, and training. Feel free to ask me about workouts, meal plans, or exercise techniques!",
    "I'm designed to help with fitness and nutrition only. Let me know if you have any workout or diet questions — I'd love to help!",
]


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _make_training_example(example: dict, idx: int) -> dict:
    """Convert one dataset example to OpenAI fine-tuning format."""
    query = example["query"]
    expected_tool = example["expected_tool"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": query},
    ]

    if expected_tool in ("workout_tool", "diet_tool"):
        # The assistant should produce a tool call
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{idx:04d}",
                    "type": "function",
                    "function": {
                        "name": expected_tool,
                        "arguments": json.dumps({"query": query}),
                    },
                }
            ],
        }
    else:
        # The assistant should respond directly (no tool call)
        refusal = REFUSAL_RESPONSES[idx % len(REFUSAL_RESPONSES)]
        assistant_msg = {"role": "assistant", "content": refusal}

    messages.append(assistant_msg)

    return {"messages": messages, "tools": TOOL_SCHEMAS}


def _write_jsonl(path: Path, data: list[dict]) -> None:
    """Write a list of dicts to JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FITGEN.AI fine-tuning data")
    parser.add_argument(
        "--source",
        choices=["static", "mongo"],
        default="static",
        help="'static' uses data/train.jsonl (routing). 'mongo' defers to "
             "mine_plan_training_data.py for plan/intent/qa.",
    )
    parser.add_argument("--domain", choices=["diet", "workout"], default=None)
    parser.add_argument("--task", choices=["plan", "intent", "qa"], default=None)
    args = parser.parse_args()

    if args.source == "mongo":
        if not args.domain or not args.task:
            raise SystemExit("--domain and --task are required when --source=mongo")
        print(f"Delegating to mine_plan_training_data.py (domain={args.domain}, task={args.task})")
        subprocess.run(
            [sys.executable, "-m", "fine_tuning.mine_plan_training_data",
             "--domain", args.domain, "--task", args.task],
            check=True,
        )
        return

    print("Loading source data...")
    train_raw = _load_jsonl(TRAIN_SRC)
    dev_raw = _load_jsonl(DEV_SRC)

    print(f"  train: {len(train_raw)} examples")
    print(f"  dev:   {len(dev_raw)} examples")

    train_ft = [_make_training_example(ex, i) for i, ex in enumerate(train_raw)]
    val_ft = [_make_training_example(ex, i + len(train_raw)) for i, ex in enumerate(dev_raw)]

    train_path = DATA_DIR / "finetune_train.jsonl"
    val_path = DATA_DIR / "finetune_val.jsonl"

    _write_jsonl(train_path, train_ft)
    _write_jsonl(val_path, val_ft)

    # Stats
    tool_calls_train = sum(1 for ex in train_ft if ex["messages"][-1].get("tool_calls"))
    direct_train = len(train_ft) - tool_calls_train
    tool_calls_val = sum(1 for ex in val_ft if ex["messages"][-1].get("tool_calls"))
    direct_val = len(val_ft) - tool_calls_val

    print(f"\n✅ Fine-tuning data prepared!")
    print(f"  {train_path}  ({len(train_ft)} examples: {tool_calls_train} tool-calls, {direct_train} direct)")
    print(f"  {val_path}  ({len(val_ft)} examples: {tool_calls_val} tool-calls, {direct_val} direct)")


if __name__ == "__main__":
    main()
