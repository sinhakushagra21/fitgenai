#!/usr/bin/env python3
"""
fine_tuning/run_finetune.py
────────────────────────────
Launch an OpenAI fine-tuning job for FITGEN.AI.

This script:
  1. Uploads training and validation files to OpenAI
  2. Creates a fine-tuning job on gpt-4o-mini-2024-07-18
  3. Polls for completion
  4. Saves the fine-tuned model ID

Prerequisites:
  - OPENAI_API_KEY set in .env
  - Run prepare_finetune_data.py first

Run:
    python -m fine_tuning.run_finetune
    python -m fine_tuning.run_finetune --dry-run   # validate only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────
FT_DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_FILE = FT_DATA_DIR / "finetune_train.jsonl"
VAL_FILE = FT_DATA_DIR / "finetune_val.jsonl"
MODEL_ID_FILE = FT_DATA_DIR / "finetuned_model_id.txt"

# Fine-tuning base model (gpt-4o-mini supports fine-tuning)
BASE_MODEL = "gpt-4o-mini-2024-07-18"


def _validate_data(path: Path) -> int:
    """Basic validation of fine-tuning JSONL file."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                assert "messages" in obj, f"Line {i}: missing 'messages'"
                msgs = obj["messages"]
                assert len(msgs) >= 2, f"Line {i}: need at least 2 messages"
                assert msgs[0]["role"] == "system", f"Line {i}: first msg must be system"
                assert msgs[-1]["role"] == "assistant", f"Line {i}: last msg must be assistant"
                count += 1
            except Exception as e:
                print(f"  ❌ Validation error in {path.name} line {i}: {e}")
                raise
    return count


def _upload_file(client: OpenAI, path: Path, purpose: str = "fine-tune") -> str:
    """Upload a file to OpenAI and return the file ID."""
    print(f"  Uploading {path.name}...", end="", flush=True)
    with open(path, "rb") as f:
        response = client.files.create(file=f, purpose=purpose)
    print(f"  ✓ {response.id}")
    return response.id


def _wait_for_job(client: OpenAI, job_id: str, poll_interval: int = 30) -> dict:
    """Poll the fine-tuning job until it completes or fails."""
    print(f"\n  Monitoring job {job_id}...")
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        print(f"  [{time.strftime('%H:%M:%S')}] Status: {status}")

        if status == "succeeded":
            return {
                "status": "succeeded",
                "model": job.fine_tuned_model,
                "trained_tokens": getattr(job, "trained_tokens", None),
            }
        elif status in ("failed", "cancelled"):
            return {
                "status": status,
                "error": getattr(job, "error", None),
            }

        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch FITGEN.AI fine-tuning job")
    parser.add_argument("--dry-run", action="store_true", help="Validate data only, don't submit job")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    args = parser.parse_args()

    # 1. Validate
    print("\n1. Validating fine-tuning data...")
    for path in (TRAIN_FILE, VAL_FILE):
        if not path.exists():
            print(f"  ❌ {path} not found. Run prepare_finetune_data.py first.")
            sys.exit(1)
        count = _validate_data(path)
        print(f"  ✓ {path.name}: {count} valid examples")

    if args.dry_run:
        print("\n✅ Dry run complete — data is valid!")
        return

    # 2. Upload files
    client = OpenAI()
    print("\n2. Uploading files to OpenAI...")
    train_id = _upload_file(client, TRAIN_FILE)
    val_id = _upload_file(client, VAL_FILE)

    # 3. Create fine-tuning job
    print(f"\n3. Creating fine-tuning job...")
    print(f"   Base model: {BASE_MODEL}")
    print(f"   Epochs: {args.epochs}")

    job = client.fine_tuning.jobs.create(
        training_file=train_id,
        validation_file=val_id,
        model=BASE_MODEL,
        hyperparameters={"n_epochs": args.epochs},
        suffix="fitgen-router",
    )
    print(f"   Job ID: {job.id}")

    # 4. Wait for completion
    result = _wait_for_job(client, job.id)

    if result["status"] == "succeeded":
        model_id = result["model"]
        print(f"\n✅ Fine-tuning succeeded!")
        print(f"   Model ID: {model_id}")
        print(f"   Trained tokens: {result.get('trained_tokens', 'N/A')}")

        # Save model ID
        MODEL_ID_FILE.write_text(model_id)
        print(f"   Saved to: {MODEL_ID_FILE}")
    else:
        print(f"\n❌ Fine-tuning {result['status']}: {result.get('error', 'unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
