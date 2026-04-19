#!/usr/bin/env python3
"""
fine_tuning/run_finetune.py
────────────────────────────
Launch an OpenAI fine-tuning job for FITGEN.AI.

Supports multiple tasks and domains:

  Routing (legacy, unchanged):
      python -m fine_tuning.run_finetune --task routing

  Per-domain plan / intent / qa (new):
      python -m fine_tuning.run_finetune --task plan --domain diet
      python -m fine_tuning.run_finetune --task plan --domain workout
      python -m fine_tuning.run_finetune --task intent --domain diet
      python -m fine_tuning.run_finetune --task qa --domain workout

Prerequisites:
  - OPENAI_API_KEY set in .env
  - For routing: prepare_finetune_data.py produced finetune_{train,val}.jsonl
  - For plan/intent/qa: mine_plan_training_data.py produced
    {domain}_{task}_{train,val}.jsonl

Dry-run (validate data only, no submission):
    python -m fine_tuning.run_finetune --task plan --domain diet --dry-run
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
MODEL_ID_REGISTRY = FT_DATA_DIR / "finetuned_model_ids.json"
LEGACY_MODEL_ID_FILE = FT_DATA_DIR / "finetuned_model_id.txt"  # backward compat

# Base model defaults per task. gpt-4.1-mini for plan/intent/qa (matches plan
# decisions). gpt-4o-mini kept for routing (legacy, existing data already
# uses this base so swapping would invalidate prior benchmarks).
BASE_MODELS = {
    "plan": "gpt-4.1-mini-2024-07-18",
    "intent": "gpt-4.1-mini-2024-07-18",
    "qa": "gpt-4.1-mini-2024-07-18",
    "routing": "gpt-4o-mini-2024-07-18",
}


def _resolve_files(task: str, domain: str | None) -> tuple[Path, Path]:
    """Resolve the train + val JSONL paths for the given task/domain."""
    if task == "routing":
        return (
            FT_DATA_DIR / "finetune_train.jsonl",
            FT_DATA_DIR / "finetune_val.jsonl",
        )
    if not domain:
        raise SystemExit(f"--domain is required for task={task!r}")
    return (
        FT_DATA_DIR / f"{domain}_{task}_train.jsonl",
        FT_DATA_DIR / f"{domain}_{task}_val.jsonl",
    )


def _suffix(task: str, domain: str | None) -> str:
    """Short suffix passed to OpenAI for identifying the run."""
    if task == "routing":
        return "fitgen-router"
    return f"fitgen-{domain}-{task}"


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


def _load_registry() -> dict:
    if MODEL_ID_REGISTRY.exists():
        try:
            return json.loads(MODEL_ID_REGISTRY.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  ⚠️  {MODEL_ID_REGISTRY.name} is malformed — recreating.")
    return {
        "_comment": "Registry of fine-tuned model IDs.",
        "diet": {"plan": None, "intent": None, "qa": None},
        "workout": {"plan": None, "intent": None, "qa": None},
        "routing": None,
    }


def _save_registry(task: str, domain: str | None, model_id: str) -> None:
    reg = _load_registry()
    if task == "routing":
        reg["routing"] = model_id
    else:
        reg.setdefault(domain, {})[task] = model_id
    MODEL_ID_REGISTRY.write_text(
        json.dumps(reg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"   Registry updated: {MODEL_ID_REGISTRY}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch FITGEN.AI fine-tuning job")
    parser.add_argument(
        "--task",
        choices=["plan", "intent", "qa", "routing"],
        default="routing",
        help="Fine-tuning task (default: routing — the legacy pipeline).",
    )
    parser.add_argument(
        "--domain",
        choices=["diet", "workout"],
        default=None,
        help="Domain for plan/intent/qa tasks. Ignored for routing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate data only, don't submit job")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    args = parser.parse_args()

    train_file, val_file = _resolve_files(args.task, args.domain)
    base_model = BASE_MODELS[args.task]
    suffix = _suffix(args.task, args.domain)

    # 1. Validate
    print(f"\n1. Validating fine-tuning data for task={args.task} domain={args.domain or '-'}...")
    for path in (train_file, val_file):
        if not path.exists():
            print(f"  ❌ {path} not found.")
            if args.task == "routing":
                print("     Run prepare_finetune_data.py first.")
            else:
                print(f"     Run: python -m fine_tuning.mine_plan_training_data "
                      f"--domain {args.domain} --task {args.task}")
            sys.exit(1)
        count = _validate_data(path)
        print(f"  ✓ {path.name}: {count} valid examples")

    if args.dry_run:
        print("\n✅ Dry run complete — data is valid!")
        return

    # 2. Upload files
    client = OpenAI()
    print("\n2. Uploading files to OpenAI...")
    train_id = _upload_file(client, train_file)
    val_id = _upload_file(client, val_file)

    # 3. Create fine-tuning job
    print(f"\n3. Creating fine-tuning job...")
    print(f"   Base model: {base_model}")
    print(f"   Suffix:     {suffix}")
    print(f"   Epochs:     {args.epochs}")

    job = client.fine_tuning.jobs.create(
        training_file=train_id,
        validation_file=val_id,
        model=base_model,
        hyperparameters={"n_epochs": args.epochs},
        suffix=suffix,
    )
    print(f"   Job ID: {job.id}")

    # 4. Wait for completion
    result = _wait_for_job(client, job.id)

    if result["status"] == "succeeded":
        model_id = result["model"]
        print(f"\n✅ Fine-tuning succeeded!")
        print(f"   Model ID: {model_id}")
        print(f"   Trained tokens: {result.get('trained_tokens', 'N/A')}")

        _save_registry(args.task, args.domain, model_id)

        # Keep the legacy flat file updated for routing so older tooling
        # (compare_models.py --no-args path) keeps working.
        if args.task == "routing":
            LEGACY_MODEL_ID_FILE.write_text(model_id)
            print(f"   Legacy file updated: {LEGACY_MODEL_ID_FILE}")
    else:
        print(f"\n❌ Fine-tuning {result['status']}: {result.get('error', 'unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
