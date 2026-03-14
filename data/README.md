# Data directory — FITGEN.AI evaluation dataset

This directory contains the curated evaluation dataset for FITGEN.AI.

## Format

Each file is [JSONL](https://jsonlines.org/) (one JSON object per line):

| Field                        | Type       | Description                                        |
|------------------------------|------------|----------------------------------------------------|
| `query`                      | `str`      | User's input query                                 |
| `expected_tool`              | `str`      | `"workout_tool"`, `"diet_tool"`, or `"none"`       |
| `expected_response_contains` | `[str]`    | Substrings expected in a good response             |
| `category`                   | `str`      | `"typical"`, `"edge"`, or `"adversarial"`          |

## Files

| File                 | Purpose              | Count |
|----------------------|----------------------|-------|
| `full_dataset.jsonl` | All examples (order) | ~100  |
| `train.jsonl`        | Training split (70%) | ~70   |
| `dev.jsonl`          | Development (15%)    | ~15   |
| `test.jsonl`         | Test (15%)           | ~15   |

## Regenerate

```bash
python -m data.generate_dataset
```

## Categories

- **Typical** (60): Clear workout or diet queries with unambiguous routing
- **Edge** (20): Ambiguous / multi-domain / borderline queries
- **Adversarial** (20): Off-topic, jailbreak attempts, harmful/unsafe requests
