# FITGEN.AI — Personalized AI Fitness Coach

> **A production-grade, multi-agent fitness coaching system** built with LangGraph, OpenAI, MongoDB Atlas, Redis, and Personal RAG. FITGEN.AI generates deeply personalized diet and workout plans, remembers your profile across sessions, answers per-day questions via vector search, and syncs plans to Google Calendar — all through a conversational chat interface.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [How It Works](#how-it-works)
7. [Personal RAG System](#personal-rag-system)
8. [MongoDB Schema](#mongodb-schema)
9. [Prompting Techniques](#prompting-techniques)
10. [Setup & Installation](#setup--installation)
11. [Environment Variables](#environment-variables)
12. [Running the App](#running-the-app)
13. [Running Tests](#running-tests)
14. [Fine-tuning](#fine-tuning)
15. [Demo Flow](#demo-flow)

---

## Overview

FITGEN.AI is a conversational AI fitness assistant that goes beyond simple Q&A. It conducts a structured intake interview to build your profile, generates a full personalized plan (diet **or** workout), handles follow-up edits, answers day-specific questions using vector search over your own saved plan, and persists everything across browser sessions via MongoDB.

The system uses a **deterministic router + specialist tools** architecture — a fast LLM classifier routes every user message to the correct domain tool, which then runs a multi-turn state machine. This is cheaper (90% cost savings vs. always using GPT-4), more predictable, and easier to test than pure LLM-driven agents.

---

## Key Features

| Feature | Details |
|---|---|
| **Personalized Plan Generation** | Diet and workout plans tailored to your age, weight, goals, dietary restrictions, experience level, schedule, and more |
| **Multi-Turn Profile Intake** | Guided Q&A collects all required fields before generating a plan; missing fields are asked one at a time |
| **Persistent User Profiles** | MongoDB stores your base profile, diet profile, and workout profile — returning users skip already-answered questions |
| **Personal RAG** | Your saved plans are chunked, embedded (text-embedding-3-small, 1536-dim), and stored in MongoDB Atlas Vector Search. Ask "what's my Thursday workout?" and get a precise answer |
| **Model C Plan Resolver** | Fuzzy LLM matching across active and archived plans using descriptors (e.g. "my old vegan plan"). Archived plans can be restored with one word |
| **One-Active-Plan Invariant** | Each domain (diet / workout) has exactly one active plan. Creating a new plan auto-archives the previous one and reactivates the correct RAG chunks |
| **Plan Update & Versioning** | Edit your plan in natural language ("add more protein on rest days"); changes are saved to MongoDB and the RAG index is refreshed |
| **Google Calendar Sync** | Confirm your diet plan → meals are created as calendar events via Google OAuth |
| **YouTube Video Enrichment** | Workout plans are enriched with relevant YouTube tutorial links, cached in Redis (30-day TTL) |
| **Plan Export** | Download your plan as a PDF from the Streamlit UI |
| **Plan Deletion** | Delete a plan with a confirmation gate; MongoDB and RAG index are cleaned up |
| **Active-Turn Gate** | While mid-flow (e.g., creating a diet plan), you can ask a cross-domain question ("get my workout for Tuesday") — the gate routes it as a side query and resumes the diet intake after |
| **Scope Guardrails** | Off-topic requests (e.g., "write me a Python script") are politely declined and redirected to fitness/nutrition |
| **LangSmith Tracing** | Optional end-to-end tracing of every LangGraph step, tool invocation, and LLM call |
| **Terminal CLI** | Run FITGEN.AI without Streamlit via `app.py` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Message                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     LangGraph Graph                                  │
│                                                                      │
│   START ──► Router Node ──► Tool Node(s) ──► State Sync ──► END     │
│                │                │                                    │
│                │         ┌──────┴──────┐                             │
│                │         │             │                             │
│                │    Diet Tool     Workout Tool                       │
│                │         │             │                             │
│                │    Multi-turn     Multi-turn                        │
│                │    State Machine  State Machine                     │
│                │                                                     │
│   Active-Turn Gate (LLM) decides:                                   │
│     stay | switch | side_diet | side_workout | direct               │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────┐   ┌────────────────────────┐
│         MongoDB Atlas                │   │      Redis Cloud       │
│                                      │   │                        │
│  users            ← profile store    │   │  YouTube URL cache     │
│  diet_plans       ← plan store       │   │  (30-day TTL)          │
│  workout_plans    ← plan store       │   └────────────────────────┘
│  sessions         ← context state    │
│  plan_chunks_vec  ← RAG embeddings   │
│  user_memory_vec  ← user memory      │
│  messages         ← chat history     │
└──────────────────────────────────────┘
```

### Router Node

Every user message goes through the **Router** (`agent/router.py`), which:

1. **Active-Turn Gate** — If there's an active multi-turn workflow (e.g., diet intake in progress), a fast LLM decides: `stay` (continue workflow), `switch` (abandon and start new), `side_diet` / `side_workout` (answer out-of-band without disturbing the workflow), or `direct` (pass through as a general query).

2. **Intent Classification** — Routes to the correct domain tool (`diet_tool` or `workout_tool`) with the correct intent (`create_diet`, `update_diet`, `get_diet`, `confirm_diet`, `delete_diet`, `restore_diet_plan`, etc.).

### Specialist Tools

Each domain tool (`diet_tool.py`, `workout_tool.py`) is a multi-turn state machine with 10+ handlers mapped 1-to-1 with intents. Each handler:
- Reads from `DietSessionContext` / `WorkoutSessionContext` (hydrated from AgentState + MongoDB)
- Performs its action (ask questions, generate plan, update plan, save to DB, etc.)
- Returns a `ToolMessage` with JSON `state_updates`

### State Sync Bridge

After every tool call, `state_sync.py` extracts the `state_updates` JSON from the `ToolMessage` and deep-merges it into `AgentState`. The `StateManager` then persists the updated state to MongoDB sessions.

---

## Tech Stack

### Core
| Layer | Technology |
|---|---|
| Agent Framework | **LangGraph** 0.2+ (graph execution, streaming) |
| LLM — Planning | **OpenAI gpt-5.1** (plan generation, complex reasoning) |
| LLM — Routing | **OpenAI gpt-4.1-mini** (intent classification, validation, profile extraction) |
| Embeddings | **OpenAI text-embedding-3-small** (1536 dimensions) |
| LLM Abstractions | **LangChain** 0.3+ |

### Storage
| Layer | Technology |
|---|---|
| Primary Database | **MongoDB Atlas** (users, plans, sessions, embeddings) |
| Vector Search | **MongoDB Atlas Vector Search** (cosine similarity, 1536-dim, `plan_chunks_vec`) |
| Cache | **Redis Cloud** (YouTube URL cache, 30-day TTL) |

### Frontend
| Layer | Technology |
|---|---|
| Web UI | **Streamlit** 1.35+ |
| Plan Export | **pdfkit** (PDF generation) |
| Visualization | **matplotlib** (plan charts) |
| Calendar | **Google Calendar API** (OAuth 2.0) |

### Testing
| Layer | Technology |
|---|---|
| Test Runner | **pytest** 8.0+ |
| MongoDB Mock | **mongomock** 4.1+ (in-memory MongoDB for unit tests) |
| Mocking | **pytest-mock** |
| Coverage | **pytest-cov** |

---

## Project Structure

```
FITGEN.AI/
│
├── app.py                        # Terminal CLI entry point
├── streamlit_app.py              # Streamlit web UI (3,200+ lines)
├── requirements.txt              # Python dependencies
├── pytest.ini                    # Test configuration
├── .env.example                  # Environment variable template
│
├── agent/
│   ├── state.py                  # AgentState TypedDict definition
│   ├── state_manager.py          # State hydration, merging & MongoDB persistence
│   ├── state_sync.py             # Tool JSON → AgentState bridge
│   ├── router.py                 # Active-turn gate + intent classifier
│   ├── graph.py                  # LangGraph graph definition
│   ├── config.py                 # Model tier configuration (PLAN_MODEL, FAST_MODEL)
│   ├── persistence.py            # MongoDB persistence facade
│   ├── tracing.py                # LangSmith integration
│   ├── logging_config.py         # Colored logging setup
│   ├── error_utils.py            # Structured exception handling
│   ├── feedback.py               # User feedback collection
│   ├── diet_visuals.py           # Diet plan charts
│   ├── workout_visuals.py        # Workout plan charts
│   │
│   ├── tools/
│   │   ├── __init__.py           # ALL_TOOLS registry
│   │   ├── diet_tool.py          # Diet domain — 10+ intent handlers
│   │   ├── workout_tool.py       # Workout domain — 10+ intent handlers
│   │   ├── calendar_integration.py  # Google Calendar event creation
│   │   └── youtube_service.py    # YouTube tutorial link enrichment (Redis-cached)
│   │
│   ├── db/
│   │   ├── mongo.py              # MongoClient singleton (connection pooling)
│   │   ├── models.py             # Pydantic schemas (User, Plan, Session, Chunk)
│   │   └── repositories/
│   │       ├── user_repo.py           # users collection CRUD
│   │       ├── diet_plan_repo.py      # diet_plans CRUD + archive/confirm/restore
│   │       ├── workout_plan_repo.py   # workout_plans CRUD + archive/confirm/restore
│   │       ├── session_repo.py        # sessions (context state persistence)
│   │       ├── plan_chunks_repo.py    # plan embeddings + RAG chunk management
│   │       ├── user_memory_repo.py    # persistent user memory
│   │       └── feedback_repo.py       # user ratings & feedback
│   │
│   ├── rag/
│   │   ├── retriever.py          # FAISS / NumPy vector search (static knowledge)
│   │   ├── knowledge_base.py     # Static fitness knowledge documents
│   │   └── personal/
│   │       ├── chunker.py        # Plan Markdown → sections with metadata
│   │       ├── embedder.py       # Text → OpenAI embeddings
│   │       ├── indexer.py        # Chunk indexing pipeline
│   │       ├── schema.py         # Chunk metadata schema
│   │       ├── retriever.py      # Vector search on plan_chunks_vec
│   │       └── plan_resolver.py  # Model C: fuzzy LLM match over archived plans
│   │
│   ├── cache/
│   │   └── redis_client.py       # Redis singleton + YouTube URL cache helpers
│   │
│   ├── prompts/
│   │   ├── base_prompts.py       # 6 prompting technique variants (zero-shot → decomposition)
│   │   ├── diet_prompts.py       # Diet plan generation & intake prompts
│   │   ├── workout_prompts.py    # Workout plan generation (per-day H3 format)
│   │   └── techniques.py         # Technique metadata for UI selector
│   │
│   ├── shared/
│   │   ├── types.py              # DietIntent, WorkoutIntent Literals + field lists
│   │   ├── llm_helpers.py        # Intent classification, profile extraction, plan Q&A
│   │   ├── plan_generation_loop.py  # Multi-turn plan refinement loop
│   │   ├── plan_evaluator.py     # Plan quality scoring
│   │   ├── profile_utils.py      # Profile validation & assembly
│   │   ├── plan_data.py          # Structured plan extraction from Markdown
│   │   └── response_builder.py   # Response formatting helpers
│   │
│   ├── auth/
│   │   ├── base.py               # Abstract auth class
│   │   └── google_auth.py        # Google OAuth 2.0 flow
│   │
│   └── safety/
│       └── guardrails.py         # Content filtering & scope redirect
│
├── scripts/
│   ├── backfill_plan_chunks.py   # Rebuild RAG index for existing plans
│   └── backfill_user_memory.py   # Backfill user memory collection
│
├── tests/                        # 19 test files, 220+ tests, mongomock
│   ├── conftest.py               # Shared fixtures
│   ├── test_router.py
│   ├── test_state_sync.py
│   ├── test_persistence.py
│   ├── test_plan_generation_loop.py
│   ├── test_plan_evaluator.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   ├── test_plan_chunks_repo.py
│   ├── test_auth.py
│   ├── test_conversation_workflow.py
│   └── ...
│
├── fine_tuning/                  # OpenAI fine-tuning utilities
│   ├── prepare_finetune_data.py
│   ├── run_finetune.py
│   ├── mine_plan_training_data.py
│   └── compare_models.py
│
├── flow_engineering/             # PromptFlow experiments
└── evaluation/                   # Evaluation scripts & metrics
```

---

## How It Works

### Creating a Diet Plan (end-to-end)

```
User:  "Create a high-protein vegan diet plan for muscle gain"
  │
  ▼
Router  →  intent: create_diet  →  diet_tool
  │
  ▼
diet_tool checks MongoDB profile:
  - base fields already filled? → skip those questions
  - diet-specific fields missing? → ask one at a time

User:  "I have a nut allergy and prefer Indian cooking"
User:  "I can cook for 30 minutes per day"
User:  "I eat 3 meals a day"
  │
  ▼
Profile complete → generate_plan_with_feedback()
  - gpt-5.1 generates full Markdown plan with macros table
  - Plan evaluator scores it (completeness, safety, personalization)
  - If score < threshold → refinement loop (up to 3 iterations)
  │
  ▼
Plan stored in diet_plans (status: "draft")
Plan chunked → embedded → stored in plan_chunks_vec
  │
  ▼
User:  "yes"  →  confirm_diet
  - Profile saved to users.diet_profile in MongoDB
  - Plan status → "confirmed"
  - Google Calendar sync prompt

User:  "skip"  →  done ✓

User (later):  "what should I eat for lunch on Wednesday?"
  │
  ▼
Router → get_diet → RAG retrieval on plan_chunks_vec
  - Filters by user_id, domain=diet
  - Cosine similarity search → top-k chunks for "Wednesday lunch"
  - LLM answers from retrieved chunks + full plan fallback
```

### Multi-Turn Workflow State Machine

Each domain tool tracks the current step via `workflow["step_completed"]`:

```
idle
  └─► profile_intake_started
        └─► profile_complete
              └─► diet_plan_generated  (draft in MongoDB)
                    └─► diet_plan_confirmed  (confirmed in MongoDB)
                          └─► diet_plan_synced_to_google_calendar
```

Any user message during this flow is interpreted in context (e.g., "yes" means "confirm the plan", not a generic affirmation).

---

## Personal RAG System

FITGEN.AI's most novel feature is its **personal RAG pipeline** — each user's own saved plans become a searchable knowledge base.

### Chunking

Plans are split into semantic sections by heading level:

```markdown
## Training Schedule

### Monday — Push Day
| Exercise | Sets x Reps | Rest | Notes |
...

### Thursday — Pull Day
| Exercise | Sets x Reps | Rest | Notes |
...
```

Each `### <DayName> — <Session>` block becomes one chunk with metadata:

```json
{
  "plan_id": "...",
  "user_id": "...",
  "domain": "workout",
  "section": "workout_day",
  "day_of_week": "monday",
  "text": "### Monday — Push Day\n..."
}
```

### Embedding & Storage

Chunks are embedded with `text-embedding-3-small` (1536-dim) and stored in MongoDB `plan_chunks_vec` with a vector index configured for cosine similarity.

### Retrieval

When a user asks "what's my Thursday workout?":
1. Query is embedded
2. MongoDB Atlas Vector Search runs cosine similarity over the user's chunks
3. Top-k chunks with `day_of_week=thursday` are returned
4. LLM answers from retrieved context
5. If RAG returns nothing → falls back to full plan Markdown from MongoDB (logged as `RAG empty — falling back`)

### Model C Plan Resolver

When a user references an archived plan by descriptor ("my old keto plan", "the omnivore plan from last week"):
1. `PlanResolver.resolve_plan()` fetches all active + archived plans for the user
2. A fast LLM fuzzy-matches the query against plan names and `profile_snapshot` fields
3. Returns the matched plan + `is_archived` flag
4. If archived, appends "Reply **restore** to make this your active plan"
5. On "restore" intent → archive current active plan → reactivate stored plan → swap RAG chunks

---

## MongoDB Schema

### `users` collection
```json
{
  "_id": "ObjectId",
  "email": "user@example.com",
  "base_profile": {
    "name": "Alex",
    "age": 28,
    "sex": "male",
    "height_cm": 178,
    "weight_kg": 75,
    "goal": "muscle_gain",
    "sleep_hours": 7,
    "stress_level": "medium",
    "job_type": "sedentary"
  },
  "diet_profile": {
    "diet_preference": "vegan",
    "allergies": ["nuts"],
    "meals_per_day": 3,
    "cooking_time": 30,
    "favourite_meals": "Indian food",
    "foods_to_avoid": "",
    "exercise_frequency": "4x/week",
    "alcohol_intake": "none"
  },
  "workout_profile": {
    "experience_level": "intermediate",
    "training_days_per_week": 4,
    "session_duration": 60,
    "daily_steps": 8000
  },
  "created_at": "ISODate",
  "updated_at": "ISODate"
}
```

### `diet_plans` / `workout_plans` collections
```json
{
  "_id": "ObjectId",
  "user_id": "ObjectId",
  "session_id": "uuid",
  "status": "confirmed",
  "plan_markdown": "## Your Diet Plan\n...",
  "profile_snapshot": {},
  "structured_data": {},
  "calendar_synced": false,
  "created_at": "ISODate",
  "confirmed_at": "ISODate",
  "archived_at": null
}
```

> `status` is one of: `draft` | `confirmed` | `archived`

### `plan_chunks_vec` collection
```json
{
  "_id": "ObjectId",
  "plan_id": "ObjectId",
  "user_id": "ObjectId",
  "domain": "workout",
  "section": "workout_day",
  "day_of_week": "thursday",
  "text": "### Thursday — Pull Day\n...",
  "embedding": [0.021, -0.043, "...1536 floats"],
  "status": "active",
  "created_at": "ISODate"
}
```

---

## Prompting Techniques

FITGEN.AI implements and compares **6 prompting techniques** in `agent/prompts/base_prompts.py`. These can be selected from the Streamlit UI sidebar for experimentation:

| Technique | Description |
|---|---|
| **Zero-Shot** | Role + task only, no examples. Fast and cheap. |
| **Few-Shot** | Role + 6 routing examples covering edge cases |
| **Chain-of-Thought (CoT)** | Explicit step-by-step reasoning before answering |
| **Analogical** | "Act like a fitness concierge at a 5-star hotel" framing |
| **Generate-Knowledge** | Generate relevant domain knowledge first, then answer |
| **Decomposition** | Break complex requests into sub-tasks before solving |

These are especially visible in the router's intent classification, where the technique selection changes how the LLM reasons about ambiguous inputs.

---

## Setup & Installation

### Prerequisites

- Python 3.10+ (3.11 recommended; FAISS unavailable on 3.13+, NumPy fallback used automatically)
- MongoDB Atlas account (free tier M0 works)
- Redis Cloud account (free tier works) or local Redis
- OpenAI API key
- Google Cloud project with Calendar API enabled *(optional, for calendar sync)*

### 1. Clone the repository

```bash
git clone https://github.com/sinhakushagra21/fitgenai.git
cd fitgenai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On Python 3.13+, `faiss-cpu` will not install. The system automatically falls back to a NumPy-based vector search — all features remain functional.

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 5. Set up MongoDB Atlas Vector Search index

In MongoDB Atlas, create a vector search index on the `plan_chunks_vec` collection with this definition:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    { "type": "filter", "path": "user_id" },
    { "type": "filter", "path": "domain" },
    { "type": "filter", "path": "status" },
    { "type": "filter", "path": "day_of_week" }
  ]
}
```

> Index name must be: **`plan_chunks_vector_index`**

### 6. Verify connections

```bash
python3 -c "
from dotenv import load_dotenv; load_dotenv()
from agent.db.mongo import get_db
from agent.cache.redis_client import get_redis
db = get_db()
print('MongoDB OK — collections:', db.list_collection_names())
r = get_redis(); r.ping()
print('Redis OK')
"
```

---

## Environment Variables

Create a `.env` file in the project root (**never commit this file**):

```bash
# ── Required ───────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...

# ── MongoDB ────────────────────────────────────────────────────────
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
MONGO_DB_NAME=fitgen_ai

# ── Redis ──────────────────────────────────────────────────────────
REDIS_URL=redis://default:<password>@<host>:<port>/0

# ── LangSmith Tracing (optional) ──────────────────────────────────
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=FITGEN-AI

# ── Model Overrides (optional — defaults shown) ────────────────────
FITGEN_PLAN_MODEL=gpt-5.1
FITGEN_FAST_MODEL=gpt-4.1-mini

# ── App Config (optional) ──────────────────────────────────────────
FITGEN_USER_EMAIL=you@example.com    # Pre-set email; skips email prompt
FITGEN_LOG_BUFFER=1000               # Log lines kept in Streamlit sidebar
YOUTUBE_CACHE_TTL_DAYS=30            # Redis cache TTL for YouTube links
```

---

## Running the App

### Streamlit Web UI (recommended)

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

**UI features:**
- Chat interface with streaming responses
- Prompting technique selector (sidebar)
- Real-time log viewer (sidebar)
- PDF export button (after plan is confirmed)
- Feedback / rating widget after each response
- Plan visualization charts

### Terminal CLI

```bash
python app.py
```

Runs the same LangGraph agent in a terminal REPL. Useful for debugging and scripting.

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=agent --cov-report=term-missing

# Run a specific test file
pytest tests/test_router.py -v

# Run tests matching a keyword
pytest tests/ -k "chunker" -v
```

**Test infrastructure:**
- All MongoDB interactions use `mongomock` — no live DB required
- LLM calls are mocked via `pytest-mock`
- 220+ tests across 19 files covering routing, state sync, persistence, RAG chunking, plan generation, auth, and end-to-end workflows

---

## Fine-tuning

The `fine_tuning/` directory contains utilities for fine-tuning the plan generation model on real user-confirmed plans:

```bash
# Step 1: Mine confirmed plans from MongoDB as training data
python fine_tuning/mine_plan_training_data.py

# Step 2: Prepare JSONL fine-tuning dataset
python fine_tuning/prepare_finetune_data.py

# Step 3: Launch OpenAI fine-tuning job
python fine_tuning/run_finetune.py

# Step 4: Compare base vs. fine-tuned model on sample prompts
python fine_tuning/compare_models.py
```

Fine-tuned model IDs are saved in `fine_tuning/data/` and can be set via the `FITGEN_PLAN_MODEL` env var.

---

## Demo Flow

The five scenarios below cover the full range from happy-path to edge cases:

### 1. Happy-Path Workout Plan Creation
```
"Create a workout plan"
→ [answer profile questions]
→ Plan generated with per-day sections (### Monday — Push Day, ### Thursday — Pull Day, ...)
"yes"
→ Confirmed + saved to MongoDB + RAG chunks indexed in plan_chunks_vec
"skip"
→ Workflow complete
```
*Covers: profile intake, plan generation, per-day H3 prompt format, confirmation, MongoDB + vector indexing.*

---

### 2. Day-Specific RAG Retrieval
```
"What's my Thursday workout?"
→ Vector search on plan_chunks_vec filtered by day_of_week=thursday
→ Precise answer from the retrieved chunk
```
*Covers: Personal RAG, day-tagged chunk retrieval, MongoDB fallback if RAG misses.*

---

### 3. Cross-Domain Side Query Mid-Flow
```
"Create a diet plan"
→ [mid profile intake — stops here]
"get my workout for tuesday"
→ Active-turn gate: side_workout (diet intake frozen, not abandoned)
→ Tuesday workout answered from stored workout plan
→ Diet intake resumes exactly where it left off
```
*Covers: active-turn gate logic, cross-domain side query, workflow state preservation.*

---

### 4. Model C Archived Plan Restore
```
[Have a confirmed omnivore diet plan]
"Create a vegan diet plan"
→ Generated + confirmed → omnivore plan auto-archived (status: archived)

"get my old omnivore plan"
→ PlanResolver fuzzy-matches archived plan by descriptor
→ "Reply restore to make this your active diet plan"

"restore"
→ Vegan plan archived, omnivore plan reactivated
→ RAG chunks swapped (omnivore chunks → active, vegan chunks → archived)

"What should I eat for lunch on Monday?"
→ Answers from the now-active omnivore plan
```
*Covers: one-active-plan invariant, PlanResolver, archive/restore flow, RAG chunk swap.*

---

### 5. Out-of-Scope Guardrail
```
"Write me a Python script to scrape Instagram"
→ Scope guardrail fires
→ "I'm your fitness and nutrition coach — I can only help with diet and workout planning..."
```
*Covers: safety guardrails, scope redirect.*

---

## Contributing

This is a research/capstone project. To extend it:

- **Add a new domain** (e.g., sleep coaching, mental wellness) — mirror the `diet_tool.py` + `workout_tool.py` pattern and register the tool in `agent/tools/__init__.py`
- **Add a new intent handler** — add the intent to `types.py`, add a handler method in the tool, register it in `_HANDLERS`
- **Swap the embedding model** — update `agent/rag/personal/embedder.py` and re-index via `scripts/backfill_plan_chunks.py`
- **Add a prompting technique** — add it to `agent/prompts/base_prompts.py` and `agent/prompts/techniques.py`

---

## License

MIT License — see `LICENSE` file for details.

---

*Built with LangGraph · OpenAI · MongoDB Atlas · Redis · Streamlit*
