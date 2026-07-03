# Automated Forecasting

An end-to-end pipeline that takes in a dataset — CSV upload or a live database table/query — and walks it through automated cleaning and (eventually) forecasting, with an LLM agent making the judgment calls a data scientist would normally make by hand: what to do with outliers, missing values, and bad dtypes, based on the statistical shape of the data rather than hardcoded rules.

The goal is "point it at a dataset, get a forecast" — minimal manual preprocessing, no per-dataset tuning.

## How it works

```
Browser (Dash :8050)
  │
  ├─ Upload a file, or point at a DB table/query
  │
  ▼
FastAPI (:8000) — job-managed pipeline stages, run in a cancellable subprocess
  │
  ├─ 1. Ingest        → infer dtypes, profile the dataset
  ├─ 2. Pre-clean EDA → compute outlier/missing/structural-break stats (no raw values)
  ├─ 3. Cleaning      → LLM agent picks a cleaning recipe from the stats, cleaner executes it
  ├─ 3.5 Validation   → gate on row loss / series length before continuing
  └─ 4-8              → forecasting EDA, model selection, training, evaluation, results (in progress)
```

Only computed statistics (~200 tokens) are ever sent to the LLM — raw data never leaves the machine. The LLM picks strategies (drop columns, cast dtypes, `clip_iqr` vs `rolling_iqr` vs `stl_residuals` for outliers, etc.) from what the stats imply; a rule-based fallback keeps things working if no LLM is configured or reachable.

Heavy stages (ingest, EDA, cleaning, validation) run in a single-slot, preemptible subprocess so that large datasets doing pandas/STL work can't freeze the machine, and starting a new run cleanly kills whatever was in flight.

## Current status

| Stage | Status |
|---|---|
| 1 — Ingestion | ✅ implemented |
| 2 — Pre-clean EDA | ✅ implemented |
| 3 — Cleaning (LLM agent + cleaner) | ✅ implemented |
| 3.5 — Validation gate | ✅ implemented |
| 4–8 — Forecasting EDA, model selection, training, evaluation, results | 🚧 stubs, not yet implemented |

## Folder Structure

```text
.
├── agents/         # LLM-driven decision agents (cleaning, model selection, results, orchestrator)
├── api/            # FastAPI routes + subprocess job manager
├── config/         # settings.yaml — LLM provider, thresholds, per-stage tuning
├── data/           # raw/cleaned parquet files
├── frontend/       # Dash UI (tab-based, one tab per pipeline stage)
├── models_lib/     # forecasting model implementations (statistical, ML, Prophet, intermittent)
├── pipeline/       # stateless pipeline stages, chained via runs/{id}/*.json
├── runs/           # per-run artifacts (metadata, EDA output, recipes, reports)
├── tests/
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Prerequisites
- Python 3.11+

## Quickstart

**One-time setup** (creates `.venv` and installs dependencies):
```bat
setup_venv.bat
```

**Start the dev server** (auto-reloads on file changes):
```bat
run_dev.bat
```

**Health check:**
```
curl http://localhost:8000/health
```

**Start the Dash UI** (in a second terminal, after the API is running):
```bat
run_frontend.bat
```
Then open `http://localhost:8050` in your browser.

## LLM configuration

The LLM provider is set in `config/settings.yaml` and can be swapped between `ollama` (local, free), `groq`, `openai`, and `gemini` without touching code. Cloud providers need their API key in `.env`; Ollama runs locally with no key required.

> **Note on `prophet`:** It requires C++ build tools on Windows and is commented out of `requirements.txt`. See the comment there for install options. All other dependencies install cleanly via pip.
