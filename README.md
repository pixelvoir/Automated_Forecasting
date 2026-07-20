# Automated Forecasting

Point it at a dataset — a CSV/Excel upload, a local file path, or a live database
table/query — and it profiles, cleans, sets up, models, trains, and reports a forecast
end to end. An LLM agent makes the judgment calls a data scientist normally makes by hand
(outlier strategy, missing-value fills, which model suits the data) — always from computed
**statistics only**, so raw data never leaves the machine. Every LLM decision has a
deterministic rule-based fallback, so the whole pipeline runs with no LLM configured at all.

The design goal is minimal manual work: no per-dataset preprocessing, no hand-tuning.

## Quickstart

**Double-click `start_app.bat`.** On first run it creates the virtual environment and
installs dependencies (one-time); afterwards it starts both servers and opens the app at
`http://localhost:8050`. It also re-syncs dependencies automatically whenever
`requirements.txt` changes, so a fresh `git pull` never leaves you on stale packages.
`stop_app.bat` shuts everything down.

Prerequisite: Python 3.11+ on PATH. Everything binds to `127.0.0.1` only — nothing is
exposed to the network.

For development (auto-reload + debugger):
```bat
setup_venv.bat      # one-time
run_dev.bat         # terminal 1: FastAPI on :8000 (auto-reload)
run_frontend.bat    # terminal 2: Dash UI on :8050 (debug mode)
```

## What the app does

- **One confirmation, full run.** With auto-run on (default) the entire chain runs itself
  and pauses exactly once — to confirm the forecast setup it inferred — then lands on
  Results.
- **Live progress.** A stage rail and streaming log show every step as it happens, down to
  per-model backtests and neural-net training epochs; refresh or reopen the tab mid-run and
  it reattaches on its own.
- **Three-section UI** — Setup, Pipeline, Results — driven entirely by a session store, so
  switching tabs never loses state or re-runs work.
- **LLM-optional.** Cleaning, forecast intent, and model selection use an LLM when
  configured and fall back to rules when not; training and features never use one.
- **Cancellable, isolated jobs.** Heavy stages run in a subprocess that a new run preempts
  instantly and that self-terminates if the server stops — no runaway background work.
- **Past runs, reloadable.** Every run's full state is reconstructable from disk; load any
  previous run and pick up where it left off.
- **Outputs.** MASE leaderboard, interactive history/backtest/forecast charts (per-series
  drill-down on panels), CSV/Parquet forecast downloads, and an opt-in plain-language LLM
  report written for a non-technical reader.

## The forecasting pipeline

Each stage is a standalone module that reads and writes JSON, chained by a FastAPI backend;
heavy stages run in a cancellable, thread-capped subprocess.

| # | Stage | What it does |
|---|---|---|
| 1 | **Ingest** | Stream any-size CSV to Parquet, infer dtypes (numeric-as-string aware), profile shape/frequency |
| 2 | **Pre-clean EDA** | Missing/outlier/structural statistics — seasonally-aware, so real signal isn't flagged as noise |
| 2.5 | **Forecast intent** | Infers timestamp, target (incl. count-of-events), scope, series key, frequency, horizon — you confirm once |
| 3 | **Cleaning** | LLM (or rules) picks a recipe; a deterministic sanitizer guards it; the cleaner executes per-series where needed |
| 3.5 | **Validation gate** | Blocks data destruction — row loss, target-level drift, variance collapse, broken timestamps |
| 4 | **Forecast EDA** | Seasonality, trend, stationarity, intermittency, structural breaks, exogenous lead/lag |
| 5 | **Model selection** | A rule ladder ranks the catalog by data characteristics; an LLM may re-rank, never override eligibility |
| 6 | **Features** | EDA-seeded lags, rolling stats, Fourier terms, and leakage-free exogenous features |
| 7 | **Training** | Walk-forward backtests, MASE leaderboard, prediction intervals, a weighted ensemble, best model serialized |
| 8 | **Results** | Opt-in LLM insight report + forecast download |

Engineered for real data: multi-million-row streaming ingest, true multi-series panels
(thousands of series), count/event-log targets, intermittent demand, multi-seasonal series,
and confirmed intent threaded through every downstream stage.

## Model catalog

14 models, chosen per-dataset by the rule engine (an LLM can re-rank; eligibility is never
overridden): **seasonal-naive** baseline, **AutoTheta**, **AutoETS**, **AutoARIMA**,
**MSTL+Theta** (multi-seasonal), **Prophet**, **Croston** / **TSB** (intermittent demand),
**LightGBM**, **XGBoost**, **CatBoost**, **NHITS** and **PatchTST** (both pure-torch), and
**Amazon Chronos** (pretrained zero-shot). Classical models run on statsmodels/pmdarima; the
seasonal-naive baseline is always trained as the accuracy reference. Every run also builds a
free **weighted ensemble** — optimal convex weights over your best models, honestly
backtested — which competes on the leaderboard like any other model.

## Accuracy controls

- **Training window** (Forecast EDA → Advanced): train on a date sub-range or exclude
  anomalous periods without re-cleaning; a structural-break hint suggests — never applies —
  a start date. Default is always the full data.
- **Accuracy profiles** (`fast` / `balanced` / `max`): the time-vs-accuracy budget for
  tuning trials and neural-net epochs, pickable per run.
- **Exogenous drivers**: leading indicators are detected in Stage 4 and fed to the ML models
  as leakage-free features.

## Security constraints

- Raw data **never** goes to any LLM or external API — only computed statistics (~200
  tokens; the opt-in Stage 8 report additionally sends derived metrics/aggregates, never raw
  rows).
- Client DB access is **read-only**; credentials live in RAM for the duration of a request
  and are never logged or stored.
- Both servers bind to `127.0.0.1` only.

## LLM configuration

The provider is set in `config/settings.yaml → llm:` — swap between `ollama` (local, free),
`groq`, `openai`, and `gemini` without touching code; cloud providers need their key in
`.env` (see `.env.example`). An optional `llm_report:` block runs the Stage 8 report on a
bigger model. All providers go through the `openai` SDK with a `base_url` swap.

## Folder structure

```text
.
├── agents/         # LLM decision agents (cleaning, intent, model selection, results) + client
├── api/            # FastAPI routes + single-slot subprocess job manager
├── config/         # settings.yaml (all thresholds/knobs) + model_categories.yaml (catalog)
├── data/           # raw / cleaned / feature parquet files (never committed)
├── frontend/       # Dash UI (3 sections, live progress rail, session store)
├── models_lib/     # model backends (statistical, intermittent, ML, NHITS, PatchTST, Chronos, Prophet)
├── pipeline/       # stateless pipeline stages, chained via runs/{id}/*.json
├── runs/           # per-run artifacts (metadata, EDA, recipes, reports, progress)
├── models/         # per-run serialized best model + timing calibration
├── start_app.bat   # one-click launcher (stop_app.bat shuts down)
└── requirements.txt
```

Deeper architecture notes (stage contracts, JSON files, gotchas) live in `CLAUDE.md`.
