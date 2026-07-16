# Automated Forecasting

An end-to-end pipeline that takes any dataset — CSV/Excel upload, a local file path, or a live database table/query — and walks it through profiling, cleaning, forecast setup, model selection, training and reporting. An LLM agent makes the judgment calls a data scientist would normally make by hand (outlier strategy, missing-value fills, which model suits the data), always from computed **statistics only** — raw data never leaves the machine. Every LLM decision has a deterministic rule-based fallback, so the pipeline works with no LLM configured at all.

The goal: **point it at a dataset, get a forecast** — minimal manual preprocessing, no per-dataset tuning.

## Quickstart

**Double-click `start_app.bat`.** It creates the virtual environment on first run (one-time, downloads dependencies), starts both servers minimized, and opens the app in your browser at `http://localhost:8050`. Shut everything down with `stop_app.bat`.

Prerequisite: Python 3.11+ on PATH. Everything binds to `127.0.0.1` only — nothing is exposed to the network.

For development (auto-reload + debugger):
```bat
setup_venv.bat      # one-time
run_dev.bat         # terminal 1: FastAPI on :8000 (auto-reload)
run_frontend.bat    # terminal 2: Dash UI on :8050 (debug mode)
```

## How it works

```
Browser (Dash :8050)  —  3 sections: Setup · Pipeline · Results
  │
  ▼
FastAPI (:8000) — heavy stages run in a cancellable, below-normal-priority subprocess
  │
  ├─ 1    Ingest          → stream to parquet, infer dtypes, profile
  ├─ 2    Pre-clean EDA   → outlier/missing/structural stats (statistics only)
  ├─ 2.5  Forecast intent → suggests timestamp/target/scope/series-key/frequency/
  │                          horizon from the evidence; YOU confirm once
  ├─ 3    Cleaning        → LLM (or rules) picks a recipe; deterministic sanitizer
  │                          guards it; cleaner executes per-series where needed
  ├─ 3.5  Validation gate → blocks data destruction (row loss, target drift, …)
  ├─ 4    Forecast EDA    → seasonality/trend/stationarity/intermittency battery
  ├─ 5    Model selection → rule ladder + optional LLM re-rank over the catalog
  ├─ 6    Features        → EDA-seeded lags/rolling/Fourier/exog features (no LLM)
  ├─ 7    Training        → walk-forward backtests, MASE leaderboard, intervals,
  │                          top-2 ensemble row, best model serialized
  └─ 8    Results         → opt-in LLM insight report + CSV/parquet forecast download
```

With **auto-run** on (default), the whole chain runs itself with exactly one pause — the forecast-setup confirmation — and lands on the Results tab.

## Model catalog

12 models, selected per-dataset by a rule engine (an LLM can re-rank, never override eligibility): seasonal-naive baseline, AutoTheta, AutoETS, AutoARIMA, MSTL+Theta (multi-seasonal), Prophet, Croston/TSB (intermittent demand), LightGBM, XGBoost, NHITS (pure-torch), and Amazon Chronos (pretrained zero-shot). Classical models run on statsmodels/pmdarima; a seasonal-naive baseline is always trained as the MASE reference. Training also adds a free **Ensemble (top 2)** leaderboard row — the mean of your two best models' forecasts, honestly backtested.

## Accuracy controls

- **Training range** (Forecast EDA tab → Advanced): train on a date sub-range (e.g. cut history before a detected level shift) and/or exclude anomalous windows (e.g. a disrupted year). Default is always the full data; a structural-break hint suggests — never applies — a start date. Changing the window re-runs Stage 4+ without re-cleaning.
- **Accuracy profiles** (`fast` / `balanced` / `max`): the time-vs-accuracy budget for tuning trials and epochs, pickable per training run.
- **Exogenous drivers**: leading indicators are detected in Stage 4 and fed to the ML models as leakage-free features.

## Security constraints

- Raw data **never** goes to any LLM or external API — only computed statistics (~200 tokens; the opt-in Stage 8 report additionally sends derived metrics/aggregates, never raw rows).
- Client DB access is **read-only**; credentials live in RAM for the duration of a request and are never logged or stored.
- Both servers bind to `127.0.0.1` only.

## Performance on small machines

Heavy stages run in a single-slot subprocess at **below-normal OS priority** with BLAS/LightGBM/XGBoost/torch capped to `cores − 1` threads (configurable under `resources:` in `config/settings.yaml`), so a big training run slows the machine down instead of freezing it. Starting a new run preempts (kills) the previous job immediately. `start_app.bat` runs both servers without dev file-watchers, which also saves constant background CPU.

## LLM configuration

Provider is set in `config/settings.yaml → llm:` — swap between `ollama` (local, free), `groq`, `openai`, `gemini` without touching code; cloud providers need their key in `.env` (see `.env.example`). An optional `llm_report:` block runs the Stage 8 report on a bigger model. All providers go through the `openai` SDK with a `base_url` swap.

## Folder structure

```text
.
├── agents/         # LLM decision agents (cleaning, intent, model selection, results) + client
├── api/            # FastAPI routes + single-slot subprocess job manager
├── config/         # settings.yaml (all thresholds/knobs) + model_categories.yaml (catalog)
├── data/           # raw / cleaned / feature parquet files (never committed)
├── frontend/       # Dash UI (3 sections, live progress rail, session store)
├── models_lib/     # model backends (statistical, intermittent, ML, NHITS, Chronos, Prophet)
├── pipeline/       # stateless pipeline stages, chained via runs/{id}/*.json
├── runs/           # per-run artifacts (metadata, EDA, recipes, reports, progress)
├── models/         # per-run serialized best model + timing calibration
├── start_app.bat   # one-click launcher (stop_app.bat shuts down)
└── requirements.txt
```

Deeper architecture notes (stage contracts, JSON files, gotchas) live in `CLAUDE.md`.
