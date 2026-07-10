# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bat
setup_venv.bat          # one-time: create .venv and install deps
run_dev.bat             # terminal 1: FastAPI on :8000 (auto-reload)
run_frontend.bat        # terminal 2: Dash UI on :8050
```

Health check: `curl http://localhost:8000/health`

No test suite (`tests/` is an empty scaffold), no linter config. To run pipeline stages manually:
```python
from pipeline import ingest, pre_clean_eda, cleaner, validation_gate
from agents import cleaning_agent

result = ingest.run(file_path="path/to/data.csv")   # or table/query from DB
run_id = result["run_id"]
pre_clean_eda.run(run_id)
cleaning_agent.run(run_id, use_llm=True)   # LLM or rule-based fallback
cleaner.run(run_id)
validation_gate.run(run_id)

from pipeline import forecast_intent, forecasting_eda
from agents import forecast_intent_agent
forecast_intent.detect(run_id)               # Stage 2.5: rule suggestions (before cleaning!)
forecast_intent_agent.refine(run_id)         # optional LLM refinement of the suggestions
# write runs/{run_id}/forecast_user_selections.json to confirm/override, then:
cleaning_agent.run(run_id); cleaner.run(run_id); validation_gate.run(run_id)
forecasting_eda.run(run_id)                  # Stage 4: full stats + model_selection_payload.json

from agents import model_selector_agent
model_selector_agent.run(run_id, use_llm=True)  # Stage 5: rule engine always runs, LLM may override

from pipeline import feature_builder, trainer
feature_builder.run(run_id, build_ml_matrix=True)          # Stage 6: EDA-seeded features (no LLM)
trainer.run(run_id, models=["lightgbm", "xgboost"],        # Stage 7: train + backtest chosen models
            profile="balanced")                            #   optional fast|balanced|max override

from agents import results_agent
results_agent.run(run_id, "lightgbm")   # Stage 8: opt-in LLM insight report (LLMError if no key)
```
Note the order: intent (Stage 2.5) comes BEFORE cleaning — the recipe and the per-series
execution both consume the confirmed intent.

LLM provider is configured in `config/settings.yaml` — switch between `ollama`, `openai`, `gemini`, `groq` without touching code. Ollama runs locally; cloud providers need their key in `.env`.

`prophet` (1.3 wheels, cmdstanpy) and `chronos-forecasting` (Amazon's zero-shot transformer, ~190MB HF weight download on first use) are both installed and working; if either install fails on a new machine, the catalog's runtime `find_spec` gate marks the card unavailable and nothing else breaks.

---

## Security Constraints (non-negotiable)

- Raw data **never** sent to any LLM or external API — only computed statistics (~200 tokens). The opt-in Stage 8 report additionally sends *derived* outputs (metrics, thinned forecast arrays, per-series summary totals + series names) — still never raw rows, and only when the user explicitly clicks Generate
- Client DB is **read-only** — no writes ever
- DB credentials **never** logged, stored to disk, or forwarded anywhere
- Credential path: browser input → POST body → FastAPI RAM → connection → disposed
- SQLAlchemy engines use `poolclass=NullPool`; `engine.dispose()` always in `finally`
- Both servers bind to `127.0.0.1` only

---

## Architecture

### Request / Data Flow

```
Browser (Dash :8050)
  │
  ├─ User action → Dash callback (callbacks.py)
  │                   │
  │                   └─ HTTP → FastAPI (:8000) → api/jobs.py job slot → pipeline module
  │                                                    │      (subprocess, cancellable)
  │                                                    ├─ reads/writes runs/{run_id}/*.json
  │                                                    └─ reads/writes data/raw | data/cleaned
  │
  └─ results-store (dcc.Store, session storage) drives all UI state
       └─ render_tab() rebuilds the active tab's content on every store/tab change
```

### Job Manager — heavy stages run in a cancellable subprocess

`api/jobs.py` runs every heavy pipeline stage (ingest, pre-clean EDA, clean, validate) in a **single-slot subprocess**, not in-process. This exists because large datasets (multi-million rows) doing pandas/STL work in-thread could pin every CPU core and freeze the whole machine, with no way to stop it once started.

- Only **one** job runs at a time. Starting a new job **preempts (kills)** whatever is currently running — this is deliberate: switching datasets or starting a new run should stop old work immediately, not queue behind it.
- `api/tasks.py` holds the top-level, picklable task functions (`ingest_task`, `eda_task`, `clean_task`, `validate_task`) that the subprocess actually runs — kept separate from `api/routes.py` because functions passed to `multiprocessing` must be importable at module level.
- `api/routes.py`'s `_run_job()` helper wraps `jobs.run_job()` and maps `JobCancelled` → HTTP 409, `JobError` → HTTP 500.
- `POST /runs/cancel` explicitly terminates the active job. The frontend calls this whenever the user switches datasets, loads a past run, or resets, so old work never keeps burning CPU in the background.
- A 409 from any stage means "preempted by a newer action" — callbacks must show this visibly, never swallow it silently (a past bug: silent 409s looked exactly like "nothing happens" from the UI).

### Pipeline Stages

Each stage is a standalone Python module in `pipeline/` with a single `run(run_id)` entry point. Stages chain by writing JSON files that the next stage reads — no shared state in memory between requests.

| Stage | Module | Writes |
|---|---|---|
| 1 — Ingestion | `pipeline/ingest.py` | `runs/{id}/metadata.json`, `data/raw/{id}_raw.parquet` |
| 2 — Pre-clean EDA | `pipeline/pre_clean_eda.py` | `runs/{id}/pre_clean_eda_full.json`, `runs/{id}/cleaning_decision_payload.json` |
| 3 — Cleaning | `agents/cleaning_agent.py` → `pipeline/cleaner.py` | `cleaning_recipe.json`, `cleaning_status.json`, `data/cleaned/{id}_cleaned.parquet`, `cleaning_report.json`, `cleaned_metadata.json` |
| 2.5 — Forecast intent | `pipeline/forecast_intent.py` (`detect()`) → `agents/forecast_intent_agent.py` (`refine()`) | `runs/{id}/forecast_intent.json` (suggestions + confidence + LLM rationale) |
| 3.5 — Validation | `pipeline/validation_gate.py` | `runs/{id}/validation_gate.json` |
| 4 — Forecast EDA | `pipeline/forecasting_eda.py` (`run()`) | `forecasting_eda_full.json`, `model_selection_payload.json` |
| 5 — Model selection | `pipeline/rule_engine.py` (`decide()`/`run()`) → `agents/model_selector_agent.py` (`run()`) | `runs/{id}/model_selection.json` |
| 6 — Feature engineering | `pipeline/feature_builder.py` (`run()`) | `data/features/{id}_model_frame.parquet` (+ `_features.parquet` when ML), `runs/{id}/feature_report.json` |
| 7 — Training | `pipeline/trainer.py` (`run()`) → `models_lib/*` via `models_lib/registry.py`; metrics in `pipeline/evaluator.py` | `runs/{id}/training_report.json`, `models/{id}/best_model.pkl` + `preprocessor.pkl` |
| 8 — Results | `agents/results_agent.py` (`run(run_id, model_id)`) — opt-in LLM insight report + forecast download (`frontend/results_view.py`) | `runs/{id}/results_report.json` |

**Still-corrupt stubs** (out of scope, nothing imports them): `pipeline/decision_extractor.py`, `agents/orchestrator.py` — literal PowerShell `` `r`n `` escapes that raise SyntaxError on import; **rewrite fully, never edit**. (`agents/results_agent.py` was rewritten this way 2026-07-10 and is now the real Stage 8.)

**Intent-first flow:** Stage 2.5 suggests every forecast choice (timestamp, target incl.
count-of-events, scope, series key, exog, frequency, horizon) from Stage 1/2 evidence +
the RAW parquet; the user confirms once on the Pipeline Setup tab; `POST /clean` writes
`forecast_user_selections.json` and the whole chain (clean → validate → forecast EDA →
model select) consumes it. **Scope, frequency and horizon can be adjusted later without
re-cleaning** (`POST /forecast-eda` overrides, which re-chains model selection) — cleaner.py
never reads any of the three; only the series key / target / agg affect cleaning, so those
stay on Pipeline Setup. The Forecast EDA tab's re-run row carries the scope/freq/horizon
controls (per-series is disabled there when no series key was confirmed — that needs a
re-clean); Pipeline Setup notes the split.

Stage 3 is two-step: the LLM agent decides the recipe (`cleaning_agent.py`), then `cleaner.py` executes it. Both are called sequentially inside `api/tasks.py::clean_task`, which the `/runs/{id}/clean` endpoint runs via the job manager. The LLM only receives `cleaning_decision_payload.json`.

**Stage 2 produces two files:** `pre_clean_eda_full.json` (full stats for UI display) and `cleaning_decision_payload.json` (counts/percentages/flags plus a compact per-column profile — the one sent to the LLM). Column names appear in the payload to key the recipe, but no raw values ever.

**Frequency inference (`ingest.py::_infer_frequency`) deduplicates timestamps first** — it measures the time *grid*, not row spacing. Panel/multi-series data has many rows per timestamp (e.g. ~1,800 store×product rows per day); the median raw row-to-row diff is then 0 and daily panels were mislabeled "hourly" (which also mis-sized every seasonal window downstream, and made ETL audit columns win the most-granular-timestamp fallback pick).

**Key signal in Stage 2 payload:** `temporal_pct << iqr_pct` means global outlier rate is inflated by seasonal patterns — the LLM should pick `rolling_iqr` or `stl_residuals` instead of `clip_iqr`.

### Dtype Inference (Stage 1) — numeric-as-string columns

`pipeline/ingest.py::_infer_dtype()` checks numeric coercion *before* datetime for object-dtype columns (`pd.to_numeric(sample, errors="coerce").notna().mean() > 0.8`). This matters: CSV/DB columns are routinely read as strings even when the values are numeric, and if Stage 1 misclassifies them as `"categorical"`, Stage 2's outlier detection (`pre_clean_eda.py::_outlier_counts()`, which only loops over columns Stage 1 called `"numeric"`) silently skips them entirely — producing a cleaning recipe that says `"keep"` for every outlier strategy with no real signal behind it. `extract_metadata()`'s numeric_stats and `_outlier_counts()` both re-coerce with `pd.to_numeric(df[col], errors="coerce")` (no-op for already-numeric columns) rather than trusting the raw dtype.

**Local files: the path input is the large-file route (2026-07-09).** The ingestion form has a "local file path" text box that goes straight into the `{"source": "file", "file_path"}` flow — zero copies, any size. The drag-drop `dcc.Upload` base64-encodes the WHOLE file in browser + Dash memory (a 254MB CSV became ~930MB of transient copies in `_save_upload` and OOM'd the Dash process before ingest ever ran), so it is capped: `max_size=100MB` on the component (oversize files are silently ignored by dcc.Upload — the visible copy carries the limit) + `MAX_CONTENT_LENGTH=150MB` on the Flask server (werkzeug rejects on the Content-Length header before buffering; clean 413 handler). Path wins over upload when both are provided.

**CSV ingest is streamed, never fully loaded** (`_stream_csv_to_parquet`): `pd.read_csv` on a large CSV (e.g. 254MB) builds a ~1GB+ DataFrame and OOM-crashes the box. Instead pyarrow's `csv.open_csv` streams the file to Parquet block-by-block (peak = one `csv_block_size_mb` block, default 64MB) — and crucially pyarrow infers **one** schema for the whole file and applies it to every block, unlike pandas' `chunksize` which infers per-chunk and diverges across chunk boundaries (a real ParquetWriter-crash bug a prior attempt hit). If inferred typing conflicts across blocks (a column pure-int early, a real string later → `ArrowInvalid`), it retries reading every column as string (downstream re-coerces, per the numeric-as-string note above). Metadata is then computed **column-by-column from the Parquet** (`extract_metadata_from_parquet` — peak = one column, exact stats incl. a bounded-memory duplicate count via row hashes) rather than materializing the frame. DB and non-CSV file paths still go through pandas `extract_metadata()`.

### LLM Client

`agents/llm_client.py` wraps all four providers (ollama / openai / gemini / groq) through the `openai` Python package with a `base_url` swap. Provider and model come from `config/settings.yaml`. Callers must catch `LLMError` and fall back — `cleaning_agent.py` has a rule-based fallback that runs when the LLM is unavailable, or when `cleaning_agent.run(run_id, use_llm=False)` is called deliberately.

`cleaning_agent.run()` always writes `cleaning_status.json` (`{"recipe_source": "llm"|"fallback", "recipe_error": str|None}`) alongside `cleaning_recipe.json`, so the actual LLM failure reason survives a reload via `/runs/{id}/summary` instead of showing as `"unknown"`. It also force-corrects `type_fix` to `cast_numeric`/`parse_datetime` for any column Stage 2 flagged in `dtype_issues`, as a safety net in case an LLM response ignores the signal.

**Profiles (2026-07-10):** `call(messages, require_json=..., profile="report")` merges the optional `llm_report:` settings block over `llm:` (absent block → `llm:` used entirely) — this is how Stage 8's report runs on a bigger model/longer timeout without touching the pipeline agents. Report-profile key resolution, first non-empty wins: `api_key_env` named in the block → `REPORT_<PROVIDER>_API_KEY` → the provider's standard env var. The Stage 8 report is the one deliberate exception to the "callers must fall back" rule: it is LLM-only (`results_agent.run` lets `LLMError` propagate → HTTP 502, shown verbatim).

### API Routes

All routes are under `/runs` (defined in `api/routes.py`):

| Endpoint | Method | Purpose |
|---|---|---|
| `/runs` | GET | List all past runs (newest first) |
| `/runs` | POST | Stage 1 — ingest data (table / query / file), job-managed |
| `/runs/tables` | GET | List DB tables (env var creds) |
| `/runs/tables-with-creds` | POST | List DB tables (request body creds, never stored) |
| `/runs/cancel` | POST | Terminate whatever heavy job is currently running, if any |
| `/runs/{id}` | DELETE | Delete run directory + parquets |
| `/runs/{id}/status` | GET | Check Stage 1 complete |
| `/runs/{id}/metadata` | GET | Stage 1 metadata |
| `/runs/{id}/summary` | GET | All accumulated run data (used to reconstruct UI state) |
| `/runs/{id}/pre-clean-eda` | POST | Stage 2 — pre-clean EDA, job-managed |
| `/runs/{id}/clean` | POST | Stage 3 — cleaning agent + cleaner, job-managed (`use_llm` body flag) |
| `/runs/{id}/validate` | POST | Stage 3.5 — validation gate only, job-managed |
| `/runs/{id}/forecast-intent` | POST | Stage 2.5 — detect + LLM-refine intent suggestions (`use_llm` body flag), job-managed |
| `/runs/{id}/forecast-eda` | POST | Stage 4 — forecast EDA from confirmed intent (body = optional `scope`/`forecast_frequency`/`horizon`/`exog_cols` overrides — none re-clean; per-series without a confirmed series key → 400; exog cols not in the cleaned data → 400), job-managed |
| `/runs/{id}/series-forecast` | GET | One series' backtest+forecast arrays for one trained model (`?model=&series=`), from `models/{id}/series_forecasts.parquet` — per-series viewer; lightweight, NOT job-managed |
| `/runs/{id}/model-select` | POST | Stage 5 — rule engine + optional LLM override (`use_llm` body flag); 400 if Stage 4 hasn't produced the payload; job-managed |
| `/runs/{id}/train` | POST | Stages 6+7 — feature engineering + training of the user-selected `models` (body list; Stage 5 pick is only the default) with an optional `profile` override (`fast`/`balanced`/`max`). 400 if Stage 5 hasn't run, if no trainable model, or — for a heavy per-series panel — if the estimate is `heavy` and `confirm_heavy` is false (the 400 detail is a dict carrying the estimate). Deletes any stale `results_report.json`. Job-managed |
| `/runs/training-config` | GET | `training.accuracy_profile` default + profile knobs from settings.yaml (the Training tab's profile picker default); lightweight, NOT job-managed. Must stay defined above the `/{run_id}` routes |
| `/runs/{id}/report` | POST | Stage 8 — opt-in LLM insight report for one trained model (body `{model}`). **Deliberately NOT job-managed**: network-bound, and the single-slot job manager PREEMPTS — job-managing it would kill an in-flight training. 400 = no training / bad model; **502** = LLM failure (detail shown verbatim in the UI, no fallback report) |
| `/runs/{id}/report` | GET | Re-read the persisted `results_report.json` (also folded into `/summary` as `_stage8`) |
| `/runs/{id}/forecast-file` | GET | Forecast rows as a download (`?model=&fmt=csv\|parquet&level=aggregate\|series`) — aggregate from the training report arrays, series from `series_forecasts.parquet`; StreamingResponse, NOT job-managed |

Every job-managed endpoint can return **409** (preempted by a newer job) — callers must handle this explicitly, not treat a non-200 as a generic failure.

### `runs/{run_id}/` Directory

The `/runs/{id}/summary` endpoint reads all accumulated files to reconstruct full run state for the UI (`_stage2`, `_intent`, `_stage3`, `_stage4`, `_stage5`, `_stage6`, `_stage7`, `_stage8`). `forecast_user_selections.json` is written by the `/clean` endpoint — it is the single confirmed-intent file every downstream stage reads (cleaning recipe, per-series cleaner execution, validation gate, forecast EDA). The legacy `user_selections.json` (timestamp only) is still read as a fallback for pre-refactor runs.

Files in order of creation:
```
metadata.json               # Stage 1
pre_clean_eda_full.json     # Stage 2 (full stats)
cleaning_decision_payload.json  # Stage 2 (LLM input)
forecast_intent.json        # Stage 2.5 (suggestions + confidence + LLM rationale)
forecast_user_selections.json  # confirm trigger (ts/target/agg/scope/group/exog/freq/horizon)
cleaning_recipe.json        # Stage 3 (LLM or fallback output; carries group_cols/target)
cleaning_status.json        # Stage 3 (recipe_source + recipe_error, survives reload)
cleaning_report.json        # Stage 3 (before/after metrics)
cleaned_metadata.json       # Stage 3 (lightweight snapshot)
validation_gate.json        # Stage 3.5
forecasting_eda_full.json   # Stage 4 (all stats + downsampled plot arrays)
model_selection_payload.json   # Stage 4 (compact decision JSON for Stage 5)
model_selection.json        # Stage 5 (final pick + rule trace + LLM info + training_hints)
feature_report.json         # Stage 6 (EDA-seeded feature recipe + rationale)
training_report.json        # Stage 7 (per-model metrics incl. train_seconds, params,
                            #          backtest+forecast arrays, target_insights, total_seconds)
results_report.json         # Stage 8 (opt-in LLM markdown report; deleted on every /train)
```
`model_selection.json` + `feature_report.json` + `training_report.json` + `results_report.json` (and the
`data/features/{id}_*.parquet` + `models/{id}/`) are all purged on re-clean
(`routes._purge_downstream`) and on a fresh `forecasting_eda.run()`, so `/summary` can't
resurrect a decision/training made on data that no longer exists.

`model_selection.json` is deleted by `/clean` (re-clean invalidates the decision) and by
`forecasting_eda.run()` just after writing a fresh payload (covers the freq/horizon-only
re-run) — otherwise `/summary` would resurrect a decision made on data that no longer exists.

### Cleaner Execution Order

`pipeline/cleaner.py` applies the recipe in this order, which matters when debugging:

1. Drop columns (`action = "drop"`)
2. Type fixes (`type_fix`)
3. **Sort by (series, time)** — `recipe.group_cols` (from the confirmed intent) + timestamp. Temporal strategies and per-series execution both assume time order and contiguous series; the raw parquet guarantees neither
4. Outlier handling — rows marked `"remove"` are **collected**, not dropped yet
5. Missing handling — rows marked `"drop_row"` are **collected**, not dropped yet
6. All accumulated row-drop masks applied once (prevents double-counting rows that match multiple criteria)
7. Drop duplicates
8. Sort by timestamp (final output order)

Outlier handling runs *before* missing-value handling — a column can still contain NaN when `_apply_outlier()` runs. This matters for any vectorized outlier math added later (see the rolling-quantile note below).

**Per-series execution:** when `recipe.group_cols` is non-empty, every series-boundary-sensitive operation runs WITHIN each series via groupby: `rolling_iqr`, `stl_residuals`, `interpolate`, `forward_fill`/`backward_fill`, `flag_and_fill`, `mean_fill`/`median_fill` (per-series statistic), and the `remove` strategy's IQR fences (per-series quantile transform). A forward-fill must never carry one society's last value into the next society's first row, and a small series' normal values are outliers only relative to ITS OWN distribution. Global-distribution strategies (`clip_iqr`, `winsorize`, `log_transform`) deliberately stay global. Per-series STL is gated at `_STL_MAX_GROUPS` (20) — past that many series the O(n) classical decomposition runs inside every series instead (statsmodels STL × 1,800 series would take tens of minutes).

### Outlier Strategies (Stage 3)

Three temporal-aware strategies beyond global IQR/winsorize:
- `rolling_iqr` — rolling window IQR (window = frequency period). Used when `temporal_pct << iqr_pct` in the decision payload, meaning global outlier rate is inflated by seasonal patterns.
- `stl_residuals` — STL decomposition; replaces outliers with `trend + seasonal` (no rows dropped). Requires ≥ 2 complete seasonal cycles. **Length-gated:** above `_STL_MAX_POINTS` (100,000 rows), `cleaner.py` swaps to `_fast_seasonal_outlier()` — an O(n) classical decomposition (rolling-mean trend + per-phase seasonal mean) instead of statsmodels' iterative-LOESS `STL`, which is minutes-slow at multi-million-row scale. Same output shape, verified equivalent behavior.
- Both fall back to `clip_iqr` when series is too short.

The `frequency` and `period` fields in `cleaning_recipe.json` drive these window sizes. `_FREQ_PERIOD = {"hourly": 24, "daily": 7, "weekly": 52, "monthly": 12, "quarterly": 4, "yearly": 1}` maps detected frequency to integer period.

**Rolling-quantile performance:** both `pre_clean_eda.py::_outlier_counts()` (temporal outlier detection) and `cleaner.py`'s `rolling_iqr` strategy compute centered rolling Q1/Q3. Above a certain column count this is the dominant cost of the whole pipeline (pandas' `rolling().quantile()` doesn't scale well per-column). Both modules have a `_fast_rolling_q1_q3()` helper (`np.lib.stride_tricks.sliding_window_view` + `np.percentile`) used when `window <= _FAST_ROLLING_MAX_WINDOW` (40) — faster than pandas for every real `_FREQ_PERIOD` value except `"weekly"` (52), verified bit-identical to the pandas method in the no-NaN case. **`cleaner.py`'s version is additionally gated on `not s.isna().any()`** — pandas skips NaN within a window by default, plain `np.percentile` does not, and (per the execution-order note above) the series here isn't guaranteed NaN-free. `pre_clean_eda.py`'s version doesn't need that gate because its input is always `.dropna()`'d first. Also watch for redundant per-column work in general here — a prior bug re-parsed the timestamp column with `pd.to_datetime()` inside the per-numeric-column loop, turning a few seconds into minutes once more columns correctly entered the outlier-detection path.

### Validation Gate (Stage 3.5)

Every check has a `severity`: **blocking** checks decide `passed` (they catch data destruction); **warning** checks surface forecasting risks without failing the run (shown with a yellow badge in the UI). Blocking: row_loss (%, configurable), series_length (absolute 30 — statistical floor — measured as **distinct periods at the forecast frequency** when a timestamp exists: a 4.5M-row event log spanning 90 days is a 90-point series), no_null_regression (columns with a `type_fix` are exempt — coercing junk strings to NaN is a repair, not a regression), numeric_variance (relative: only fails if cleaning *destroyed* variance that existed before), timestamp_nulls, forecastable_columns (≥ 1 numeric column with variance — **skipped for count-type intents**, where an event log with zero numeric measures is perfectly forecastable), **target_survived** (the confirmed target must exist post-cleaning; measure targets need variance, count targets need non-null IDs), **group_cols_survived** (per-series scope), timestamp_monotonic (parses to datetime before checking — string-sorted dates pass a naive string comparison while being chronologically wrong). Warnings: future_timestamps, seasonal_history, per_series_history (panel median series ≥ 2 cycles).

**`target_level_preserved` (blocking, 2026-07-09, sum/mean measure targets):** the cleaned target's TOTAL may drift at most `max_target_sum_drift_pct` (default 5%) beyond what row loss explains — catches recipes that clip/winsorize the target and silently shrink every aggregate. Inputs come from `cleaning_report.json`'s `target_total_before/after` (tracked by `cleaner.py::_target_total`; also shown on the cleaning results card with a red highlight when large). Related cleaner fix: outlier strategies coerce nullable-Int64 columns to float64 first (`.clip()` with fractional bounds raises on pandas `Int64`).

Thresholds in `config/settings.yaml` under `validation_gate`:
- `max_row_loss_pct: 15` — fail if cleaning drops > 15% of rows
- `min_series_length: 30` — fail if fewer than 30 rows remain
- `max_target_sum_drift_pct: 5` — fail if the target total drifts beyond row loss + this

### Recipe sanitizer (`cleaning_agent.py::_sanitize_recipe`)

Deterministic safety net applied to BOTH LLM and fallback recipes, after the user timestamp override. It exists because prompt guidance is not a guarantee — every rule here reverts an observed real failure: ≥95%-null columns → drop; timestamp col → always `drop_row` + never droppable; inferred-vs-stored dtype mismatches → forced `cast_numeric`/`parse_datetime` (dtype_issues alone only covers categorical-inferred columns — without the forced cast the cleaned parquet ships string "numerics" and zero forecast targets); non-numeric fills → `forward_fill`; IQR-family outlier strategies (incl. stl_residuals/remove) on columns with IQR = 0 or ≥50% zeros → `keep` (quartiles collapse and the column gets destroyed — a 94%-zeros column once failed the variance gate this way); unjustified column drops → reverted (an LLM run once dropped the entire `total_*` measure family — the forecast targets; drops must be provably no-signal: constant, ~all-null, per-row ID, or audit-named).

**Intent-aware accuracy rules** (only when `forecast_user_selections.json` exists): target / exogenous / series-key / timestamp columns are **never droppable** (even when a generic rule would drop them — the gate fails loudly instead of data silently vanishing); a count-type target (`agg: nunique`) gets NO fills and NO outlier treatment (filling fabricates events, clipping an identifier is meaningless — nunique simply skips NaN); row-dropping (`drop_row`/`remove`) is only allowed on the timestamp and target columns — a null in an irrelevant column (e.g. `Member Contact`) must never delete a row that carries target signal (the vet event-log failure mode). **Level-mutating strategies never touch intent columns (2026-07-09):** `clip_iqr`/`winsorize` on the target/exog → `keep` (an observed real failure: clip_iqr on a confirmed sum-target compressed the top tail and cut the cleaned aggregate totals far below raw — every forecast inherited the bias), `remove` on a sum/mean target → `keep`, and `log_transform` → `keep` on EVERY column (it permanently rewrites stored units; modeling transforms non-destructively at Stage 4/6). Level-neutral repairs (`rolling_iqr`/`stl_residuals`) stay allowed on the target. The recipe also carries `group_cols`/`target_col`/`target_agg` so the cleaner and gate stay recipe-driven.

### LLM recipe inputs

`cleaning_decision_payload.json` includes `column_profile` (per column: dtype, null_pct, distinct_pct, zero_pct, skew) and `n_rows` — statistics only, never raw values (~900 tokens on a 19-col dataset). This is what lets the model choose strategies from evidence (zero-inflation → no IQR clipping, skew → median over mean fill, distinct_pct ≈ 100 → ID column). LLM `temperature` is configurable in `settings.yaml` (default 0.2 — provider default 1.0 produced wildly inconsistent recipes; the sanitizer guarantees safety regardless).

### Forecast Intent (Stage 2.5)

`pipeline/forecast_intent.py::detect()` runs on Stage 1/2 evidence + the **raw** parquet (cleaning hasn't happened yet) and suggests every intent field with `confidence: high|medium|low` — low renders a yellow "confirm" badge in the UI, it never blocks. It also records top-level `data_end` (max parsed timestamp of the SUGGESTED ts column) — this powers the "…or forecast until \<date\>" pickers next to both horizon inputs (`ui.horizon_datepicker` + the `sync_intent_horizon`/`sync_fcst_horizon` callbacks, which convert a picked end date to the canonical integer horizon via `pd.Period` subtraction; the Forecast EDA picker prefers Stage 4's `aggregate.end`). The integer horizon stays authoritative end-to-end — a stale `data_end` can only make the note approximate, never corrupt a run. Key detection logic:
- **Timestamps** ranked business-vs-audit (`created/loaded/inserted…` names lose); multiple business timestamps (vet data has 3 event times) → low confidence, the user picks.
- **Targets** = numeric measures (name-heuristics + variance) **plus count candidates**: high-distinct columns with event-identity names (visit/order/ticket/receipt…), `agg_hint: nunique` — "number of visits per day" is `nunique(VISIT ID)` resampled daily. **Continuous-signal scoring (2026-07-09):** keyword score ties are broken by graded, unit-free bonuses — log-cardinality (up to +1.0), CV = |std/mean| (up to +0.5), float-like values (+0.25), hard-capped at +1.75 so an unhinted column can never cross the measure-keyword bar of 2 — plus near-constant penalties (CV < 0.05 → −1.5, < 0.15 → −0.75) and a junk-name penalty (−3). This fixed the observed failure where `total_farmers` (distinct 0.04%, near-constant) beat `total_litre`/`total_kilo` purely by schema column order. `event_log_mode` is now gated on **keyword presence** (a measure-named column with score > 0), deliberately independent of these bonuses/penalties so their tuning can never flip a measure dataset into count mode. Candidates carry `cv`; the dropdown cap is `forecast_intent.max_target_candidates` (30) so real measures can't fall off.
- **Event-log mode** (no convincing measure + a count candidate): group candidates switch from `(key, ts)`-uniqueness scoring (meaningless when every row is one event) to name-ranked categorical *dimensions* (facility/region/category), and hourly event timestamps suggest `daily` forecast frequency.
- **Panel mode**: group-key scoring by `(key, ts)` uniqueness, trying **pairs** of the best singles when no single reaches 0.995 (Favorita needs `store_nbr × family`); measure/rate-named columns are excluded from keys.

`agents/forecast_intent_agent.py::refine()` is the LLM pass (mirrors the cleaning agent: pydantic enums, statistics + column names only, ~1.5KB payload). Its sanitizer is **field-level**: an invalid field reverts to the rule suggestion for that field only, never rejecting the whole response. LLM confidence is dampened to at most one level above the rule confidence (small models are overconfident on genuine ties). Failure → rule suggestions stand, error recorded in `forecast_intent.json → llm` for the UI banner.

### Forecast EDA (Stage 4)

`pipeline/forecasting_eda.py::run()` — runs on **cleaned data only**, all statistics computed in Python (no LLM in computation; Stage 5 will only ever see `model_selection_payload.json`). Reads the confirmed intent (`_resolve_selections`: selections file → intent suggestions → cleaning recipe for ts/frequency, so pre-refactor runs still resolve). ~40s aggregate / ~140s per-series on 3M rows. Computes: STL strengths per candidate period (primary period = **max seasonal strength**, not the frequency default — the milk data is annual-365, not weekly-7), FFT peaks, ACF/PACF + significant lags, ADF/KPSS + `ndiffs`/`nsdiffs` (ADF/KPSS/PP consensus via pmdarima), spectral/sample/approx entropy + DFA (antropy), ADI/CV² intermittency quadrant, heteroskedasticity → log-transform recommendation, structural breaks (ruptures), Ljung-Box on STL residuals, exog cross-correlation (positive lags = usable lead) and VIF.

Gotchas encoded in the module:
- **Empty grid bins**: `resample().sum(min_count=1)` so empty bins are NaN, then filled with **0 when zero-inflated sum-aggregated demand** (absent period = zero demand — required for honest ADI) vs time-interpolation otherwise. For **count aggs (`nunique`/`count`) empty bins are TRUE zeros** (no events), never missing. `fill_report` records which.
- A non-numeric target with `sum`/`mean` is force-switched to `nunique` (recorded as `agg_forced`).
- Per-series scope computes a **vectorized panel summary** (one groupby: per-series length/ADI/CV²/class mix; computed on per-period counts for count targets) + deep-dive stats on top-K series by volume — this feeds the global-ML vs per-series-classical routing decision in Stage 5. The aggregate series is *always* analyzed too.
- All O(n²)/slow stats are capped via `config/settings.yaml → forecast_eda` (`entropy_max_points`, `stat_test_max_points`, `stl_max_points` — above which classical `seasonal_decompose` replaces STL).
- Timestamp column is defensively re-parsed (`pd.to_datetime`) — stale cleaned parquets can carry string dates.
- Plot arrays in `forecasting_eda_full.json` are downsampled to `plot_max_points` (JSON stays ~150–500KB; it travels through the store).

Frontend split: `frontend/intent_view.py` renders the Pipeline Setup tab's intent form (confidence badges, LLM rationale banner, always-visible series-key picker — group cols drive per-series *cleaning* even in aggregate scope); `frontend/fcst_eda_view.py` renders the results-only Forecast EDA tab (confirmed-setup summary + a **scope/freq/horizon re-run** that skips cleaning + verdict tiles + Plotly charts); `frontend/model_select_view.py` renders the Model Select tab (decision hero + provenance banner + training hints + eligibility table + rule trace + Stage-5-only re-run with its own LLM switch). Callbacks: `confirm_and_run` (the one chain: `/clean` → `/validate` → `/forecast-eda` → `/model-select`, drops stale `_stage4`/`_stage5`, lands on the Forecast EDA tab on success; a model-select 409/failure keeps the EDA results and surfaces a visible warning), `run_intent_redetect`, `rerun_fcst_eda` (sends `scope`/`forecast_frequency`/`horizon` to `/forecast-eda` — no re-clean — then re-chains `/model-select` and drops stale `_stage5`/`_stage6`/`_stage7`; its `switch-use-llm` State can be `None` for pre-refactor runs — treated as `True`), `run_model_select`; `trigger_run`/`run_eda_retry` auto-chain `/forecast-intent` after Stage 2. `frontend/training_view.py` renders the **Training tab** (Stage 6+7): a multi-select of eligible models (default = top-2 of the Stage 5 ranking, cap 5; excluded models listed with reasons under the picker; seasonal-naive baseline always trained as the MASE *reference* — it can rank in the table but never wins the hero card), a live cost-estimate banner (`build_estimate_banner`, a client-side mirror of `trainer.estimate_cost` incl. nhits = 3 / chronos = 0.5 units) with a heavy-case confirm switch, then a MASE leaderboard + a **model-switchable** history/backtest/forecast chart (`dropdown-train-chart-model` → `update_train_chart` re-renders `forecast_figure` from the store — every trained model is viewable, not just the best) with a **series picker** on per-series runs (`dropdown-train-chart-series` → `GET /series-forecast` charts one series, e.g. a single taluka) + collapsible feature-recipe/params. Callbacks: `render_training_body`, `update_train_estimate` (banner follows the model multiselect), `run_training` (`/train` → store `_stage7`+`_stage6`, lands on the Training tab; a 400 with a dict detail is the heavy-cost confirm prompt). The one `switch-use-llm` on Pipeline Setup gates the LLM for BOTH the cleaning recipe and model selection (training uses no LLM). Chart colors are dataviz-validated against the `#161b2e` card surface: series `#6366f1`, trend `#059669`, seasonal `#d97706`, residual `#e66767`.

**2026-07-10 additions:** `frontend/results_view.py` renders the **Results tab** (Stage 8): trained-model picker (best pre-selected) + opt-in "Generate report" (POST `/report`; a 502 shows the LLM error verbatim — no fallback) + `dcc.Markdown` rendering of `_stage8.markdown` (`.results-report-md` CSS) + CSV/Parquet download buttons (server-side `requests` fetch → `dcc.send_bytes` through the root-level always-mounted `dcc.Download(id="download-forecast")` in layout.py). `frontend/ui.py` holds the shared primitives: `collapse_section` (the one `html.Details` expander every verbose table now uses — Data-tab schema/numeric-stats/missing/outliers, Forecast EDA seasonality + 16-test detail, Model Select eligibility + rule trace) and `horizon_datepicker`. The Forecast EDA setup recap is a one-liner; `_verdict_tiles` keeps only the 4 decision-driving tiles (the other 4 moved into the detail table). Training tab additions: accuracy-profile dropdown (`dropdown-train-profile`), leaderboard Time column, and a `train-insights-strip` (4 target-level tiles from `target_insights`) that `update_train_chart` re-renders when the chart model changes. All charts share `_GRAPH_CONFIG` (`displayModeBar: "hover"`, `scrollZoom`, double-click reset) + dark-themed vertical modebar in `_style_fig`; the training chart adds an x-axis rangeslider. Every new callback keeps the house rules: `n_clicks`/`not date`/`ctx.triggered_id` mount-fire guards, string-id-only `running=` targets.

### Model Selection (Stage 5)

Two-layer, mirroring the intent agent: `pipeline/rule_engine.py` ALWAYS decides first (pure `decide(payload, catalog, cfg)` + thin `run(run_id)` wrapper — writes nothing), then `agents/model_selector_agent.py::run(run_id, use_llm)` optionally lets the LLM override and persists `model_selection.json`. The LLM sees statistics + eligible model cards + the rule pick with its trace (~2KB, never raw data); its answer passes a **field-level sanitizer** (model ∉ eligible → revert to rule pick; valid model in wrong category → fix category from the catalog) and confidence dampening (capped at rule confidence + 1 level; a reverted model → low). Top-level `source` names who actually picked the final model (`"llm"` only when the LLM's model survived the sanitizer) — deliberate deviation from the intent agent's `suggested_by`, because the UI hero card must attribute the decision honestly.

The catalog is `config/model_categories.yaml` — **11 models (2026-07-09)**: seasonal_naive (baseline reference), auto_theta, auto_ets, auto_arima (pmdarima), prophet, croston/tsb (intermittent niche), lightgbm, xgboost, nhits (`library: torch` — pure-torch backend, see the statsforecast warning below), and **chronos** (`import_check: chronos` — Amazon's pretrained zero-shot transformer via `chronos-forecasting`, `models_lib/chronos_model.py`; variant in `settings.yaml → training.chronos_variant`, default chronos-bolt-small). mstl stays out. `available` is derived at runtime via `importlib.util.find_spec`. Rule thresholds live in `config/settings.yaml → model_selection`.

Rule ladder (first match wins, R1–R9; every rule leaves a trace entry — ML-first): R1 too-short → seasonal_naive; R2 intermittent/lumpy → croston/tsb; R3 many-series panel → global lightgbm, runner-up nhits (medium confidence when `panel.length_unit == "rows"` — per-series cycle depth is unverifiable for sum/mean aggs); R4 strong **leading** exog (+ ≥100 points) → lightgbm, runner-up xgboost; R5 multiple seasonal periods → lightgbm (Fourier/calendar features per period); R6 strong seasonality + ≥3 cycles → lightgbm (the backend deseasonalizes via seasonal indices), runner-up auto_theta; R7 trend-dominant → auto_theta, runner-up lightgbm; R8 near-noise → seasonal_naive + warning; R9 default → lightgbm (low). `_FALLBACK_ORDER = ["lightgbm", "auto_theta", "seasonal_naive"]`.

**Ranked top-5 (2026-07-09):** `decide()` additionally returns `ranking` — every eligible model scored by `_suitability_scores()` (same cfg thresholds/derived signals as the ladder, zero dataset-specific constants; each score term carries a human-readable reason), ordered by score with `_PREF_ORDER` tie-break, and the **ladder pick + runner-up hard-promoted to ranks 1–2** (the ladder stays authoritative; scores order ranks 3+). The agent's LLM pass now returns a re-ranked top-5 (`ModelRanking` schema, one-sentence reason per entry); the **list-level sanitizer** drops ineligible/duplicate entries, backfills from the rule ranking, and forces the rule pick back to rank 1 whenever the LLM's #1 didn't survive (`overrides: ["rank1"]` → confidence low, `source: "rule_engine"`). `model_selection.json` gains top-level `ranking` (`rank/model/category/label/reason/source/rule_score`); `model`/`runner_up` = ranks 1/2 for backward compat; `rule.ranking` keeps the full deterministic list. Consumers: `model_select_view._ranking_card` renders it; `training_view._default_models` pre-selects the top 2; old runs without `ranking` degrade gracefully everywhere.

Encoded domain guards: **eligibility filter** vetoes croston/tsb on smooth/erratic demand with `zero_pct < 20` (makes an LLM intermittent-pick on smooth data structurally impossible); **global-capable models (ML/deep) evaluate `min_length` against the panel's TOTAL observations** (`n_series × median length`) on per-series scope — nhits (min 200) is eligible for a 1,800-series monthly panel but stays excluded on a 60-point aggregate; exog is "usable" only when it **leads** (`best_lag ≥ 1`, |corr| ≥ 0.4) — lag-0 correlation needs future values; a lead within ±2 of the seasonal period is flagged `seasonal_echo` (milk's lag-13 corr on period-12 data); `vif_max > 10` → collinearity warning. `training_hints` (transform / exog_usable / seasonal_periods / policy aggregate-vs-per_series_local-vs-global / metric MASE) is always computed for Stages 6–7 — `transform` copies Stage 4's recommendation, never invents one.

Reference behavior: milk `run_20260630_144854` → R6 lightgbm high (runner-up auto_theta; ranking then chronos/auto_arima/auto_ets); vet `run_20260703_120826` → R7 auto_theta medium (runner-up lightgbm; then auto_arima/chronos/auto_ets).

### ⚠ statsforecast is UNUSABLE in this environment

statsforecast 2.0.3 is installed and Stage 5's catalog lists its models, but its compiled kernel `coreforecast 0.0.17` **segfaults (access violation 0xC0000005) under this venv's numpy 2.4.6** — and 0.0.17 is already the latest, so there's no upgrade. Stage 5 never triggered it because it only does an `importlib.util.find_spec` *presence* check. **Never `import statsforecast` in a code path that executes** — it crashes the whole job subprocess. Consequently Stage 7's classical models are built on **statsmodels + pmdarima** (the libraries Stage 4 already uses successfully here), and the intermittent ones are hand-rolled. The catalog `library` field still says `statsforecast` for those ids (harmless — it's only a `find_spec` key + a UI label); `models_lib/registry.py` is the real dispatch and routes them to statsmodels/pmdarima.

**The same applies to `neuralforecast`**: it depends on coreforecast (re-verified segfaulting 2026-07-09), so NHITS is a **hand-rolled pure-torch implementation** (`models_lib/nhits_model.py`; torch 2.5.1+cpu is installed and works). Never install/import neuralforecast expecting it to run here.

### Feature Engineering (Stage 6)

`pipeline/feature_builder.py::run(run_id, build_ml_matrix)` — deterministic, EDA-seeded, **no LLM**. Reuses `forecasting_eda._resolve_selections` / `_build_series` / `_pandas_freq` so the training series is *identical* to what Stages 4/5 analyzed. Two products:
- **Modeling frame** (`data/features/{id}_model_frame.parquet`, long `unique_id, ds, y[, exog]`) — the universal training input every model reads. Aggregate scope = one series via `_build_series`; per-series = `_build_panel_frame` (one vectorized `groupby([*group_cols, pd.Grouper(freq)])`, same empty-bin rules as Stage 4: count/nunique→0, zero-inflated sum→0, else per-series interpolate).
- **Feature recipe** (in `feature_report.json`) seeded straight off `forecasting_eda_full.json`: lags = significant ACF lags (∪ {1, period}, capped `max_lags`), rolling mean/std over the seasonal period, Fourier for `seasonal_periods`, momentum diffs when ndiffs/nsdiffs ≥ 1, exog **leading**-lags (only cols the user actually selected as exog), `log1p` transform when Stage 4 flagged heteroskedastic+skew. Only the ML models consume the materialized matrix (`_features.parquet`, built only when `build_ml_matrix`); the classical models train on raw `y`.

`build_supervised(frame, recipe)` is **shared** with `models_lib/ml_models.py` so training and forecasting use the SAME leakage-free features (every feature is a function of PAST values only, so the last origin row — whose future targets are unknown — is still fully featurizable; the direct multi-horizon backend predicts all h steps from that one origin row). **Gotcha:** JSON round-trips the `fourier` dict's int keys to strings (`{12:3}`→`{"12":3}`) — `add_fourier_terms` re-casts `int(period)` before the divide, or a numeric-vs-string `ufunc 'divide'` TypeError surfaces only when the recipe is reloaded from disk (not in-memory).

### Model Training (Stage 7)

`pipeline/trainer.py::run(run_id, models, confirm_heavy)` — hardcoded, **no LLM**. The Stage 5 pick is a *default*; the user trains up to **5** **eligible** models (`training.max_models_per_run`). **The user's selection is the ultimate choice (2026-07):** `best_model` = best MASE **among the requested models only** — the always-trained `seasonal_naive` baseline is the MASE reference row in the leaderboard but can never be crowned/serialized unless explicitly selected (verified: vet with only xgboost requested crowns xgboost despite the baseline's better MASE). ONE walk-forward CV harness (`_walk_forward` / `_run_per_series`) backtests every model family through a uniform `models_lib/base_model.Forecaster` interface (`fit(hist, period)` / `predict(h, future_ds, future_exog)`); the log transform round-trip lives in the base class. Metrics (`pipeline/evaluator.py`: MAE/RMSE/MAPE/sMAPE/**MASE**/R², zero-safe) rank the leaderboard by MASE. Prediction intervals: **native quantile forecasts** from the ML/nhits/chronos backends when available (`predict_quantiles`, single-series display only — quantiles don't sum across a panel), **merged with conformal for any level a backend can't produce** (chronos-bolt's trained quantile range stops at 0.9, so its 80% band is native and 95% is conformal — `forecast.interval_method` records `quantile` / `quantile+conformal` / `conformal`). Final = refit on all history → horizon forecast; best model pickled to `models/{id}/best_model.pkl` (+ recipe as `preprocessor.pkl`). **Per-series runs additionally persist every model's per-series backtest+forecast rows to `models/{id}/series_forecasts.parquet`** (long: model/unique_id/ds/kind/actual/predicted; report gains `series_ids` + `has_series_forecasts`) so the UI can chart any single series (e.g. one taluka) via `GET /runs/{id}/series-forecast?model=&series=` — lightweight, not job-managed.

**Direct multi-horizon ML backend** (`ml_models.py`, rewritten 2026-07 from the manual milk pipeline's strategies — the recursive rollout is gone): per-series **seasonal-index deseasonalization** (phase mean / overall mean, clipped 0.2–5.0 — trees can't extrapolate multiplicative seasonality; flat-index fallback for negative/zero-level series), origin features from `feature_builder.build_supervised` expanded × horizon steps (`horizon`/`target_phase`/`target_sindex` features), **quantile objectives** (P50 = the point model; interval alphas fitted lazily by `predict_quantiles`, non-crossing enforced by sorting), **early stopping on the last ~20% of origins then refit at the found tree count on ALL rows**, temp.txt-derived base params (n_estimators 800 cap, lr 0.025, leaves 63…). `run_global` adds `series_id` + per-series level/scale scalars + `share_lag1` (share of panel total at t−1) and a stratified `(series, horizon)` row cap (`training.max_train_rows`). Optuna (`tune`) searches around the base params on the same direct dataset; trials come from the **accuracy profile**.

**Accuracy profiles** (`training.accuracy_profile: fast|balanced|max` + `training.profiles`): the time-vs-accuracy budget — Optuna trials (0/30/100), early-stopping rounds, nhits epochs; global-panel Optuna only in `max` (the 1–3h-class budget). Report carries `accuracy_profile`. **Per-run override (2026-07-10):** `trainer.run(..., profile=...)` / `POST /train {"profile": ...}` — the Training tab's profile dropdown (default fetched via `GET /runs/training-config`); an unknown override silently falls back to the settings default. The report also records wall-clock: per-result `train_seconds` (success AND error paths) + top-level `total_seconds` (leaderboard Time column), and per-result `target_insights` (next-period value, horizon total vs trailing same-length history total from the EXACT frame, widest 95% band) feeding the Training tab's insights strip.

**NHITS** (`models_lib/nhits_model.py`, pure torch): stacked MLP blocks with input max-pooling (coarse→fine) + backcast residual stacking + linear-interpolated forecast knots; direct multi-horizon **quantile heads** (pinball loss) → native intervals; per-series standard scaling; Adam + early stopping on the chronologically-last windows. Single-series `NHITSForecaster` + `run_global` (one network across the panel; truncated-data model for the honest holdout backtest, fresh full-data model for the final forecast). **On per-series scope nhits ALWAYS routes global** regardless of policy (trainer's `route_global` via `registry.always_global()`). Counted 3 units in `estimate_cost` (mirrored in `training_view.estimate_cost`). NHITS works on aggregate scope too — its `min_length: 200` just excludes short aggregates (the Training picker now lists excluded models with reasons).

**Chronos** (`models_lib/chronos_model.py`): Amazon's pretrained zero-shot transformer — **no training at all**; `BaseChronosPipeline.from_pretrained` (cached per process, NEVER pickled — `_fitted` carries only the variant id) + one batched `predict_quantiles` inference call. v2.x API: `predict_quantiles(inputs, prediction_length, quantile_levels)`, list-of-tensors batching, long horizons handled natively (advisory warning suppressed). Native quantiles only within the trained range [0.1, 0.9] — `predict_quantiles` returns level 80 only and the trainer conformal-fills 95. `run_global` = batched holdout backtest (last h per series from truncated contexts, ~256-series chunks) + batched final forecast. Also **always routes global** on per-series scope; 0.5 units flat in `estimate_cost` (zero-shot). Weak on intermittent/zero-heavy demand, no exog — encoded in the suitability scoring and the agent prompt.

Backends dispatch off ids via `models_lib/registry.py`: `statistical.py` (statsmodels ETSModel/ThetaModel + pmdarima auto_arima + hand-rolled seasonal_naive; mstl implemented but not in the catalog), `intermittent.py` (hand-rolled Croston/TSB), `ml_models.py`, `nhits_model.py` (lazy torch import), `chronos_model.py` (lazy transformers import), `prophet_model.py` (prophet 1.3 installed and working). Classical models are **univariate**; exog is exploited only by the ML backends — origin-time leading-lag features PLUS **known-at-target features** (`exogtgt_*`: for a driver with lead `lag`, the value at t+h−lag is already known whenever the horizon step h ≤ lag — a per-step feature added in `_direct_dataset`/`_origin_rows`; NaN when unknown, trees handle it).

Adaptive CV (`_cv_plan`): `n_windows` sized so `n_windows·h ≤ len − min_train`, capped at `training.cv_windows` (shrinks to `cv_windows_large_panel` on big panels); too short → single holdout → else fit-only. **Cost guardrails** (`estimate_cost`, tiers fast/moderate/heavy from `policy × n_series × models`): the `/train` route computes the estimate up-front and returns a 400 with the estimate dict when `heavy` and `confirm_heavy` is false; the trainer also caps per-series-local fan-out to the top-K series by volume (`per_series_cap`) and caps simultaneous models (`max_models_per_run: 5`). Thresholds in `config/settings.yaml → feature_engineering` / `→ training`. Reference (balanced profile, direct backend): milk → **xgboost 0.795** best / lightgbm 0.899 (old recursive LGBM was 0.9365) with native quantile intervals; vet → auto_theta 0.5998 best (lightgbm 0.655). Milk 5-model run (2026-07-09, 74s): **chronos 0.861** best-of-requested / lightgbm 0.899 / baseline 0.949 / prophet 1.25 / auto_theta 1.33 / auto_arima 1.39 — the zero-shot model is a genuinely strong cheap candidate.

### Frontend — Tab-based UI (Dash 4.3.0)

The UI is a single-page Dash app with a **static tab bar** (`dcc.Tabs` in `frontend/layout.py`, `_build_tabs()`) — one tab per pipeline stage (Data & Pre-clean EDA, Pipeline Setup, Forecast EDA, Model Select, Training, Results). The Pipeline Setup tab keeps value `tab-clean` (renamed label only — the value maps to `pane-clean` and persists in browser session storage). All state lives in `dcc.Store(id="results-store", storage_type="session")` — session storage matters: the default (`memory`) is wiped on every page refresh, which previously made the UI look randomly broken.

**All tab panes stay mounted; tab switching is clientside-only.** The layout has six always-present pane divs (`pane-data` … `pane-results`) inside `tab-content`; a `clientside_callback` on `stage-tabs.value` toggles their `style.display`. Server callbacks re-render only the data-dependent bodies (`data-tab-results`, `clean-tab-body`, `past-runs-list`, the ingestion form/loaded-state toggle) and only on `results-store` changes — one round trip per store change (`render_data_pane`), zero per tab switch. Do **not** go back to rebuilding tab content on `stage-tabs.value` (the old `render_tab` pattern): constant unmount/remount of components triggered both renderer bugs below on every switch, and pinned the server for pure UI navigation. Keeping the form mounted also preserves typed DB credentials across tab switches.

**Dash fires callbacks for dynamically inserted/removed Inputs, ignoring `prevent_initial_call=True`** (unless *all* of the callback's Outputs are inside the same inserted chunk). Consequences in this codebase:
- Every button callback guards `if not n_clicks: return no_update, ...`. Before these guards, `clear_results`/`new_dataset` fired when their buttons mounted → wiped the store ("switching back to the Data tab resets the run") and their `_cancel_running()` call killed in-flight jobs (bogus 409s on Stage 2).
- Pattern-matching `ALL` input callbacks (`load_past_run`, `delete_run`) also fire when matched components are added *or removed* by a re-render, with an **empty** `changedPropIds` — keep their `ctx.triggered_id` guards.

**Never use pattern-matching ids in a `running=` spec (Dash 4.3.0 renderer bug).** When a callback with `ALL` inputs is fired by add/remove (empty `changedPropIds`), the renderer's `replacePMC` does `parsedChangedPropsIds[0][key]` on an empty array → browser error "Cannot read properties of undefined (reading 'run_id')" (the old `_PAST_RUN_RUNNING` bug — the error on switching to the Cleaning tab). String-id `running` targets are safe even if the component is unmounted mid-callback (renderer explicitly tolerates missing string ids). The past-runs double-click lock is `_RUNS_LIST_LOCK`: `running=[(Output("past-runs-list", "style"), RUNS_LIST_STYLE_LOCKED, RUNS_LIST_STYLE)]` — locks the whole always-mounted list container via `pointerEvents: none` instead of disabling each pattern-id button.

**Tabs are never locked/disabled.** Earlier versions gated the Cleaning tab on `_stage2` being present via a per-tab `disabled` prop; this was unreliable and was removed entirely. Each pane's builder shows a short, plain message when its prerequisite hasn't run yet (e.g. `_render_clean_tab()`: "Run pre-clean EDA on the Data tab first"). Prefer this pattern for any new tab — content-level guidance, not component-level locking.

**Button UX pattern — use the callback `running=` argument, not manual `dcc.Loading` + returned `disabled`:**
```python
@callback(
    Output("alert-div", "children"),
    Input("btn-x", "n_clicks"),
    running=[(Output("btn-x", "disabled"), True, False)],
    prevent_initial_call=True,
)
def do_thing(n): ...
```
Verified in this codebase: `running=` works on a plain synchronous `@callback` — no `background=True` or `background_callback_manager` needed. The frontend applies the "running" value optimistically before the request is even sent, which is what makes double-click races structurally impossible, not just less likely. Do **not** use the older pattern of returning `disabled=False` as a normal callback output — that only re-enables *after* the request, never disables *during* it. And per the renderer bug above: string-id `running` targets only, never pattern-matching ids.

**Critical Dash 4 patterns used here:**
- Global `@callback` decorator (not `@app.callback`)
- `app.py` forces `Cache-Control: no-store` on the index page (and `/_dash-layout`, `/_dash-dependencies`). Inline clientside callback functions ship *inside* the index HTML; a browser-cached stale index combined with live-fetched dependencies crashes any clientside callback added since with "Cannot read properties of undefined (reading 'apply')". Fingerprinted `_dash-component-suites` bundles stay cacheable.
- `suppress_callback_exceptions=True` — still required for the components rendered dynamically inside the pane bodies (`btn-confirm-run`, the `dropdown-intent-*` family, `btn-fcst-rerun`, `btn-model-rerun`, `switch-model-llm`, `btn-run-eda`, `btn-clear-results`)
- `allow_duplicate=True` on secondary callbacks that share an output with a primary callback
- `dcc.Loading` with `target_components={"component-id": "prop"}` to show spinners near buttons when the actual output target is elsewhere in the layout (e.g. the root-level `cleaning-status` div, or `results-store.data` for the broad loading overlay wrapping `tab-content`)

**`dcc.Dropdown` CSS:** Dash 4 replaced `.Select-*` React-Select class names with `.dash-dropdown-*`. Always use the new names. Override via CSS custom properties on `.dash-dropdown-wrapper`:
```css
.dash-dropdown-wrapper { --Dash-Fill-Inverse-Strong: #1e2235; --Dash-Text-Strong: #e2e8f0; ... }
```

**Dynamic vs persistent components:** callbacks that need to update UI regardless of which tab is currently active (e.g. `run_cleaning`) should output to a persistent root-level component (`cleaning-status`), not a component that only exists inside one tab's dynamically-rendered content — outputting to a component that Dash has since unmounted fails silently.

**Metric text color:** number displays (e.g. "Rows before/after") set `"color": "var(--bs-body-color)"` explicitly rather than relying on inherited color — Bootstrap's `.card-body` color chain through empty custom-property fallbacks can behave unexpectedly across the dark/light theme toggle.
