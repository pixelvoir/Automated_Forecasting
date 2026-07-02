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
```

LLM provider is configured in `config/settings.yaml` — switch between `ollama`, `openai`, `gemini`, `groq` without touching code. Ollama runs locally; cloud providers need their key in `.env`.

`prophet` is commented out of `requirements.txt` (needs C++ build tools on Windows) — see the comment there for install options.

---

## Security Constraints (non-negotiable)

- Raw data **never** sent to any LLM or external API — only computed statistics (~200 tokens)
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
| 3.5 — Validation | `pipeline/validation_gate.py` | `runs/{id}/validation_gate.json` |
| 4–8 | stubs in `pipeline/`, `agents/`, `models_lib/` | not yet implemented (`def run(): pass` / `def process(): pass`) |

Stage 3 is two-step: the LLM agent decides the recipe (`cleaning_agent.py`), then `cleaner.py` executes it. Both are called sequentially inside `api/tasks.py::clean_task`, which the `/runs/{id}/clean` endpoint runs via the job manager. The LLM only receives `cleaning_decision_payload.json`.

**Stage 2 produces two files:** `pre_clean_eda_full.json` (full stats for UI display) and `cleaning_decision_payload.json` (counts/percentages/flags plus a compact per-column profile — the one sent to the LLM). Column names appear in the payload to key the recipe, but no raw values ever.

**Frequency inference (`ingest.py::_infer_frequency`) deduplicates timestamps first** — it measures the time *grid*, not row spacing. Panel/multi-series data has many rows per timestamp (e.g. ~1,800 store×product rows per day); the median raw row-to-row diff is then 0 and daily panels were mislabeled "hourly" (which also mis-sized every seasonal window downstream, and made ETL audit columns win the most-granular-timestamp fallback pick).

**Key signal in Stage 2 payload:** `temporal_pct << iqr_pct` means global outlier rate is inflated by seasonal patterns — the LLM should pick `rolling_iqr` or `stl_residuals` instead of `clip_iqr`.

### Dtype Inference (Stage 1) — numeric-as-string columns

`pipeline/ingest.py::_infer_dtype()` checks numeric coercion *before* datetime for object-dtype columns (`pd.to_numeric(sample, errors="coerce").notna().mean() > 0.8`). This matters: CSV/DB columns are routinely read as strings even when the values are numeric, and if Stage 1 misclassifies them as `"categorical"`, Stage 2's outlier detection (`pre_clean_eda.py::_outlier_counts()`, which only loops over columns Stage 1 called `"numeric"`) silently skips them entirely — producing a cleaning recipe that says `"keep"` for every outlier strategy with no real signal behind it. `extract_metadata()`'s numeric_stats and `_outlier_counts()` both re-coerce with `pd.to_numeric(df[col], errors="coerce")` (no-op for already-numeric columns) rather than trusting the raw dtype.

### LLM Client

`agents/llm_client.py` wraps all four providers (ollama / openai / gemini / groq) through the `openai` Python package with a `base_url` swap. Provider and model come from `config/settings.yaml`. Callers must catch `LLMError` and fall back — `cleaning_agent.py` has a rule-based fallback that runs when the LLM is unavailable, or when `cleaning_agent.run(run_id, use_llm=False)` is called deliberately.

`cleaning_agent.run()` always writes `cleaning_status.json` (`{"recipe_source": "llm"|"fallback", "recipe_error": str|None}`) alongside `cleaning_recipe.json`, so the actual LLM failure reason survives a reload via `/runs/{id}/summary` instead of showing as `"unknown"`. It also force-corrects `type_fix` to `cast_numeric`/`parse_datetime` for any column Stage 2 flagged in `dtype_issues`, as a safety net in case an LLM response ignores the signal.

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

Every job-managed endpoint can return **409** (preempted by a newer job) — callers must handle this explicitly, not treat a non-200 as a generic failure.

### `runs/{run_id}/` Directory

The `/runs/{id}/summary` endpoint reads all accumulated files to reconstruct full run state for the UI. `user_selections.json` is written by the `/clean` endpoint and always overrides the LLM's `timestamp_col` choice.

Files in order of creation:
```
metadata.json               # Stage 1
pre_clean_eda_full.json     # Stage 2 (full stats)
cleaning_decision_payload.json  # Stage 2 (LLM input)
user_selections.json        # Stage 3 trigger (user's timestamp_col)
cleaning_recipe.json        # Stage 3 (LLM or fallback output)
cleaning_status.json        # Stage 3 (recipe_source + recipe_error, survives reload)
cleaning_report.json        # Stage 3 (before/after metrics)
cleaned_metadata.json       # Stage 3 (lightweight snapshot)
validation_gate.json        # Stage 3.5
```

### Cleaner Execution Order

`pipeline/cleaner.py` applies the recipe in this order, which matters when debugging:

1. Drop columns (`action = "drop"`)
2. Type fixes (`type_fix`)
3. Outlier handling — rows marked `"remove"` are **collected**, not dropped yet
4. Missing handling — rows marked `"drop_row"` are **collected**, not dropped yet
5. All accumulated row-drop masks applied once (prevents double-counting rows that match multiple criteria)
6. Drop duplicates
7. Sort by timestamp

Outlier handling runs *before* missing-value handling — a column can still contain NaN when `_apply_outlier()` runs. This matters for any vectorized outlier math added later (see the rolling-quantile note below).

### Outlier Strategies (Stage 3)

Three temporal-aware strategies beyond global IQR/winsorize:
- `rolling_iqr` — rolling window IQR (window = frequency period). Used when `temporal_pct << iqr_pct` in the decision payload, meaning global outlier rate is inflated by seasonal patterns.
- `stl_residuals` — STL decomposition; replaces outliers with `trend + seasonal` (no rows dropped). Requires ≥ 2 complete seasonal cycles. **Length-gated:** above `_STL_MAX_POINTS` (100,000 rows), `cleaner.py` swaps to `_fast_seasonal_outlier()` — an O(n) classical decomposition (rolling-mean trend + per-phase seasonal mean) instead of statsmodels' iterative-LOESS `STL`, which is minutes-slow at multi-million-row scale. Same output shape, verified equivalent behavior.
- Both fall back to `clip_iqr` when series is too short.

The `frequency` and `period` fields in `cleaning_recipe.json` drive these window sizes. `_FREQ_PERIOD = {"hourly": 24, "daily": 7, "weekly": 52, "monthly": 12, "quarterly": 4, "yearly": 1}` maps detected frequency to integer period.

**Rolling-quantile performance:** both `pre_clean_eda.py::_outlier_counts()` (temporal outlier detection) and `cleaner.py`'s `rolling_iqr` strategy compute centered rolling Q1/Q3. Above a certain column count this is the dominant cost of the whole pipeline (pandas' `rolling().quantile()` doesn't scale well per-column). Both modules have a `_fast_rolling_q1_q3()` helper (`np.lib.stride_tricks.sliding_window_view` + `np.percentile`) used when `window <= _FAST_ROLLING_MAX_WINDOW` (40) — faster than pandas for every real `_FREQ_PERIOD` value except `"weekly"` (52), verified bit-identical to the pandas method in the no-NaN case. **`cleaner.py`'s version is additionally gated on `not s.isna().any()`** — pandas skips NaN within a window by default, plain `np.percentile` does not, and (per the execution-order note above) the series here isn't guaranteed NaN-free. `pre_clean_eda.py`'s version doesn't need that gate because its input is always `.dropna()`'d first. Also watch for redundant per-column work in general here — a prior bug re-parsed the timestamp column with `pd.to_datetime()` inside the per-numeric-column loop, turning a few seconds into minutes once more columns correctly entered the outlier-detection path.

### Validation Gate (Stage 3.5)

Every check has a `severity`: **blocking** checks decide `passed` (they catch data destruction); **warning** checks surface forecasting risks without failing the run (shown with a yellow badge in the UI). Blocking: row_loss (%, configurable), series_length (absolute 30 — statistical floor, deliberately not a percentage), no_null_regression (columns with a `type_fix` are exempt — coercing junk strings to NaN is a repair, not a regression), numeric_variance (relative: only fails if cleaning *destroyed* variance that existed before — an already-constant column isn't cleaning's fault), timestamp_nulls, forecastable_columns (≥ 1 numeric column with variance must survive), timestamp_monotonic (parses to datetime before checking — string-sorted dates pass a naive string comparison while being chronologically wrong). Warnings: future_timestamps, seasonal_history (≥ 2 cycles of the recipe period).

Thresholds in `config/settings.yaml` under `validation_gate`:
- `max_row_loss_pct: 15` — fail if cleaning drops > 15% of rows
- `min_series_length: 30` — fail if fewer than 30 rows remain

### Recipe sanitizer (`cleaning_agent.py::_sanitize_recipe`)

Deterministic safety net applied to BOTH LLM and fallback recipes, after the user timestamp override. It exists because prompt guidance is not a guarantee — every rule here reverts an observed real failure: ≥95%-null columns → drop; timestamp col → always `drop_row` + never droppable; inferred-vs-stored dtype mismatches → forced `cast_numeric`/`parse_datetime` (dtype_issues alone only covers categorical-inferred columns — without the forced cast the cleaned parquet ships string "numerics" and zero forecast targets); non-numeric fills → `forward_fill`; IQR-family outlier strategies (incl. stl_residuals/remove) on columns with IQR = 0 or ≥50% zeros → `keep` (quartiles collapse and the column gets destroyed — a 94%-zeros column once failed the variance gate this way); unjustified column drops → reverted (an LLM run once dropped the entire `total_*` measure family — the forecast targets; drops must be provably no-signal: constant, ~all-null, per-row ID, or audit-named).

### LLM recipe inputs

`cleaning_decision_payload.json` includes `column_profile` (per column: dtype, null_pct, distinct_pct, zero_pct, skew) and `n_rows` — statistics only, never raw values (~900 tokens on a 19-col dataset). This is what lets the model choose strategies from evidence (zero-inflation → no IQR clipping, skew → median over mean fill, distinct_pct ≈ 100 → ID column). LLM `temperature` is configurable in `settings.yaml` (default 0.2 — provider default 1.0 produced wildly inconsistent recipes; the sanitizer guarantees safety regardless).

### Frontend — Tab-based UI (Dash 4.3.0)

The UI is a single-page Dash app with a **static tab bar** (`dcc.Tabs` in `frontend/layout.py`, `_build_tabs()`) — one tab per pipeline stage (Data & Pre-clean EDA, Cleaning, Forecast EDA, Model Select, Training, Results). All state lives in `dcc.Store(id="results-store", storage_type="session")` — session storage matters: the default (`memory`) is wiped on every page refresh, which previously made the UI look randomly broken.

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
- `suppress_callback_exceptions=True` — still required for the components rendered dynamically inside the pane bodies (`btn-run-cleaning`, `dropdown-ts-confirm`, `btn-run-eda`, `btn-clear-results`)
- `allow_duplicate=True` on secondary callbacks that share an output with a primary callback
- `dcc.Loading` with `target_components={"component-id": "prop"}` to show spinners near buttons when the actual output target is elsewhere in the layout (e.g. the root-level `cleaning-status` div, or `results-store.data` for the broad loading overlay wrapping `tab-content`)

**`dcc.Dropdown` CSS:** Dash 4 replaced `.Select-*` React-Select class names with `.dash-dropdown-*`. Always use the new names. Override via CSS custom properties on `.dash-dropdown-wrapper`:
```css
.dash-dropdown-wrapper { --Dash-Fill-Inverse-Strong: #1e2235; --Dash-Text-Strong: #e2e8f0; ... }
```

**Dynamic vs persistent components:** callbacks that need to update UI regardless of which tab is currently active (e.g. `run_cleaning`) should output to a persistent root-level component (`cleaning-status`), not a component that only exists inside one tab's dynamically-rendered content — outputting to a component that Dash has since unmounted fails silently.

**Metric text color:** number displays (e.g. "Rows before/after") set `"color": "var(--bs-body-color)"` explicitly rather than relying on inherited color — Bootstrap's `.card-body` color chain through empty custom-property fallbacks can behave unexpectedly across the dark/light theme toggle.
