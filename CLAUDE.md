# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bat
setup_venv.bat          # one-time: create .venv and install deps
run_dev.bat             # terminal 1: FastAPI on :8000 (auto-reload)
run_frontend.bat        # terminal 2: Dash UI on :8050
```

Health check: `curl http://localhost:8000/health`

No test suite, no linter config. To run pipeline stages manually:
```python
from pipeline import ingest, pre_clean_eda, cleaner, validation_gate
from agents import cleaning_agent

result = ingest.run(file_path="path/to/data.csv")   # or table/query from DB
run_id = result["run_id"]
pre_clean_eda.run(run_id)
cleaning_agent.run(run_id)   # LLM or rule-based fallback
cleaner.run(run_id)
validation_gate.run(run_id)
```

LLM provider is configured in `config/settings.yaml` — switch between `ollama`, `openai`, `gemini`, `groq` without touching code. Ollama runs locally; cloud providers need their key in `.env`.

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
  │                   └─ HTTP → FastAPI (:8000) → pipeline module
  │                                                    │
  │                                                    ├─ reads/writes runs/{run_id}/*.json
  │                                                    └─ reads/writes data/raw | data/cleaned
  │
  └─ results-store (dcc.Store) drives all UI state
       └─ render_results() rebuilds right panel on every store change
```

### Pipeline Stages

Each stage is a standalone Python module in `pipeline/` with a single `run(run_id)` entry point. Stages chain by writing JSON files that the next stage reads — no shared state in memory between requests.

| Stage | Module | Writes |
|---|---|---|
| 1 — Ingestion | `pipeline/ingest.py` | `runs/{id}/metadata.json`, `data/raw/{id}_raw.parquet` |
| 2 — Pre-clean EDA | `pipeline/pre_clean_eda.py` | `runs/{id}/pre_clean_eda_full.json`, `runs/{id}/cleaning_decision_payload.json` |
| 3 — Cleaning | `agents/cleaning_agent.py` → `pipeline/cleaner.py` | `cleaning_recipe.json`, `data/cleaned/{id}_cleaned.parquet`, `cleaning_report.json`, `cleaned_metadata.json` |
| 3.5 — Validation | `pipeline/validation_gate.py` | `runs/{id}/validation_gate.json` |
| 4–8 | stubs in `pipeline/` and `agents/` | not yet implemented |

Stage 3 is two-step: the LLM agent decides the recipe (`cleaning_agent.py`), then `cleaner.py` executes it. Both are called sequentially by the `/runs/{id}/clean` API endpoint. The LLM only receives `cleaning_decision_payload.json`.

**Stage 2 produces two files:** `pre_clean_eda_full.json` (full stats for UI display) and `cleaning_decision_payload.json` (minimal counts/flags only, ~200 tokens — the one sent to the LLM). Column names appear in the payload to key the recipe, but no raw values.

**Key signal in Stage 2 payload:** `temporal_pct << iqr_pct` means global outlier rate is inflated by seasonal patterns — the LLM should pick `rolling_iqr` or `stl_residuals` instead of `clip_iqr`.

### LLM Client

`agents/llm_client.py` wraps all four providers (ollama / openai / gemini / groq) through the `openai` Python package with a `base_url` swap. Provider and model come from `config/settings.yaml`. Callers must catch `LLMError` and fall back — `cleaning_agent.py` has a rule-based fallback that runs when the LLM is unavailable.

### API Routes

All routes are under `/runs` (defined in `api/routes.py`):

| Endpoint | Method | Purpose |
|---|---|---|
| `/runs` | GET | List all past runs (newest first) |
| `/runs` | POST | Stage 1 — ingest data (table / query / file) |
| `/runs/tables` | GET | List DB tables (env var creds) |
| `/runs/tables-with-creds` | POST | List DB tables (request body creds, never stored) |
| `/runs/{id}` | DELETE | Delete run directory + parquets |
| `/runs/{id}/status` | GET | Check Stage 1 complete |
| `/runs/{id}/metadata` | GET | Stage 1 metadata |
| `/runs/{id}/summary` | GET | All accumulated run data (used to reconstruct UI state) |
| `/runs/{id}/pre-clean-eda` | POST | Stage 2 — pre-clean EDA |
| `/runs/{id}/clean` | POST | Stage 3 — cleaning agent + cleaner + validation |
| `/runs/{id}/validate` | POST | Stage 3.5 — validation gate only |

### `runs/{run_id}/` Directory

The `/runs/{id}/summary` endpoint reads all accumulated files to reconstruct full run state for the UI. `user_selections.json` is written by the `/clean` endpoint and always overrides the LLM's `timestamp_col` choice.

Files in order of creation:
```
metadata.json               # Stage 1
pre_clean_eda_full.json     # Stage 2 (full stats)
cleaning_decision_payload.json  # Stage 2 (LLM input)
user_selections.json        # Stage 3 trigger (user's timestamp_col)
cleaning_recipe.json        # Stage 3 (LLM or fallback output)
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

### Outlier Strategies (Stage 3)

Three temporal-aware strategies beyond global IQR/winsorize:
- `rolling_iqr` — rolling window IQR (window = frequency period). Used when `temporal_pct << iqr_pct` in the decision payload, meaning global outlier rate is inflated by seasonal patterns.
- `stl_residuals` — STL decomposition; replaces outliers with `trend + seasonal` (no rows dropped). Requires ≥ 2 complete seasonal cycles.
- Both fall back to `clip_iqr` when series is too short.

The `frequency` and `period` fields in `cleaning_recipe.json` drive these window sizes. `_FREQ_PERIOD = {"hourly": 24, "daily": 7, "weekly": 52, "monthly": 12, "quarterly": 4, "yearly": 1}` maps detected frequency to integer period.

### Validation Gate Thresholds

Configurable in `config/settings.yaml` under `validation`:
- `max_row_loss_pct: 15` — fail if cleaning drops > 15% of rows
- `min_series_length: 30` — fail if fewer than 30 rows remain

### Frontend — Dash 4.3.0 Specifics

The UI is a single-page Dash app. All state lives in `dcc.Store(id="results-store")`. Updating the store triggers `render_results()` in `callbacks.py`, which rebuilds the entire right panel.

**Critical Dash 4 patterns used here:**
- Global `@callback` decorator (not `@app.callback`)
- `suppress_callback_exceptions=True` — required because `btn-run-cleaning`, `dropdown-ts-confirm` are rendered dynamically inside `results-panel`
- `allow_duplicate=True` on secondary callbacks that share an output with a primary callback
- `dcc.Loading` with `target_components={"component-id": "prop"}` to show spinners near buttons when the actual output target is elsewhere in the layout

**`dcc.Dropdown` CSS:** Dash 4 replaced `.Select-*` React-Select class names with `.dash-dropdown-*`. Always use the new names. Override via CSS custom properties on `.dash-dropdown-wrapper`:
```css
.dash-dropdown-wrapper { --Dash-Fill-Inverse-Strong: #1e2235; --Dash-Text-Strong: #e2e8f0; ... }
```

**Dynamic vs persistent components:** `btn-run-cleaning` and `dropdown-ts-confirm` are rendered dynamically (only when stage 2 is done and stage 3 is not). Callbacks that output to these components can fail silently in Dash 4 when the component is removed by a concurrent re-render. The fix used here: output to `cleaning-status` (a persistent `html.Div` in the permanent layout) and use `target_components` on any `dcc.Loading` wrappers near dynamic buttons.

**Button UX pattern:** Every action button must have a paired `dcc.Loading` and a `disabled` output so the button disables during the request and a spinner appears near it.
