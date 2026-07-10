"""API routes for pipeline runs."""
import json
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator

from api import jobs, tasks
from pipeline import ingest  # list_tables only — light, stays in-process

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

router = APIRouter(prefix="/runs", tags=["runs"])


def _run_job(func, *args, track_id=None, **kwargs):
    """Run a heavy stage in the cancellable job slot, mapping job outcomes to HTTP errors."""
    try:
        return jobs.run_job(func, *args, track_id=track_id, **kwargs)
    except jobs.JobCancelled as e:
        raise HTTPException(status_code=409, detail=str(e))
    except jobs.JobError as e:
        raise HTTPException(status_code=500, detail=str(e))


def _purge_downstream(run_id: str):
    """Remove Stage 5/6/7 artifacts that describe data about to be replaced (re-clean or a
    fresh forecast EDA) so /summary can never resurrect a decision/training made on data
    that no longer exists."""
    run_dir = RUNS_DIR / run_id
    for name in ("model_selection.json", "feature_report.json", "training_report.json",
                 "results_report.json"):
        (run_dir / name).unlink(missing_ok=True)
    feat = ROOT / "data" / "features"
    for p in (feat / f"{run_id}_model_frame.parquet", feat / f"{run_id}_features.parquet"):
        p.unlink(missing_ok=True)
    shutil.rmtree(ROOT / "models" / run_id, ignore_errors=True)


# ── Shared credential model ─────────────────────────────────────────────────
# Credentials are accepted in the request body only.
# They are NEVER logged, stored to disk, or forwarded to any external service.
# The DB connection is opened, used, and disposed within each request.

class CredentialsPayload(BaseModel):
    host: str
    port: int = 5432
    database: str
    user: str
    password: str


# ── Table listing ───────────────────────────────────────────────────────────
# Two variants: env-var creds (GET, no body) and UI-supplied creds (POST, body).
# Both must be defined before /{run_id} routes to avoid path conflicts.

@router.get("/tables")
def get_tables():
    """List tables using credentials from environment variables (.env)."""
    print("[API] GET /runs/tables called", flush=True)
    try:
        tables = ingest.list_tables()
        print(f"[API] list_tables() returned {len(tables)} tables", flush=True)
        return {"tables": tables}
    except KeyError as e:
        print(f"[API] KeyError — missing env var: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Missing env variable: {e}")
    except Exception as e:
        print(f"[API] Exception in list_tables(): {type(e).__name__}: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tables-with-creds")
def get_tables_with_creds(req: CredentialsPayload):
    """List tables using credentials supplied in the request body.
    Credentials are used once and discarded — nothing is stored."""
    try:
        creds = {
            "host": req.host,
            "port": req.port,
            "db": req.database,
            "user": req.user,
            "password": req.password,
        }
        tables = ingest.list_tables(credentials=creds)
        return {"tables": tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run listing ─────────────────────────────────────────────────────────────

@router.get("")
def list_runs():
    """List all past runs from disk (newest first). No DB connection."""
    runs = []
    if not RUNS_DIR.exists():
        return {"runs": []}
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            runs.append({
                "run_id": run_dir.name,
                "rows": meta.get("shape", {}).get("rows", 0),
                "cols": meta.get("shape", {}).get("cols", 0),
                "source": meta.get("source", {}),
                "has_stage2": (run_dir / "cleaning_decision_payload.json").exists(),
                "has_stage3": (run_dir / "cleaning_recipe.json").exists(),
                "has_stage4": (run_dir / "model_selection_payload.json").exists(),
                "has_stage5": (run_dir / "model_selection.json").exists(),
                "has_stage6": (run_dir / "feature_report.json").exists(),
                "has_stage7": (run_dir / "training_report.json").exists(),
            })
        except Exception:
            continue
    return {"runs": runs}


# ── Cancel the running job ──────────────────────────────────────────────────
# Defined before /{run_id} routes so the literal path isn't captured as a run_id.

@router.post("/cancel")
def cancel_running():
    """Terminate whatever heavy stage is currently running (if any). Used when the user
    starts a new run or switches datasets so old work stops burning CPU immediately."""
    return {"cancelled": jobs.cancel_active()}


@router.get("/training-config")
def get_training_config():
    """Accuracy-profile default + per-profile knobs from settings.yaml, so the UI picker
    can show the configured default without the Dash process reading the API's config
    file. Light, never job-managed. (Must stay above the /{run_id} routes.)"""
    import yaml
    cfg = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text()) or {}
    tcfg = cfg.get("training", {}) or {}
    return {"accuracy_profile": tcfg.get("accuracy_profile", "balanced"),
            "profiles": tcfg.get("profiles", {}) or {}}


# ── Run management ──────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    source: str = "database"           # "database" | "file"
    credentials: CredentialsPayload | None = None  # None = fall back to env vars
    table: str | None = None
    query: str | None = None
    file_path: str | None = None

    @model_validator(mode="after")
    def validate_inputs(self):
        if self.source == "database":
            if not self.table and not self.query:
                raise ValueError("Provide 'table' or 'query' for database source.")
        elif self.source == "file":
            if not self.file_path:
                raise ValueError("Provide 'file_path' for file source.")
        return self


class RunResponse(BaseModel):
    run_id: str
    status: str
    rows: int
    cols: int
    metadata_path: str


@router.post("", response_model=RunResponse)
def create_run(req: RunRequest):
    # Starting a new run preempts any heavy stage still running for a previous dataset.
    creds = None
    if req.credentials:
        creds = {
            "host": req.credentials.host,
            "port": req.credentials.port,
            "db": req.credentials.database,
            "user": req.credentials.user,
            "password": req.credentials.password,
        }
    result = _run_job(
        tasks.ingest_task,
        table=req.table,
        query=req.query,
        credentials=creds,
        file_path=req.file_path,
    )
    return RunResponse(status="completed", **result)


@router.post("/{run_id}/pre-clean-eda")
def run_pre_clean_eda(run_id: str):
    """Stage 2: pre-cleaning EDA on the locally saved parquet. No DB connection made."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return _run_job(tasks.eda_task, run_id, track_id=run_id)


@router.get("/{run_id}/summary")
def get_run_summary(run_id: str):
    """Return metadata + Stage 2 results for a past run, read entirely from disk.
    Used to reload a run into the UI without touching the database."""
    meta_path = RUNS_DIR / run_id / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    data = json.loads(meta_path.read_text())
    dp_path = RUNS_DIR / run_id / "cleaning_decision_payload.json"
    if dp_path.exists():
        data["_stage2"] = {"decision_payload": json.loads(dp_path.read_text())}

    recipe_path = RUNS_DIR / run_id / "cleaning_recipe.json"
    report_path = RUNS_DIR / run_id / "cleaning_report.json"
    vg_path = RUNS_DIR / run_id / "validation_gate.json"
    status_path = RUNS_DIR / run_id / "cleaning_status.json"
    if recipe_path.exists():
        status = (
            json.loads(status_path.read_text()) if status_path.exists()
            else {"recipe_source": "unknown", "recipe_error": None}
        )
        stage3: dict = {
            "recipe_source": status.get("recipe_source", "unknown"),
            "recipe_error": status.get("recipe_error"),
            "llm_model": status.get("llm_model"),
            "llm_response": status.get("llm_response"),
            "recipe": json.loads(recipe_path.read_text()),
        }
        if report_path.exists():
            stage3.update(json.loads(report_path.read_text()))
        if vg_path.exists():
            stage3["_validation"] = json.loads(vg_path.read_text())
        data["_stage3"] = stage3

    # Stage 2.5 — forecast intent: suggestions (detect + LLM) and the user-confirmed
    # selections are independent files (a run can have suggestions but no confirmation).
    intent_path = RUNS_DIR / run_id / "forecast_intent.json"
    fsel_path = RUNS_DIR / run_id / "forecast_user_selections.json"
    if intent_path.exists() or fsel_path.exists():
        intent: dict = {}
        if intent_path.exists():
            intent["suggestions"] = json.loads(intent_path.read_text())
        if fsel_path.exists():
            intent["selections"] = json.loads(fsel_path.read_text())
        data["_intent"] = intent

    # Stage 4 — forecast EDA results
    feda_path = RUNS_DIR / run_id / "forecasting_eda_full.json"
    payload_path = RUNS_DIR / run_id / "model_selection_payload.json"
    if feda_path.exists() and payload_path.exists():
        data["_stage4"] = {"eda": {
            "payload": json.loads(payload_path.read_text()),
            "full": json.loads(feda_path.read_text()),
        }}

    # Stage 5 — model selection decision
    msel_path = RUNS_DIR / run_id / "model_selection.json"
    if msel_path.exists():
        data["_stage5"] = json.loads(msel_path.read_text())

    # Stage 6 — feature engineering recipe/report
    fr_path = RUNS_DIR / run_id / "feature_report.json"
    if fr_path.exists():
        data["_stage6"] = json.loads(fr_path.read_text())

    # Stage 7 — training results
    tr_path = RUNS_DIR / run_id / "training_report.json"
    if tr_path.exists():
        data["_stage7"] = json.loads(tr_path.read_text())

    # Stage 8 — persisted results report (opt-in LLM narrative)
    rr_path = RUNS_DIR / run_id / "results_report.json"
    if rr_path.exists():
        data["_stage8"] = json.loads(rr_path.read_text())

    return data


@router.delete("/{run_id}")
def delete_run(run_id: str):
    """Delete all local files for a run: run directory + raw parquet. No DB involvement."""
    if not run_id.startswith("run_") or ".." in run_id or "/" in run_id or "\\" in run_id:
        raise HTTPException(status_code=400, detail="Invalid run_id.")
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    shutil.rmtree(run_dir)
    for rel in [f"data/raw/{run_id}_raw.parquet", f"data/cleaned/{run_id}_cleaned.parquet"]:
        p = ROOT / rel
        if p.exists():
            p.unlink()
    return {"deleted": run_id, "status": "ok"}


@router.get("/{run_id}/status")
def get_status(run_id: str):
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    status = "completed" if (run_dir / "metadata.json").exists() else "in_progress"
    return {"run_id": run_id, "status": status}


@router.get("/{run_id}/metadata")
def get_metadata(run_id: str):
    meta_path = RUNS_DIR / run_id / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Metadata for run '{run_id}' not found.")
    return json.loads(meta_path.read_text())


@router.get("/{run_id}/series-forecast")
def get_series_forecast(run_id: str, model: str, series: str):
    """One series' backtest + forecast arrays for one trained model (the per-series
    viewer on the Training tab). Reads models/{id}/series_forecasts.parquet written by
    the trainer — sub-second, deliberately NOT job-managed. Intervals are per-series
    conformal from that series' own backtest residuals."""
    import numpy as np
    import pandas as pd
    from models_lib.base_model import conformal_intervals

    pq = ROOT / "models" / run_id / "series_forecasts.parquet"
    if not pq.exists():
        raise HTTPException(status_code=404,
                            detail="No per-series forecasts for this run — train first "
                                   "(per-series scope only).")
    df = pd.read_parquet(pq)
    sub = df[(df["model"] == model) & (df["unique_id"].astype(str) == series)]
    if sub.empty:
        raise HTTPException(status_code=404,
                            detail=f"No rows for model '{model}' / series '{series}'.")
    bt = sub[sub["kind"] == "backtest"].sort_values("ds")
    fc = sub[sub["kind"] == "forecast"].sort_values("ds")

    out = {
        "model": model, "series": series,
        "backtest": {"ds": [str(d) for d in bt["ds"]],
                     "actual": [float(v) if np.isfinite(v) else None for v in bt["actual"]],
                     "predicted": [float(v) if np.isfinite(v) else None for v in bt["predicted"]]},
        "forecast": {"ds": [str(d) for d in fc["ds"]],
                     "yhat": [float(v) if np.isfinite(v) else None for v in fc["predicted"]]},
    }
    if len(bt) and len(fc):
        resid = (bt["actual"] - bt["predicted"]).to_numpy(dtype=float)
        resid = resid[np.isfinite(resid)]
        point = fc["predicted"].to_numpy(dtype=float)
        if len(resid) >= 2:
            iv = conformal_intervals(point, {k + 1: resid for k in range(len(point))}, [95])
            lo, hi = iv[95]
            out["forecast"]["lo95"] = [round(float(v), 4) for v in lo]
            out["forecast"]["hi95"] = [round(float(v), 4) for v in hi]
    return out


# ── Stage 8: results report + forecast download ──────────────────────────────

class ReportRequest(BaseModel):
    """The trained model the Stage 8 narrative should cover."""
    model: str


@router.post("/{run_id}/report")
def generate_report(run_id: str, req: ReportRequest):
    """Stage 8: opt-in LLM insight report for one trained model. Sends DERIVED data only
    (statistics, metrics, forecast summaries — never raw rows) via the `llm_report`
    settings profile. LLM-only: no rule fallback — a 502 carries the failure verbatim.

    Deliberately NOT job-managed: the call is network-bound (no CPU work), and the
    single-slot job manager PREEMPTS — routing this through it would kill an in-flight
    training when the user clicks Generate. FastAPI runs sync routes in a threadpool,
    and llm_report.timeout_seconds bounds the wait."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    if not (run_dir / "training_report.json").exists():
        raise HTTPException(status_code=400,
                            detail="Train at least one model first — the report reads "
                                   "training results.")
    from agents import results_agent
    from agents.llm_client import LLMError
    try:
        report = results_agent.run(run_id, req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"Report LLM call failed: {e}")
    return {"run_id": run_id, "status": "completed", "report": report}


@router.get("/{run_id}/report")
def get_report(run_id: str):
    """Re-read the persisted Stage 8 report (also folded into /summary as _stage8)."""
    p = RUNS_DIR / run_id / "results_report.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="No report generated for this run yet.")
    return json.loads(p.read_text())


@router.get("/{run_id}/forecast-file")
def download_forecast(run_id: str, model: str, fmt: str = "csv", level: str = "aggregate"):
    """Forecast rows as a downloadable file. `aggregate` = the training_report display
    arrays (ds/yhat + interval bounds); `series` = series_forecasts.parquet filtered to
    the model's forecast rows (per-series runs only). Light, NOT job-managed."""
    import io

    import pandas as pd
    from fastapi.responses import StreamingResponse

    if not run_id.startswith("run_") or ".." in run_id or "/" in run_id or "\\" in run_id:
        raise HTTPException(status_code=400, detail="Invalid run_id.")
    if fmt not in ("csv", "parquet"):
        raise HTTPException(status_code=400, detail="fmt must be 'csv' or 'parquet'.")
    if level not in ("aggregate", "series"):
        raise HTTPException(status_code=400, detail="level must be 'aggregate' or 'series'.")

    if level == "series":
        pq = ROOT / "models" / run_id / "series_forecasts.parquet"
        if not pq.exists():
            raise HTTPException(status_code=404,
                                detail="No per-series forecasts for this run — per-series "
                                       "scope only.")
        df = pd.read_parquet(pq)
        df = df[(df["model"] == model) & (df["kind"] == "forecast")]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No forecast rows for '{model}'.")
        df = df[["unique_id", "ds", "predicted"]].rename(columns={"predicted": "yhat"})
    else:
        tr_path = RUNS_DIR / run_id / "training_report.json"
        if not tr_path.exists():
            raise HTTPException(status_code=404, detail="No training report for this run.")
        report = json.loads(tr_path.read_text())
        entry = next((e for e in report.get("results") or [] if e.get("model") == model), None)
        if entry is None or entry.get("error") or not (entry.get("forecast") or {}).get("ds"):
            raise HTTPException(status_code=404,
                                detail=f"No forecast available for model '{model}'.")
        fc = entry["forecast"]
        df = pd.DataFrame({k: v for k, v in fc.items() if isinstance(v, list)})

    buf = io.BytesIO()
    if fmt == "parquet":
        df.to_parquet(buf, index=False)
        media = "application/octet-stream"
    else:
        buf.write(df.to_csv(index=False).encode("utf-8"))
        media = "text/csv"
    buf.seek(0)
    fname = f"{run_id}_{model}_forecast_{level}.{fmt}"
    return StreamingResponse(buf, media_type=media,
                             headers={"Content-Disposition": f'attachment; filename="{fname}"'})


class CleanRequest(BaseModel):
    """The confirmed forecast intent — the single user checkpoint of the pipeline.
    Everything downstream (cleaning recipe, per-series execution, validation gate,
    forecast EDA) consumes this file; only use_llm is a transient flag."""
    timestamp_col: str | None = None
    target_col: str | None = None
    agg: str = "sum"                       # "sum" | "mean" | "nunique" (events per period)
    scope: str = "aggregate"               # "aggregate" | "per_series"
    group_cols: list[str] = []
    exog_cols: list[str] = []
    forecast_frequency: str | None = None
    horizon: int | None = None
    use_llm: bool = True


@router.post("/{run_id}/clean")
def run_clean(run_id: str, req: CleanRequest = CleanRequest()):
    """Stage 3: call LLM cleaning agent then execute the recipe on the raw parquet.
    No DB connection. No raw data sent to LLM — only cleaning_decision_payload.json
    plus the confirmed intent (names + choices, no values)."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    selections = {k: v for k, v in req.model_dump().items() if k != "use_llm"}
    (run_dir / "forecast_user_selections.json").write_text(json.dumps(selections, indent=2))
    # A re-clean invalidates every decision made on the old cleaned data (model selection,
    # features, training) — remove them so /summary can't resurrect a stale result if the
    # chain aborts early.
    _purge_downstream(run_id)

    result = _run_job(tasks.clean_task, run_id, use_llm=req.use_llm, track_id=run_id)
    return {"run_id": run_id, "status": "completed", **result}


@router.post("/{run_id}/validate")
def run_validate(run_id: str):
    """Stage 3.5: run post-cleaning validation gate checks. No LLM, no DB."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return _run_job(tasks.validate_task, run_id, track_id=run_id)


class ForecastIntentRequest(BaseModel):
    use_llm: bool = True


@router.post("/{run_id}/forecast-intent")
def run_forecast_intent(run_id: str, req: ForecastIntentRequest = ForecastIntentRequest()):
    """Stage 2.5: detect forecast-intent suggestions (timestamp/target/scope/series key/
    frequency/horizon) from Stage 1/2 evidence + the RAW parquet, then optionally refine
    with the LLM (statistics + column names only). Job-managed; auto-triggered by the
    frontend after pre-clean EDA. The user confirms everything before cleaning runs."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return _run_job(tasks.forecast_intent_task, run_id, use_llm=req.use_llm, track_id=run_id)


class ForecastEDARequest(BaseModel):
    """Optional overrides for a Stage-4-only re-run. None of these affect cleaning
    (cleaner.py never reads scope/frequency/horizon/exog — only the series key / target /
    agg do), so adjusting them must NOT force a re-clean. `scope` toggles
    overall-vs-per-series forecasting; changing the series key still requires a re-clean
    and stays on Pipeline Setup. `exog_cols` re-selects drivers from columns that
    SURVIVED cleaning (the sanitizer never drops columns silently, so most survive)."""
    forecast_frequency: str | None = None
    horizon: int | None = None
    scope: str | None = None               # "aggregate" | "per_series"
    exog_cols: list[str] | None = None     # None = keep current; [] = clear


@router.post("/{run_id}/forecast-eda")
def run_forecast_eda(run_id: str, req: ForecastEDARequest = ForecastEDARequest()):
    """Stage 4: full forecasting EDA on the confirmed intent. All statistics computed
    in Python — the LLM only ever sees model_selection_payload.json (Stage 5). Frequency,
    horizon, scope and exogenous drivers can be re-run here without re-cleaning
    (they're Stage-4+ concerns)."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    if req.forecast_frequency or req.horizon or req.scope or req.exog_cols is not None:
        sel_path = run_dir / "forecast_user_selections.json"
        sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
        # Per-series forecasting needs a series key; if none was ever confirmed, per-series
        # cleaning never ran, so this genuinely requires a re-clean on Pipeline Setup.
        if req.scope == "per_series" and not (sel.get("group_cols")):
            raise HTTPException(
                status_code=400,
                detail="Per-series forecasting needs a series key. Set one on the Pipeline "
                       "Setup tab and confirm (that choice affects cleaning, so it re-cleans).")
        if req.exog_cols is not None:
            # exog columns must exist in the CLEANED data (Stage 4 reads only that)
            cm_path = run_dir / "cleaned_metadata.json"
            cleaned_cols = set((json.loads(cm_path.read_text()).get("nulls") or {})
                               if cm_path.exists() else {})
            missing = [c for c in req.exog_cols if cleaned_cols and c not in cleaned_cols]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Exogenous column(s) not in the cleaned data: {missing}. "
                           "Columns dropped by cleaning need a re-clean on Pipeline Setup.")
            sel["exog_cols"] = [c for c in req.exog_cols if c != sel.get("target_col")]
        if req.forecast_frequency:
            sel["forecast_frequency"] = req.forecast_frequency
        if req.horizon:
            sel["horizon"] = req.horizon
        if req.scope:
            sel["scope"] = req.scope
        sel_path.write_text(json.dumps(sel, indent=2))
    return _run_job(tasks.forecast_eda_task, run_id, track_id=run_id)


class ModelSelectRequest(BaseModel):
    use_llm: bool = True


@router.post("/{run_id}/model-select")
def run_model_select(run_id: str, req: ModelSelectRequest = ModelSelectRequest()):
    """Stage 5: pick the forecasting model from the Stage 4 evidence. The rule engine
    always decides first; the LLM (when enabled) may override — statistics and model
    names only, never raw data. Job-managed like every other stage."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    if not (run_dir / "model_selection_payload.json").exists():
        raise HTTPException(
            status_code=400,
            detail="Run forecast EDA first — model selection needs its evidence payload.")
    return _run_job(tasks.model_select_task, run_id, use_llm=req.use_llm, track_id=run_id)


class TrainRequest(BaseModel):
    """The user's model choice for Stage 7. The Stage 5 pick is only the default; the user
    may train one or more of the ELIGIBLE models (capped in trainer). `confirm_heavy`
    acknowledges a 'heavy' cost estimate for a big per-series panel. `profile` optionally
    overrides training.accuracy_profile for this run (fast/balanced/max)."""
    models: list[str] = []
    confirm_heavy: bool = False
    profile: str | None = None


@router.post("/{run_id}/train")
def run_train(run_id: str, req: TrainRequest = TrainRequest()):
    """Stages 6+7: feature engineering + training of the user-selected models. The rule
    engine's pick is a default, not a mandate. Job-managed (cancellable). Returns 400 if
    model selection hasn't run, if no trainable model was chosen, or — for a big per-series
    panel — if the estimate is 'heavy' and `confirm_heavy` is false (the response carries
    the estimate so the UI can prompt for confirmation)."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    if req.profile not in (None, "fast", "balanced", "max"):
        raise HTTPException(status_code=400,
                            detail=f"Unknown accuracy profile '{req.profile}'.")
    msel_path = run_dir / "model_selection.json"
    if not msel_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Run model selection (Stage 5) first — training reads its hints and "
                   "eligibility.")

    from pipeline import trainer  # lazy — keeps API startup light

    msel = json.loads(msel_path.read_text())
    hints = msel.get("training_hints") or {}
    policy = hints.get("policy") or "aggregate"
    eligible = {m["id"]: m for m in (msel.get("eligible_models") or [])}

    def _trainable(mid):
        m = eligible.get(mid)
        return (mid in trainer.registry.KNOWN_IDS and m is not None
                and m.get("available", True) and not m.get("excluded_reason"))

    valid = [m for m in dict.fromkeys(req.models) if _trainable(m)]
    if not valid:
        sug = msel.get("model")
        if sug and sug in trainer.registry.KNOWN_IDS:
            valid = [sug]
        else:
            raise HTTPException(status_code=400,
                                detail="No trainable model selected. Pick at least one "
                                       "eligible model.")

    n_series = 1
    payload_path = run_dir / "model_selection_payload.json"
    if payload_path.exists():
        panel = (json.loads(payload_path.read_text()).get("panel") or {})
        n_series = panel.get("n_series") or 1

    estimate = trainer.estimate_cost(n_series, policy, valid, trainer._load_config())
    if estimate["tier"] == "heavy" and not req.confirm_heavy:
        raise HTTPException(status_code=400, detail={
            "message": "Training this many series/models is heavy — confirm to proceed.",
            "estimate": estimate, "needs_confirm": True})

    # A re-train invalidates any Stage 8 narrative written about the old training.
    (run_dir / "results_report.json").unlink(missing_ok=True)

    result = _run_job(tasks.train_task, run_id, valid, confirm_heavy=req.confirm_heavy,
                      profile=req.profile, track_id=run_id)
    return {"run_id": run_id, "status": "completed", **result}
