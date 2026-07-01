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
            "recipe": json.loads(recipe_path.read_text()),
        }
        if report_path.exists():
            stage3.update(json.loads(report_path.read_text()))
        if vg_path.exists():
            stage3["_validation"] = json.loads(vg_path.read_text())
        data["_stage3"] = stage3

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


class CleanRequest(BaseModel):
    timestamp_col: str | None = None
    use_llm: bool = True


@router.post("/{run_id}/clean")
def run_clean(run_id: str, req: CleanRequest = CleanRequest()):
    """Stage 3: call LLM cleaning agent then execute the recipe on the raw parquet.
    No DB connection. No raw data sent to LLM — only cleaning_decision_payload.json."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    selections = {"timestamp_col": req.timestamp_col}
    (run_dir / "user_selections.json").write_text(json.dumps(selections, indent=2))

    result = _run_job(tasks.clean_task, run_id, use_llm=req.use_llm, track_id=run_id)
    return {"run_id": run_id, "status": "completed", **result}


@router.post("/{run_id}/validate")
def run_validate(run_id: str):
    """Stage 3.5: run post-cleaning validation gate checks. No LLM, no DB."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return _run_job(tasks.validate_task, run_id, track_id=run_id)
