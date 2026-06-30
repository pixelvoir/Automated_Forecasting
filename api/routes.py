"""API routes for pipeline runs."""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator

from pipeline import ingest, pre_clean_eda

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

router = APIRouter(prefix="/runs", tags=["runs"])


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
            })
        except Exception:
            continue
    return {"runs": runs}


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
    try:
        creds = None
        if req.credentials:
            creds = {
                "host": req.credentials.host,
                "port": req.credentials.port,
                "db": req.credentials.database,
                "user": req.credentials.user,
                "password": req.credentials.password,
            }
        result = ingest.run(
            table=req.table,
            query=req.query,
            credentials=creds,
            file_path=req.file_path,
        )
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing env variable: {e}")
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return RunResponse(status="completed", **result)


@router.post("/{run_id}/pre-clean-eda")
def run_pre_clean_eda(run_id: str):
    """Stage 2: pre-cleaning EDA on the locally saved parquet. No DB connection made."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    try:
        result = pre_clean_eda.run(run_id)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    return data


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
