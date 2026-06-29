"""API routes for pipeline runs."""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator

from pipeline import ingest

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

router = APIRouter(prefix="/runs", tags=["runs"])


# ── Table listing (must be defined before /{run_id} routes to avoid path conflict) ──

@router.get("/tables")
def get_tables():
    try:
        return {"tables": ingest.list_tables()}
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing env variable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run management ──────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    table: str | None = None
    query: str | None = None

    @model_validator(mode="after")
    def at_least_one(self):
        if not self.table and not self.query:
            raise ValueError("Provide either 'table' or 'query'.")
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
        result = ingest.run(table=req.table, query=req.query)
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing env variable: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return RunResponse(status="completed", **result)


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
