"""API entrypoint."""
from pipeline import resource_limits

# Cap BLAS/OpenMP threads before anything imports numpy/pandas (routes pulls in the
# pipeline modules) — keeps the API process itself from spawning all-core thread pools.
resource_limits.apply()

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from api import routes

load_dotenv()

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    RUNS_DIR.mkdir(exist_ok=True)
    yield


app = FastAPI(title="Automated Forecasting Agent", lifespan=lifespan)
app.include_router(routes.router)


@app.get("/health")
def health():
    return {"status": "ok"}
