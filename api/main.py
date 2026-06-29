"""API entrypoint."""
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
    (RUNS_DIR / "startup_test.txt").write_text("Server started.")
    yield


app = FastAPI(title="Automated Forecasting Agent", lifespan=lifespan)
app.include_router(routes.router)


@app.get("/health")
def health():
    return {"status": "ok"}
