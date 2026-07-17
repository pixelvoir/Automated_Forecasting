"""Per-run pipeline progress file — the UI's live progress rail + log read this.

Every heavy stage runs in a spawned job subprocess whose only IPC back to the API
parent is the terminal result queue, so progress travels through the filesystem
instead: writers here mutate ``runs/{run_id}/progress.json`` atomically (tmp file +
``os.replace``) and the API's ``GET /runs/{id}/progress`` simply reads it. Keep this
module dependency-free — the spawn child imports it on every stage.

Writers are deliberately non-fatal: a progress write must never fail a pipeline stage.
"""
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

# Canonical stage order — the UI rail renders exactly these, in this order.
STAGES = ["ingest", "pre_clean_eda", "intent", "clean", "validate",
          "forecast_eda", "model_select", "train", "report"]

_LOG_CAP = 200  # ring buffer: keep the newest N log events


def _now() -> str:
    # Local-aware, not UTC: the UI log panel slices the wall-clock time straight out of
    # this string, and UTC read as local time was hours off for anyone not in UTC.
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _path(run_id: str) -> Path:
    return RUNS_DIR / run_id / "progress.json"


def _load(run_id: str) -> dict:
    try:
        return json.loads(_path(run_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(run_id: str, prog: dict) -> None:
    prog["run_id"] = run_id
    prog["updated_at"] = _now()
    prog["log"] = (prog.get("log") or [])[-_LOG_CAP:]
    p = _path(run_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(prog), encoding="utf-8")
    os.replace(tmp, p)  # atomic on NTFS — the poller never sees a half-written file


def _mutate(run_id: str, fn) -> None:
    """Load → mutate → atomically save; swallow everything (progress is telemetry)."""
    try:
        if not (RUNS_DIR / run_id).is_dir():
            return
        prog = _load(run_id)
        prog.setdefault("stages", {})
        fn(prog)
        _save(run_id, prog)
    except Exception:  # noqa: BLE001 — never fail a stage over progress bookkeeping
        pass


def _log(prog: dict, stage: str, msg: str) -> None:
    prog.setdefault("log", []).append({"t": _now(), "stage": stage, "msg": msg})


def stage_start(run_id: str, stage: str, detail: str | None = None,
                total: int | None = None, eta_seconds: float | None = None,
                log_msg: str | None = None) -> None:
    def fn(prog):
        # Single-slot job manager: a new job stage starting means any OTHER stage still
        # marked "running" was killed/preempted and its stage_finish never ran — without
        # this sweep the rail showed two stages running at once (e.g. a preempted intent
        # job stuck "running" for the rest of the run while cleaning re-ran). "report"
        # is exempt both ways: it runs in-process and can legitimately overlap a job.
        if stage != "report":
            for other, e in prog["stages"].items():
                if other not in (stage, "report") and (e or {}).get("status") == "running":
                    e["status"] = "cancelled"
        entry = {"status": "running", "started_at": _now()}
        if detail:
            entry["detail"] = detail
        if total is not None:
            entry["total"] = int(total)
        # Keep a pre-dispatch ETA (written by the route before the job child boots)
        # unless the caller supplies a fresh one.
        prev_eta = (prog["stages"].get(stage) or {}).get("eta_seconds")
        if eta_seconds is not None:
            entry["eta_seconds"] = round(float(eta_seconds), 1)
        elif prev_eta is not None:
            entry["eta_seconds"] = prev_eta
        prog["stages"][stage] = entry
        prog["active_stage"] = stage
        _log(prog, stage, log_msg or ((detail or stage.replace("_", " ")) + " started"))
    _mutate(run_id, fn)


def stage_update(run_id: str, stage: str, msg: str, current: int | None = None,
                 total: int | None = None) -> None:
    def fn(prog):
        entry = prog["stages"].setdefault(stage, {"status": "running"})
        entry["detail"] = msg
        if current is not None:
            entry["current"] = int(current)
        if total is not None:
            entry["total"] = int(total)
        _log(prog, stage, msg)
    _mutate(run_id, fn)


def stage_finish(run_id: str, stage: str, status: str = "done",
                 detail: str | None = None) -> None:
    def fn(prog):
        entry = prog["stages"].setdefault(stage, {})
        entry["status"] = status
        entry["ended_at"] = _now()
        started = entry.get("started_at")
        if started:
            try:
                # Both sides are timezone-aware, so this stays correct even across a
                # mix of old UTC-stamped and new local-stamped entries on disk.
                t0 = datetime.fromisoformat(started)
                entry["seconds"] = round(
                    (datetime.now().astimezone() - t0).total_seconds(), 1)
            except ValueError:
                pass
        if detail:
            entry["detail"] = detail
        if prog.get("active_stage") == stage:
            prog["active_stage"] = None
        secs = entry.get("seconds")
        suffix = f" in {secs}s" if secs is not None else ""
        msg = (detail + suffix) if detail else \
            (f"{stage.replace('_', ' ')} {status}{suffix}")
        _log(prog, stage, msg)
    _mutate(run_id, fn)


def model_event(run_id: str, model_id: str, index: int, total: int, status: str,
                seconds: float | None = None, mase: float | None = None,
                eta_seconds: float | None = None) -> None:
    """Per-model progress inside the training loop (the rail's indented sub-rows)."""
    def fn(prog):
        entry = prog["stages"].setdefault("train", {"status": "running"})
        models = entry.setdefault("models", [])
        rec = next((m for m in models if m["id"] == model_id), None)
        if rec is None:
            rec = {"id": model_id}
            models.append(rec)
        rec.update({"index": int(index), "status": status})
        if seconds is not None:
            rec["seconds"] = round(float(seconds), 1)
        if mase is not None:
            rec["mase"] = round(float(mase), 4)
        if eta_seconds is not None:
            rec["eta_seconds"] = round(float(eta_seconds), 1)
        entry["current"] = int(index)
        entry["total"] = int(total)
        if status == "running":
            entry["detail"] = f"training {model_id} ({index}/{total})"
            _log(prog, "train", f"training {model_id} ({index}/{total})…")
        elif status == "done":
            bits = []
            if mase is not None:
                bits.append(f"MASE {mase:.3f}")
            if seconds is not None:
                bits.append(f"{seconds:.0f}s")
            _log(prog, "train", f"{model_id} done" + (" — " + ", ".join(bits) if bits else ""))
        else:
            _log(prog, "train", f"{model_id} {status}")
    _mutate(run_id, fn)


def reset_downstream(run_id: str, from_stage: str) -> None:
    """Mark `from_stage` and everything after it pending — a re-run invalidates them."""
    def fn(prog):
        if from_stage not in STAGES:
            return
        for stage in STAGES[STAGES.index(from_stage):]:
            if stage in prog["stages"]:
                prog["stages"][stage] = {"status": "pending"}
        prog["active_stage"] = None
    _mutate(run_id, fn)
