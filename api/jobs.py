"""Single-slot cancellable runner for heavy pipeline stages.

Heavy stages (ingest, pre-clean EDA, cleaning, validation) run in a child process so a
stuck multi-million-row computation can be terminated the moment a new run starts, instead
of pinning every CPU core and wedging the machine. Only ONE heavy job runs at a time — a new
job preempts (kills) whatever is currently running — which also prevents concurrent
million-row jobs from exhausting memory and freezing the box.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import threading
from typing import Any, Callable

_ctx = mp.get_context("spawn")
_lock = threading.Lock()
_active: dict[str, Any] = {"proc": None, "track_id": None}


class JobCancelled(Exception):
    """The job was terminated because a newer run preempted it (or it was cancelled)."""


class JobError(Exception):
    """The job process reported an exception; the message carries the original error text."""


def _entry(func, args, kwargs, q):
    """Child-process entry point: run the task and report result/error back through the queue."""
    try:
        q.put({"ok": True, "value": func(*args, **kwargs)})
    except Exception as exc:  # noqa: BLE001 — surface any failure text to the parent
        q.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def cancel_active() -> bool:
    """Terminate the currently running job, if any. Returns True if something was killed."""
    with _lock:
        p = _active["proc"]
        _active["proc"] = None
        _active["track_id"] = None
    if p is None or not p.is_alive():
        return False
    p.terminate()
    p.join(timeout=5)
    if p.is_alive():
        p.kill()
        p.join(timeout=5)
    return True


def run_job(func: Callable, *args, track_id: str | None = None, **kwargs):
    """Run ``func(*args, **kwargs)`` in a child process, cancelling any active job first.

    Blocks until the job finishes. Returns the task's return value. Raises ``JobError`` if the
    task raised, or ``JobCancelled`` if a newer run preempted this one before it completed.
    """
    cancel_active()

    q = _ctx.Queue()
    p = _ctx.Process(target=_entry, args=(func, args, kwargs, q), daemon=True)
    with _lock:
        _active["proc"] = p
        _active["track_id"] = track_id
    p.start()

    result = None
    while True:
        try:
            result = q.get(timeout=0.5)
            break
        except _queue.Empty:
            with _lock:
                still_ours = _active["proc"] is p
            if not still_ours:
                # A newer job took the slot (and already terminated us).
                raise JobCancelled("Job preempted by a newer run.")
            if not p.is_alive():
                break  # died without reporting (killed / crashed)

    p.join(timeout=5)
    with _lock:
        if _active["proc"] is p:
            _active["proc"] = None
            _active["track_id"] = None

    if result is None:
        raise JobCancelled("Job was cancelled or terminated before completing.")
    if not result["ok"]:
        raise JobError(result["error"])
    return result["value"]
