"""CPU throttling so heavy pipeline stages can't freeze the machine.

The job subprocess used to run at normal priority with every parallel library
(BLAS/OpenMP, LightGBM/XGBoost, torch) defaulting to ALL cores — on a small box a
training run pinned 100% CPU at normal priority and the desktop became unusable.

Two levers, both configured under ``resources:`` in config/settings.yaml:

- ``max_threads`` (default ``auto`` = all cores minus one): exported as the
  OMP/MKL/OpenBLAS/NUMEXPR env caps and used for ML ``n_jobs`` / torch threads.
  ``apply()`` MUST run before numpy/pandas are first imported in a process —
  BLAS pools read the env vars at library load. It is invoked at the top of
  ``api/jobs.py`` / ``api/tasks.py`` (both re-imported inside the spawn child),
  ``api/main.py`` and ``frontend/app.py``.
- ``job_priority`` (default ``below_normal``): ``lower_priority()`` drops the
  calling process to BELOW_NORMAL on Windows (``os.nice`` elsewhere) so the OS
  scheduler keeps the UI responsive even when every core is busy. Called from
  the job child's entry point (api/jobs.py::_entry).

This module must stay import-light (os/sys/yaml only — no numpy/pandas).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "settings.yaml"

_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# BELOW_NORMAL_PRIORITY_CLASS — https://learn.microsoft.com/windows/win32/procthread/scheduling-priorities
_BELOW_NORMAL = 0x00004000


def _resources_cfg() -> dict:
    try:
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        return cfg.get("resources") or {}
    except Exception:  # noqa: BLE001 — a broken/missing config must never block startup
        return {}


def thread_cap() -> int:
    """Resolved thread budget: ``resources.max_threads`` or auto = max(1, cores − 1)."""
    raw = _resources_cfg().get("max_threads", "auto")
    if isinstance(raw, int) and raw > 0:
        return raw
    return max(1, (os.cpu_count() or 2) - 1)


def apply() -> int:
    """Export the BLAS/OpenMP thread-cap env vars (setdefault — explicit env wins).

    Returns the cap. Safe to call repeatedly; only effective if it runs before the
    first numpy/pandas import in the process.
    """
    cap = thread_cap()
    for var in _ENV_VARS:
        os.environ.setdefault(var, str(cap))
    return cap


def lower_priority() -> None:
    """Drop the calling process to below-normal CPU priority (no-op if disabled)."""
    if _resources_cfg().get("job_priority", "below_normal") != "below_normal":
        return
    try:
        if sys.platform == "win32":
            import ctypes

            # Explicit types: HANDLE is pointer-sized — the default c_int restype
            # truncates the -1 pseudo-handle on 64-bit and SetPriorityClass fails
            # with ERROR_INVALID_HANDLE.
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), _BELOW_NORMAL)
        else:
            os.nice(10)
    except Exception:  # noqa: BLE001 — priority is best-effort, never fail the job
        pass
