from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import json
import math

import numpy as np


@dataclass
class DriftSignal:
    psi: float
    mae_drift: float
    combined_score: float
    trigger: bool
    reference_summary: dict[str, float]
    current_summary: dict[str, float]


@dataclass
class RunState:
    selected_model: str
    residual_summary: dict[str, float]
    drift_score: float
    residual_sample: list[float] | None = None


def compute_drift_signal(reference_residuals: np.ndarray | list[float], current_residuals: np.ndarray | list[float], threshold: float = 0.2) -> DriftSignal:
    reference = np.asarray(reference_residuals, dtype=float)
    current = np.asarray(current_residuals, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) < 10 or len(current) < 10:
        return DriftSignal(0.0, 0.0, 0.0, False, _summary(reference), _summary(current))

    psi = _population_stability_index(reference, current)
    mae_drift = abs(float(np.mean(np.abs(current))) - float(np.mean(np.abs(reference)))) / (float(np.mean(np.abs(reference))) + 1e-9)
    combined_score = float(psi + mae_drift)
    return DriftSignal(
        psi=float(psi),
        mae_drift=float(mae_drift),
        combined_score=combined_score,
        trigger=combined_score >= threshold,
        reference_summary=_summary(reference),
        current_summary=_summary(current),
    )


def load_state(path: str | Path, series_key: str | None = None) -> RunState | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if series_key is not None:
        # Panel runs now store every series in one file; missing keys are treated like no prior state.
        data = data.get(series_key)
        if not isinstance(data, dict):
            return None
    return RunState(
        selected_model=str(data.get("selected_model", "naive")),
        residual_summary=dict(data.get("residual_summary", {})),
        drift_score=float(data.get("drift_score", 0.0)),
        residual_sample=list(data.get("residual_sample", [])) or None,
    )


def save_state(path: str | Path, state: RunState, series_key: str | None = None) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    if series_key is not None:
        # Load-modify-write preserves other series entries during sequential panel forecasting.
        existing = {}
        if file_path.exists():
            loaded = json.loads(file_path.read_text(encoding="utf-8"))
            existing = loaded if isinstance(loaded, dict) else {}
        existing[series_key] = payload
        payload = existing
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def residual_summary(residuals: np.ndarray | list[float]) -> dict[str, float]:
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0, "p90_abs": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p90_abs": float(np.quantile(np.abs(values), 0.9)),
    }


def _population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(reference, quantiles))
    if len(cuts) < 3:
        return 0.0
    ref_hist, edges = np.histogram(reference, bins=cuts)
    cur_hist, _ = np.histogram(current, bins=cuts)
    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    cur_pct = cur_hist / max(cur_hist.sum(), 1)
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _summary(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0, "count": 0.0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "count": float(len(values))}
