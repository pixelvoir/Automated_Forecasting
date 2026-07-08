"""Stage 7 evaluation metrics — one place, zero-safe, scale-aware.

Every backend returns aligned (actual, predicted) arrays from its walk-forward backtest;
the trainer scores them all through ``compute_metrics`` so the leaderboard is apples-to-
apples. MASE is the ranking metric (Stage 5 sets ``training_hints.metric``): it divides the
model's MAE by the in-sample MAE of the seasonal-naive forecast, so MASE < 1 means "beats
the naive benchmark" and it stays defined on the count/intermittent series where MAPE blows
up on zeros.
"""
import numpy as np


def _round(x, nd=4):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def mase_scale(y_insample, season: int) -> float | None:
    """In-sample MAE of the seasonal-naive forecast (the MASE denominator). Falls back to
    the lag-1 naive when the series is shorter than one seasonal cycle."""
    y = np.asarray(y_insample, dtype=float)
    y = y[np.isfinite(y)]
    m = max(int(season or 1), 1)
    if len(y) <= m:
        m = 1
    if len(y) <= m:
        return None
    d = np.abs(y[m:] - y[:-m])
    scale = float(np.mean(d)) if len(d) else np.nan
    return scale if (np.isfinite(scale) and scale > 0) else None


def compute_metrics(actual, pred, scale: float | None = None) -> dict:
    """MAE / RMSE / MAPE / sMAPE / MASE / R² over the aligned backtest points. Non-finite
    pairs are dropped; ratio metrics guard against division by (near-)zero actuals."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    n = min(len(a), len(p))
    a, p = a[:n], p[:n]
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    empty = {"mae": None, "rmse": None, "mape": None, "smape": None, "mase": None,
             "r2": None, "n": 0}
    if len(a) == 0:
        return empty

    err = a - p
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    nz = np.abs(a) > 1e-9
    mape = float(np.mean(np.abs(err[nz] / a[nz])) * 100) if nz.any() else None

    denom = np.abs(a) + np.abs(p)
    sm = denom > 1e-9
    smape = float(np.mean(2.0 * np.abs(err[sm]) / denom[sm]) * 100) if sm.any() else None

    mase = float(mae / scale) if (scale and scale > 0) else None

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else None

    return {"mae": _round(mae), "rmse": _round(rmse), "mape": _round(mape, 3),
            "smape": _round(smape, 3), "mase": _round(mase), "r2": _round(r2, 4),
            "n": int(len(a))}
