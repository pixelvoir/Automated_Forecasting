"""Shared forecaster interface + helpers for every backend in models_lib.

statsforecast's compiled ``coreforecast`` kernel segfaults under this environment's numpy
2.4.x, so the classical models are built directly on statsmodels + pmdarima (the same
libraries Stage 4 already uses successfully here) and the intermittent ones are hand-rolled.
Everything shares ONE tiny interface so a single walk-forward CV harness in
``pipeline/trainer.py`` can backtest classical, intermittent and ML models identically:

    class Forecaster:
        def fit(self, hist: DataFrame[ds, y, *exog], period: int) -> self
        def predict(self, h, future_ds, future_exog=None) -> np.ndarray   # point forecast

Prediction intervals are NOT produced per-model (analytic intervals differ wildly and some
models have none). Instead the trainer builds **conformal** intervals from the model's own
backtest residuals via ``conformal_intervals`` here — model-agnostic and uniformly
calibrated, which is the recommended approach for short series.

The log transform (Stage 4's heteroskedastic + skew recommendation) is applied to y before
fitting and inverted on the forecast; ``Forecaster`` centralizes it so subclasses only ever
see/produce transformed values through ``self._to`` / ``self._back``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# z-multipliers for a Gaussian interval fallback when too few residuals exist for empirical
# quantiles.
_Z = {80: 1.2816, 95: 1.9600}


def apply_transform(y, transform: str):
    """log1p when the recipe asked for a log transform."""
    arr = np.asarray(y, dtype=float)
    if transform == "log":
        return np.log1p(np.clip(arr, a_min=0, a_max=None))
    return arr


def invert_transform(y, transform: str):
    if transform == "log":
        return np.expm1(np.asarray(y, dtype=float))
    return np.asarray(y, dtype=float)


class Forecaster:
    """Base class: subclasses implement ``_fit`` / ``_predict`` on the (possibly
    log-transformed) target; the transform round-trip lives here."""

    #: subclasses that consume exog / calendar features override to True
    uses_features = False

    def __init__(self, transform: str = "none"):
        self.transform = transform
        self.period = 1

    # public API -------------------------------------------------------------
    def fit(self, hist: pd.DataFrame, period: int) -> "Forecaster":
        self.period = max(int(period or 1), 1)
        h = hist.copy().reset_index(drop=True)
        h["ds"] = pd.to_datetime(h["ds"])
        self._y_raw = h["y"].to_numpy(dtype=float)
        h = h.assign(y=apply_transform(h["y"].to_numpy(), self.transform))
        self._hist = h
        self._fit(h)
        return self

    def predict(self, h: int, future_ds, future_exog=None) -> np.ndarray:
        yhat_t = np.asarray(self._predict(h, pd.to_datetime(pd.Index(future_ds)), future_exog),
                            dtype=float)
        return invert_transform(yhat_t, self.transform)

    # subclass hooks ---------------------------------------------------------
    def _fit(self, hist: pd.DataFrame):
        raise NotImplementedError

    def _predict(self, h: int, future_ds: pd.DatetimeIndex, future_exog) -> np.ndarray:
        raise NotImplementedError


def conformal_intervals(point: np.ndarray, residuals_by_step: dict, levels) -> dict:
    """Two-sided prediction bands for a point forecast from the backtest error distribution.

    ``residuals_by_step`` maps horizon step (1..h) → array of (actual − predicted) errors seen
    at that step across CV windows; a step with too few errors falls back to the pooled
    residuals, then to a Gaussian on their std. Returns {level: (lo_array, hi_array)}."""
    point = np.asarray(point, dtype=float)
    h = len(point)
    pooled = np.concatenate([np.asarray(v, dtype=float) for v in residuals_by_step.values()]) \
        if residuals_by_step else np.array([])
    pooled = pooled[np.isfinite(pooled)]

    out = {lvl: (np.full(h, np.nan), np.full(h, np.nan)) for lvl in levels}
    for k in range(h):
        res = np.asarray(residuals_by_step.get(k + 1, []), dtype=float)
        res = res[np.isfinite(res)]
        if len(res) < 5:
            res = pooled
        for lvl in levels:
            lo_arr, hi_arr = out[lvl]
            if len(res) >= 8:
                lo_q = np.percentile(res, (100 - lvl) / 2)
                hi_q = np.percentile(res, 100 - (100 - lvl) / 2)
            else:
                sd = float(np.std(res)) if len(res) > 1 else 0.0
                z = _Z.get(int(lvl), 1.96)
                lo_q, hi_q = -z * sd, z * sd
            lo_arr[k] = point[k] + lo_q
            hi_arr[k] = point[k] + hi_q
    return out


def future_index(last_ds, h: int, freq_alias: str) -> pd.DatetimeIndex:
    """The h future timestamps on the regular grid, starting one step after ``last_ds``."""
    return pd.date_range(start=pd.Timestamp(last_ds), periods=h + 1, freq=freq_alias)[1:]
