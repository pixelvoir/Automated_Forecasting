"""Intermittent-demand forecasters — Croston (optimized) and TSB, hand-rolled.

statsforecast is unavailable in this environment (compiled-kernel segfault under numpy
2.4.x), so these classic sparse-demand methods are implemented directly. Both forecast the
demand RATE (a flat level), which is the honest thing to do when most periods are zero:

  croston → SES on non-zero demand sizes and on the intervals between them; forecast =
            size / interval. "Optimized" grid-searches the smoothing constant on in-sample
            rolling MSE.
  tsb     → updates a demand-probability every period (not just at demand epochs), so it
            reacts to demand that is dying out (obsolescence). forecast = prob × size.
"""
from __future__ import annotations

import numpy as np

from models_lib.base_model import Forecaster

IDS = {"croston", "tsb"}
_ALPHA_GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
_TSB_ALPHA_D = 0.1
_TSB_ALPHA_P = 0.1


def _croston_rolling(y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """One-step-ahead in-sample forecasts + the final demand-rate level."""
    fcs = np.zeros(len(y))
    z = p = None
    last_nz = -1
    forecast = 0.0
    for t in range(len(y)):
        fcs[t] = forecast
        if y[t] > 0:
            if z is None:
                z, p = float(y[t]), float(t + 1)
            else:
                interval = t - last_nz
                z = alpha * y[t] + (1 - alpha) * z
                p = alpha * interval + (1 - alpha) * p
            last_nz = t
            forecast = z / p if p > 0 else 0.0
    return fcs, forecast


class CrostonForecaster(Forecaster):
    def _fit(self, hist):
        y = np.clip(hist["y"].to_numpy(dtype=float), 0, None)
        best_alpha, best_mse, best_level = _ALPHA_GRID[0], np.inf, 0.0
        for a in _ALPHA_GRID:
            fcs, level = _croston_rolling(y, a)
            mse = float(np.mean((y - fcs) ** 2))
            if mse < best_mse:
                best_alpha, best_mse, best_level = a, mse, level
        self._level = best_level
        self.params_ = {"alpha": best_alpha}

    def _predict(self, h, future_ds, future_exog=None):
        return np.repeat(self._level, h)


class TSBForecaster(Forecaster):
    def __init__(self, transform="none", alpha_d=_TSB_ALPHA_D, alpha_p=_TSB_ALPHA_P):
        super().__init__(transform)
        self.alpha_d, self.alpha_p = alpha_d, alpha_p

    def _fit(self, hist):
        y = np.clip(hist["y"].to_numpy(dtype=float), 0, None)
        nz = y[y > 0]
        z = float(nz.mean()) if len(nz) else 0.0          # demand size
        p = float((y > 0).mean())                          # demand probability
        for t in range(len(y)):
            occurred = 1.0 if y[t] > 0 else 0.0
            p = self.alpha_p * occurred + (1 - self.alpha_p) * p
            if y[t] > 0:
                z = self.alpha_d * y[t] + (1 - self.alpha_d) * z
        self._level = p * z
        self.params_ = {"alpha_d": self.alpha_d, "alpha_p": self.alpha_p}

    def _predict(self, h, future_ds, future_exog=None):
        return np.repeat(self._level, h)


def make(model_id: str, transform: str):
    if model_id == "croston":
        return CrostonForecaster(transform)
    if model_id == "tsb":
        return TSBForecaster(transform)
    raise KeyError(model_id)
