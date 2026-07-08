"""Classical univariate forecasters on statsmodels + pmdarima.

statsforecast's compiled kernel segfaults under this env's numpy 2.4.x, so these are built
on the libraries Stage 4 already runs here. Each is a thin ``Forecaster`` (fit on the
possibly log-transformed series, point forecast out); the trainer's walk-forward harness
handles CV, metrics and conformal intervals uniformly.

  seasonal_naive → repeat the last observed seasonal cycle
  auto_ets       → statsmodels ETSModel, small AICc search over trend/damped/seasonal
  auto_theta     → statsmodels ThetaModel (deseasonalize + damped Theta extrapolation)
  auto_arima     → pmdarima.auto_arima (seasonal only when the period is SARIMA-feasible)
  mstl           → statsmodels MSTL decomposition + Theta on the deseasonalized series

All are univariate: exogenous drivers are exploited by the ML backend (leading-lag
features); a robust classical exog/SARIMAX path is deliberately out of scope here.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from models_lib.base_model import Forecaster

IDS = {"seasonal_naive", "auto_ets", "auto_arima", "auto_theta", "mstl"}


def _seasonal_ok(n: int, period: int) -> bool:
    return period and period > 1 and n >= 2 * period + 1


# ── Seasonal naive ────────────────────────────────────────────────────────────

class SeasonalNaiveForecaster(Forecaster):
    def _fit(self, hist):
        self._y = hist["y"].to_numpy(dtype=float)

    def _predict(self, h, future_ds, future_exog=None):
        y, m = self._y, self.period
        if m <= 1 or len(y) < m:
            return np.repeat(y[-1] if len(y) else 0.0, h)
        cycle = y[-m:]
        return np.array([cycle[k % m] for k in range(h)], dtype=float)


# ── AutoETS (exponential smoothing) ───────────────────────────────────────────

class AutoETSForecaster(Forecaster):
    def _fit(self, hist):
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        y = pd.Series(hist["y"].to_numpy(dtype=float))
        n = len(y)
        seas = self.period if _seasonal_ok(n, self.period) else None
        positive = bool(np.all(y > 0)) and self.transform != "log"

        configs = []
        for trend in ("add", None):
            damps = (True, False) if trend else (False,)
            for damped in damps:
                for seasonal in (("add", "mul") if seas else (None,)):
                    if seasonal == "mul" and not positive:
                        continue
                    err = "add"
                    configs.append(dict(error=err, trend=trend, damped_trend=damped,
                                        seasonal=seasonal,
                                        seasonal_periods=seas if seasonal else None))
        best, best_aicc, best_cfg = None, np.inf, None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for cfg in configs:
                try:
                    res = ETSModel(y, **cfg).fit(disp=False, maxiter=200)
                    aicc = getattr(res, "aicc", None)
                    aicc = aicc if aicc is not None and np.isfinite(aicc) else res.aic
                    if aicc < best_aicc:
                        best, best_aicc, best_cfg = res, aicc, cfg
                except Exception:
                    continue
        if best is None:
            raise RuntimeError("AutoETS: no configuration converged")
        self._res, self.params_ = best, {k: v for k, v in best_cfg.items() if v is not None}

    def _predict(self, h, future_ds, future_exog=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(self._res.forecast(h), dtype=float)


# ── AutoTheta ─────────────────────────────────────────────────────────────────

class AutoThetaForecaster(Forecaster):
    def _fit(self, hist):
        from statsmodels.tsa.forecasting.theta import ThetaModel
        y = pd.Series(hist["y"].to_numpy(dtype=float),
                      index=pd.to_datetime(hist["ds"]))
        deseason = _seasonal_ok(len(y), self.period)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._res = ThetaModel(y, period=self.period if deseason else 1,
                                   deseasonalize=deseason).fit()

    def _predict(self, h, future_ds, future_exog=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(self._res.forecast(h), dtype=float)


# ── AutoARIMA (pmdarima) ──────────────────────────────────────────────────────

class AutoARIMAForecaster(Forecaster):
    def __init__(self, transform="none", sarima_max_period: int = 24):
        super().__init__(transform)
        self.sarima_max_period = sarima_max_period

    def _fit(self, hist):
        import pmdarima as pm
        y = hist["y"].to_numpy(dtype=float)
        seasonal = bool(_seasonal_ok(len(y), self.period)
                        and self.period <= self.sarima_max_period)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = pm.auto_arima(
                y, m=self.period if seasonal else 1, seasonal=seasonal,
                stepwise=True, suppress_warnings=True, error_action="ignore",
                max_p=5, max_q=5, max_P=2, max_Q=2, maxiter=50)
        self.params_ = {"order": list(self._model.order),
                        "seasonal_order": list(self._model.seasonal_order)}

    def _predict(self, h, future_ds, future_exog=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(self._model.predict(n_periods=h), dtype=float)


# ── MSTL (multi-seasonal decomposition + Theta on the deseasonalized series) ──

class MSTLForecaster(Forecaster):
    def __init__(self, transform="none", seasonal_periods=None):
        super().__init__(transform)
        self._periods_req = [int(p) for p in (seasonal_periods or []) if p and int(p) > 1]

    def _fit(self, hist):
        from statsmodels.tsa.forecasting.theta import ThetaModel
        from statsmodels.tsa.seasonal import MSTL
        y = pd.Series(hist["y"].to_numpy(dtype=float), index=pd.to_datetime(hist["ds"]))
        periods = [p for p in (self._periods_req or [self.period]) if _seasonal_ok(len(y), p)]
        if not periods:
            raise RuntimeError("MSTL: no seasonal period with 2+ cycles of history")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = MSTL(y, periods=tuple(periods) if len(periods) > 1 else periods[0]).fit()
        seasonal = res.seasonal
        if isinstance(seasonal, pd.DataFrame):
            seasonal_total = seasonal.sum(axis=1)
            self._season_cycles = [(int(p), seasonal.iloc[:, i].to_numpy(dtype=float))
                                   for i, p in enumerate(periods)]
        else:
            seasonal_total = seasonal
            self._season_cycles = [(periods[0], np.asarray(seasonal, dtype=float))]
        deseason = (y - seasonal_total)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._trend_res = ThetaModel(deseason, period=1, deseasonalize=False).fit()

    def _predict(self, h, future_ds, future_exog=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = np.asarray(self._trend_res.forecast(h), dtype=float)
        season_future = np.zeros(h)
        for period, comp in self._season_cycles:
            cycle = comp[-period:]
            season_future += np.array([cycle[k % period] for k in range(h)], dtype=float)
        return base + season_future


def make(model_id: str, transform: str, seasonal_periods, sarima_max_period: int):
    if model_id == "seasonal_naive":
        return SeasonalNaiveForecaster(transform)
    if model_id == "auto_ets":
        return AutoETSForecaster(transform)
    if model_id == "auto_theta":
        return AutoThetaForecaster(transform)
    if model_id == "auto_arima":
        return AutoARIMAForecaster(transform, sarima_max_period=sarima_max_period)
    if model_id == "mstl":
        return MSTLForecaster(transform, seasonal_periods=seasonal_periods)
    raise KeyError(model_id)
