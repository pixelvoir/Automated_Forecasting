"""Prophet forecaster — guarded (prophet is not installed on this machine).

Written so that ``pip install prophet`` enables it with zero other changes (the catalog's
runtime ``available`` check already gates selectability). Additive trend + configurable
seasonalities; the log transform is handled by the base class. Changepoints are left to
Prophet's automatic detection (Stage 4's structural-break dates could seed
``changepoints=`` here once prophet is available and worth tuning).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_lib.base_model import Forecaster

IDS = {"prophet"}

# frequency label → approximate period length in days, used to register custom seasonalities.
_PERIOD_DAYS = {"hourly": 1 / 24, "daily": 1, "weekly": 7, "monthly": 30.5,
                "quarterly": 91.3, "yearly": 365.25}


class ProphetForecaster(Forecaster):
    def __init__(self, transform="none", seasonal_periods=None, frequency="monthly"):
        super().__init__(transform)
        self._periods = [int(p) for p in (seasonal_periods or []) if p and int(p) > 1]
        self._frequency = frequency

    def _fit(self, hist):
        from prophet import Prophet
        import logging
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        df = pd.DataFrame({"ds": pd.to_datetime(hist["ds"]),
                           "y": hist["y"].to_numpy(dtype=float)})
        m = Prophet(weekly_seasonality=False, daily_seasonality=False,
                    yearly_seasonality=False)
        days = _PERIOD_DAYS.get(self._frequency, 30.5)
        for p in self._periods:
            m.add_seasonality(name=f"seas_{p}", period=p * days, fourier_order=min(p, 10))
        m.fit(df)
        self._model = m

    def _predict(self, h, future_ds, future_exog=None):
        future = pd.DataFrame({"ds": pd.to_datetime(pd.Index(future_ds))})
        fc = self._model.predict(future)
        return np.asarray(fc["yhat"].to_numpy(), dtype=float)


def make(model_id: str, transform: str, seasonal_periods, frequency: str):
    if model_id == "prophet":
        return ProphetForecaster(transform, seasonal_periods, frequency)
    raise KeyError(model_id)
