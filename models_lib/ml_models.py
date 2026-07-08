"""Gradient-boosted-tree forecasters (LightGBM / XGBoost) on EDA-seeded features.

The feature recipe comes from Stage 6 (``pipeline/feature_builder``); we reuse its
``build_supervised`` so the SAME transformation drives training and the recursive multi-step
forecast (rebuild the lags/rolling from the growing history each step). Every feature is a
function of PAST values, so recursion is leakage-free.

Two shapes:
  * ``MLForecaster`` — one series (aggregate scope, or per-series-local). Implements the
    standard ``Forecaster`` interface so the trainer's walk-forward harness backtests it
    exactly like a classical model.
  * ``run_global`` — one model across an entire panel (Stage 5 policy ``global``): trains on
    every series' rows with a series-id categorical, holdout-backtests, and forecasts each
    series recursively. This is the scalable path R3 routes big panels to.

``tune`` runs a small Optuna search (budget shrinks on large data) seeded by series length,
minimizing RMSE on a chronological validation tail — no LLM involvement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_lib.base_model import Forecaster, apply_transform, invert_transform, conformal_intervals
from pipeline.feature_builder import build_supervised

_ML_IDS = {"lightgbm", "xgboost"}


def _make_regressor(model_id: str, params: dict):
    if model_id == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**params)
    if model_id == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(**params)
    raise KeyError(model_id)


def _default_params(model_id: str, n_rows: int) -> dict:
    if model_id == "lightgbm":
        return dict(n_estimators=300, learning_rate=0.05,
                    num_leaves=min(31, max(7, n_rows // 3)),
                    min_child_samples=max(5, n_rows // 20),
                    subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                    reg_lambda=1.0, random_state=0, n_jobs=1, verbose=-1)
    return dict(n_estimators=300, learning_rate=0.05,
                max_depth=min(6, max(2, int(np.log2(max(n_rows, 4))))),
                min_child_weight=max(1, n_rows // 40),
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=0, n_jobs=1, verbosity=0)


def _supervised(hist: pd.DataFrame, recipe: dict) -> tuple[pd.DataFrame, list[str]]:
    sup = build_supervised(hist, recipe, dropna=True)
    return sup, list(sup.attrs.get("feature_cols") or [])


def _recursive(model, feature_cols, hist, recipe, future_ds, exog_cols, future_exog=None,
               series_id=None):
    """Recursive multi-step point forecast on the (already transformed) history."""
    cols = ["ds", "y", *exog_cols]
    work = hist[cols].copy().reset_index(drop=True)
    work["pos"] = np.arange(len(work), dtype=float)
    preds = []
    for k in range(len(future_ds)):
        row = {"ds": future_ds[k], "y": np.nan, "pos": float(len(work))}
        for c in exog_cols:
            row[c] = (future_exog[c].iloc[k]
                      if future_exog is not None and c in future_exog.columns else np.nan)
        work = pd.concat([work, pd.DataFrame([row])], ignore_index=True)
        sup = build_supervised(work, recipe, dropna=False)
        x = sup.iloc[[-1]].copy()
        if series_id is not None:
            x["series_id"] = series_id
        yhat = float(model.predict(x[feature_cols])[0])
        work.loc[work.index[-1], "y"] = yhat
        preds.append(yhat)
    return np.asarray(preds, dtype=float)


class MLForecaster(Forecaster):
    uses_features = True

    def __init__(self, model_id: str, recipe: dict, params: dict | None = None,
                 exog_cols=None, transform: str = "none"):
        super().__init__(transform)
        self.model_id = model_id
        self.recipe = recipe
        self.params = params
        self.exog_cols = list(exog_cols or [])

    def _fit(self, hist):
        sup, feats = _supervised(hist, self.recipe)
        if len(sup) < 5 or not feats:
            raise RuntimeError(f"{self.model_id}: not enough featurized rows to train "
                               f"({len(sup)} after warmup)")
        self._feature_cols = feats
        params = self.params or _default_params(self.model_id, len(sup))
        self._model = _make_regressor(self.model_id, params)
        self._model.fit(sup[feats], sup["y"].to_numpy(dtype=float))
        self.params_ = params

    def _predict(self, h, future_ds, future_exog=None):
        # self._hist already carries the transformed y (base_model.fit); predict returns
        # transformed values, base_model inverts.
        return _recursive(self._model, self._feature_cols, self._hist, self.recipe,
                          list(future_ds), self.exog_cols, future_exog)


# ── Hyperparameter tuning (Optuna) ────────────────────────────────────────────

def tune(model_id: str, hist: pd.DataFrame, recipe: dict, cfg: dict, n_trials: int):
    """Small chronological-validation Optuna search. Returns (best_params, summary)."""
    sup, feats = _supervised(hist, recipe)
    if len(sup) < 12 or not feats or n_trials <= 0:
        return _default_params(model_id, max(len(sup), 4)), {"trials": 0, "note": "defaults (too little data to tune)"}

    n = len(sup)
    val = max(recipe.get("horizon", 1), max(3, n // 5))
    val = min(val, n - 6)
    X, y = sup[feats], sup["y"].to_numpy(dtype=float)
    Xtr, ytr = X.iloc[:-val], y[:-val]
    Xva, yva = X.iloc[-val:], y[-val:]

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        if model_id == "lightgbm":
            p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 600),
                     learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                     num_leaves=trial.suggest_int("num_leaves", 7, 63),
                     min_child_samples=trial.suggest_int("min_child_samples", 3, max(4, n // 8)),
                     subsample=trial.suggest_float("subsample", 0.6, 1.0),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                     reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                     subsample_freq=1, random_state=0, n_jobs=1, verbose=-1)
        else:
            p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 600),
                     learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                     max_depth=trial.suggest_int("max_depth", 2, 8),
                     min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                     subsample=trial.suggest_float("subsample", 0.6, 1.0),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                     reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                     random_state=0, n_jobs=1, verbosity=0)
        model = _make_regressor(model_id, p)
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        return float(np.sqrt(np.mean((yva - pred) ** 2)))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    best = dict(study.best_params)
    best.update({"random_state": 0, "n_jobs": 1})
    best.update({"verbose": -1} if model_id == "lightgbm" else {"verbosity": 0})
    return best, {"trials": int(n_trials), "best_rmse": round(float(study.best_value), 4),
                  "validation_points": int(val)}


# ── Global panel path (one model, all series) ─────────────────────────────────

def run_global(frame, recipe, model_id, cfg, horizon, freq_alias, transform, params=None,
               levels=(80, 95)):
    """Train ONE ML model across the whole panel (series-id categorical), holdout-backtest,
    and forecast every series recursively. Returns the uniform per-model result dict."""
    from models_lib.base_model import future_index

    exog_cols = [c for c in frame.columns if c not in ("unique_id", "ds", "y")]
    work = frame.copy()
    work["ds"] = pd.to_datetime(work["ds"])
    work["y"] = apply_transform(work["y"].to_numpy(), transform)

    ids = work["unique_id"].astype("category")
    code_of = {uid: i for i, uid in enumerate(ids.cat.categories)}

    # Per-series supervised rows with a series-id feature.
    train_parts, holdout_records, feats = [], [], []
    per_series = {uid: g.sort_values("ds").reset_index(drop=True)
                  for uid, g in work.groupby("unique_id", observed=True, sort=False)}
    for uid, g in per_series.items():
        sup = build_supervised(g, recipe, dropna=True)
        if len(sup) < 5:
            continue
        feats = sup.attrs.get("feature_cols") or feats
        sup = sup.copy()
        sup["series_id"] = code_of[uid]
        # hold out the last `horizon` featurized rows of each series for backtesting
        hcut = max(len(sup) - horizon, 5)
        train_parts.append(sup.iloc[:hcut])
        holdout_records.append((uid, sup.iloc[hcut:]))
    if not train_parts or not feats:
        return {"cv": None, "forecast": None, "params": {}, "tuning": None,
                "strategy": "none", "error": "global ML: no series had enough history"}

    feat_all = [*feats, "series_id"]
    train_df = pd.concat(train_parts, ignore_index=True)
    params = params or _default_params(model_id, len(train_df))
    model = _make_regressor(model_id, params)
    model.fit(train_df[feat_all], train_df["y"].to_numpy(dtype=float))

    # Holdout backtest (already-trained model, direct one-shot on held rows).
    cv_rows = []
    for uid, hold in holdout_records:
        if hold.empty:
            continue
        pred = model.predict(hold[feat_all])
        cv_rows.append(pd.DataFrame({
            "unique_id": uid, "ds": hold["ds"].to_numpy(),
            "actual": invert_transform(hold["y"].to_numpy(), transform),
            "predicted": invert_transform(pred, transform)}))
    cv_df = pd.concat(cv_rows, ignore_index=True) if cv_rows else None

    # Refit on ALL rows, then recursive forecast per series.
    full_parts = []
    for uid, g in per_series.items():
        sup = build_supervised(g, recipe, dropna=True)
        if len(sup) < 5:
            continue
        sup = sup.copy(); sup["series_id"] = code_of[uid]
        full_parts.append(sup)
    full_df = pd.concat(full_parts, ignore_index=True)
    model.fit(full_df[feat_all], full_df["y"].to_numpy(dtype=float))

    fc_parts = []
    for uid, g in per_series.items():
        fut_ds = future_index(g["ds"].iloc[-1], horizon, freq_alias)
        yhat_t = _recursive(model, feat_all, g, recipe, list(fut_ds), exog_cols,
                            series_id=code_of[uid])
        fc_parts.append(pd.DataFrame({"unique_id": uid, "ds": fut_ds,
                                      "yhat": invert_transform(yhat_t, transform)}))
    fc_df = pd.concat(fc_parts, ignore_index=True)

    return {"cv": cv_df, "forecast": fc_df, "params": params, "tuning": None,
            "strategy": "holdout", "error": None, "_global": True,
            "_fitted": {"backend": "ml_global", "model": model, "feature_cols": feat_all,
                        "recipe": recipe, "transform": transform}}


def make(model_id: str, recipe: dict, params, exog_cols, transform: str):
    return MLForecaster(model_id, recipe, params=params, exog_cols=exog_cols, transform=transform)
