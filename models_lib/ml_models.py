"""Gradient-boosted-tree forecasters (LightGBM / XGBoost) — direct multi-horizon design.

Rebuilt (2026-07) around the strategies that made the manual milk-collection pipeline
substantially more accurate than the original recursive backend here:

* **Seasonal-index deseasonalization** — per-series multiplicative index (phase mean /
  overall mean, train-only, clipped 0.2–5.0); the trees model the deseasonalized target and
  the index is multiplied back at predict time. Trees cannot extrapolate multiplicative
  seasonality, so removing it first is the single biggest accuracy win.
* **Direct multi-horizon training** — each origin row is paired with every horizon step
  h=1..H (features = origin features + `horizon` + target-phase + target seasonal index);
  one model predicts the whole horizon in one batch. No recursive rollout, so no error
  compounding step over step.
* **Quantile objectives** — the point model is the P50 (median) quantile regressor; interval
  quantiles (from `interval_levels`, e.g. 80→.10/.90) are fitted lazily by
  ``predict_quantiles`` so walk-forward CV only ever pays for the P50 fit. Non-crossing is
  enforced by sorting across the quantile axis.
* **Early stopping on a chronological split** — last ~20% of origin rows are the eval set;
  the found tree count is then refit on ALL rows so recent history is never wasted.

The feature recipe still comes from Stage 6 (``pipeline/feature_builder.build_supervised``)
— lags/rolling/Fourier/calendar are computed on the DEseasonalized series. Exogenous
drivers enter through their origin-time leading-lag features (a direct model needs no
future exog values).

``run_global`` trains ONE model across an entire panel (policy ``global``): per-series
seasonal indices, a `series_id` categorical, per-series level/scale scalars and a
share-of-total lag feature, with a stratified (series × horizon) row cap for huge panels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_lib.base_model import Forecaster, apply_transform, invert_transform
from pipeline.feature_builder import _safe, build_supervised

_ML_IDS = {"lightgbm", "xgboost"}

# temp.txt-derived base params (the manual pipeline's settings, generalized). n_estimators
# is a CAP — early stopping finds the real tree count.
_BASE_LGBM = dict(n_estimators=800, learning_rate=0.025, num_leaves=63, max_depth=7,
                  min_child_samples=15, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.75, reg_alpha=0.3, reg_lambda=1.5,
                  random_state=0, n_jobs=-1, verbose=-1)
_BASE_XGB = dict(n_estimators=800, learning_rate=0.025, max_depth=7,
                 min_child_weight=5, subsample=0.8, colsample_bytree=0.75,
                 reg_alpha=0.3, reg_lambda=1.5, random_state=0, n_jobs=-1, verbosity=0)

_DIRECT_FEATS = ["horizon", "target_phase", "target_sindex"]


def _default_params(model_id: str, n_rows: int) -> dict:
    p = dict(_BASE_LGBM if model_id == "lightgbm" else _BASE_XGB)
    if model_id == "lightgbm":
        p["num_leaves"] = min(63, max(7, n_rows // 8))
        p["min_child_samples"] = min(15, max(3, n_rows // 30))
    else:
        p["min_child_weight"] = min(5, max(1, n_rows // 60))
    return p


def _alpha_params(model_id: str, params: dict, alpha: float) -> dict:
    """Quantile-objective params for one alpha (P50 = the point model)."""
    p = dict(params)
    if model_id == "lightgbm":
        p.update(objective="quantile", alpha=float(alpha))
    else:
        p.update(objective="reg:quantileerror", quantile_alpha=float(alpha))
    return p


def _make_regressor(model_id: str, params: dict, early_stopping_rounds: int | None = None):
    if model_id == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**params)
    import xgboost as xgb
    if early_stopping_rounds:
        return xgb.XGBRegressor(**params, early_stopping_rounds=int(early_stopping_rounds))
    return xgb.XGBRegressor(**params)


def _fit_quantile(model_id, params, alpha, X, y, Xv=None, yv=None, es_rounds=80):
    """Fit one quantile regressor with early stopping on the chronological eval split, then
    refit at the found tree count on ALL rows (eval included) so the most recent data —
    exactly the rows that matter most for forecasting — still trains the final model."""
    p = _alpha_params(model_id, params, alpha)
    if Xv is None or len(Xv) < 8:
        m = _make_regressor(model_id, p)
        m.fit(X, y)
        return m, None
    if model_id == "lightgbm":
        import lightgbm as lgb
        m = _make_regressor(model_id, p)
        m.fit(X, y, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(int(es_rounds), verbose=False),
                         lgb.log_evaluation(0)])
        best = int(m.best_iteration_ or p["n_estimators"])
    else:
        m = _make_regressor(model_id, p, early_stopping_rounds=es_rounds)
        m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        bi = getattr(m, "best_iteration", None)
        best = int(bi) + 1 if bi is not None else int(p["n_estimators"])
    p2 = dict(p)
    p2["n_estimators"] = max(best, 10)
    final = _make_regressor(model_id, p2)
    final.fit(pd.concat([X, Xv]), np.concatenate([y, yv]))
    return final, best


# ── Seasonal index ─────────────────────────────────────────────────────────────

def seasonal_index(y: np.ndarray, period: int) -> np.ndarray:
    """Per-phase multiplicative index (phase mean / overall mean), clipped 0.2–5.0.
    Falls back to a flat index (no deseasonalization) whenever the ratio is meaningless:
    period <= 1, under two full cycles, negative values, or a ~zero overall level."""
    period = max(int(period or 1), 1)
    ones = np.ones(period)
    y = np.asarray(y, dtype=float)
    if period <= 1 or len(y) < 2 * period or np.nanmin(y) < 0:
        return ones
    overall = np.nanmean(y)
    if not np.isfinite(overall) or abs(overall) < 1e-12:
        return ones
    phase = np.arange(len(y)) % period
    idx = np.ones(period)
    for p in range(period):
        m = np.nanmean(y[phase == p]) if np.any(phase == p) else np.nan
        if np.isfinite(m):
            idx[p] = m / overall
    idx = np.clip(idx, 0.2, 5.0)
    return idx if np.all(np.isfinite(idx)) else ones


# ── Direct multi-horizon dataset ───────────────────────────────────────────────

def _exog_target_features(recipe: dict, work: pd.DataFrame) -> tuple[list[dict], dict]:
    """Leading exog drivers usable AT the target time: for a driver with lead `lag`,
    the value at t+h−lag is already known whenever h ≤ lag — a per-horizon feature the
    origin-time exoglag alone can't express. Returns (specs, {col: np.ndarray})."""
    specs, arrays = [], {}
    for e in recipe.get("exog_lags") or []:
        col, lag = e["col"], int(e["lag"])
        if lag >= 1 and col in work.columns:
            specs.append({"col": col, "lag": lag, "name": f"exogtgt_{_safe(col)}"})
            arrays[col] = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
    return specs, arrays


def _add_exog_target(p: pd.DataFrame, tgt_pos: np.ndarray, h: int,
                     exog_specs: list[dict], exog_arrays: dict) -> None:
    for spec in exog_specs:
        arr = exog_arrays.get(spec["col"])
        src = tgt_pos - spec["lag"]
        known = (h <= spec["lag"]) & (src >= 0) & (src < len(arr))
        vals = np.full(len(tgt_pos), np.nan)
        vals[known] = arr[src[known]]
        p[spec["name"]] = vals


def _direct_dataset(sup: pd.DataFrame, feats: list[str], y_ds: np.ndarray,
                    idx: np.ndarray, horizon: int,
                    exog_specs: list[dict] | None = None,
                    exog_arrays: dict | None = None) -> pd.DataFrame:
    """Expand origin rows × horizon steps. `sup` must carry the original `pos` column;
    labels come from the deseasonalized series at pos + h."""
    n, period = len(y_ds), len(idx)
    parts = []
    for h in range(1, int(horizon) + 1):
        tgt = sup["pos"].to_numpy(dtype=int) + h
        valid = tgt < n
        if not valid.any():
            continue
        p = sup.loc[valid, [*feats, "pos"]].copy()
        tp = (tgt[valid] % period).astype(int)
        p["horizon"] = float(h)
        p["target_phase"] = tp.astype(float)
        p["target_sindex"] = idx[tp]
        _add_exog_target(p, tgt[valid], h, exog_specs or [], exog_arrays or {})
        p["label"] = y_ds[tgt[valid]]
        parts.append(p)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _origin_rows(work: pd.DataFrame, recipe: dict, feats: list[str], idx: np.ndarray,
                 h: int, exog_specs: list[dict] | None = None,
                 exog_arrays: dict | None = None) -> pd.DataFrame:
    """The h prediction rows: the LAST origin row's features replicated across horizons
    with the target-side columns updated."""
    sup = build_supervised(work, recipe, dropna=False)
    last = sup.iloc[[-1]]
    period = len(idx)
    origin_pos = int(last["pos"].iloc[0])
    rows = pd.concat([last[[*feats, "pos"]]] * h, ignore_index=True)
    tgt = origin_pos + np.arange(1, h + 1)
    tp = (tgt % period).astype(int)
    rows["horizon"] = np.arange(1, h + 1, dtype=float)
    rows["target_phase"] = tp.astype(float)
    rows["target_sindex"] = idx[tp]
    steps = np.arange(1, h + 1)  # per-row horizon step: value known iff step ≤ lag
    for spec in exog_specs or []:
        arr = (exog_arrays or {}).get(spec["col"])
        if arr is None:
            continue
        src = tgt - spec["lag"]
        known = (steps <= spec["lag"]) & (src >= 0) & (src < len(arr))
        vals = np.full(h, np.nan)
        vals[known] = arr[src[known]]
        rows[spec["name"]] = vals
    return rows


def _chrono_split(train_df: pd.DataFrame, es_rounds: int):
    """Chronological eval split on origin position (last ~20% of origins)."""
    pos = train_df["pos"].to_numpy()
    cut = np.quantile(np.unique(pos), 0.8)
    val_mask = pos > cut
    if val_mask.sum() < 8 or (~val_mask).sum() < 20:
        return train_df, None, None
    return train_df[~val_mask], train_df[val_mask], es_rounds


# ── Single-series forecaster (aggregate scope / per-series-local) ─────────────

class DirectMLForecaster(Forecaster):
    """Direct multi-horizon quantile GBM on one series. Implements the standard
    ``Forecaster`` interface so the trainer's walk-forward harness backtests it exactly
    like a classical model — each CV fit trains only the P50 model; the interval
    quantiles are fitted lazily on the final model via ``predict_quantiles``."""

    uses_features = True

    def __init__(self, model_id: str, recipe: dict, params: dict | None = None,
                 exog_cols=None, transform: str = "none", train_cfg: dict | None = None):
        super().__init__(transform)
        self.model_id = model_id
        self.recipe = recipe
        self.params = params
        self.exog_cols = list(exog_cols or [])
        self.train_cfg = train_cfg or {}
        self._qmodels: dict[float, object] = {}

    def _fit(self, hist):
        period = max(int(self.recipe.get("primary_period") or self.period or 1), 1)
        horizon = max(int(self.recipe.get("horizon") or 1), 1)
        y = hist["y"].to_numpy(dtype=float)
        self._idx = seasonal_index(y, period)
        phase = np.arange(len(y)) % len(self._idx)
        y_ds = y / self._idx[phase]

        work = hist.copy()
        work["y"] = y_ds
        self._work = work
        sup = build_supervised(work, self.recipe, dropna=True)
        feats = list(sup.attrs.get("feature_cols") or [])
        if len(sup) < 5 or not feats:
            raise RuntimeError(f"{self.model_id}: not enough featurized rows to train "
                               f"({len(sup)} after warmup)")
        self._feats = feats
        self._exog_specs, self._exog_arrays = _exog_target_features(self.recipe, work)
        train_df = _direct_dataset(sup, feats, y_ds, self._idx, horizon,
                                   self._exog_specs, self._exog_arrays)
        if train_df.empty:
            raise RuntimeError(f"{self.model_id}: no (origin, horizon) pairs to train on")
        self._feat_all = [*feats, "pos", *_DIRECT_FEATS,
                          *[s["name"] for s in self._exog_specs]]

        params = self.params or _default_params(self.model_id, len(train_df))
        es_rounds = int(self.train_cfg.get("early_stopping_rounds", 80))
        tr, va, es = _chrono_split(train_df, es_rounds)
        self._split = (tr, va, es)
        self._model, best_iter = _fit_quantile(
            self.model_id, params, 0.5,
            tr[self._feat_all], tr["label"].to_numpy(dtype=float),
            va[self._feat_all] if va is not None else None,
            va["label"].to_numpy(dtype=float) if va is not None else None,
            es_rounds=es or es_rounds)
        self.params_ = {**params, "best_iteration": best_iter}

    def _predict(self, h, future_ds, future_exog=None):
        rows = _origin_rows(self._work, self.recipe, self._feats, self._idx, int(h),
                            self._exog_specs, self._exog_arrays)
        self._pred_rows = rows
        yhat_ds = self._model.predict(rows[self._feat_all])
        return np.asarray(yhat_ds, dtype=float) * rows["target_sindex"].to_numpy()

    def predict_quantiles(self, h: int, future_ds, levels=(80, 95)) -> dict:
        """{level: (lo, hi)} on the ORIGINAL scale. Fits the interval quantile models
        lazily (cached), enforces non-crossing, clips at 0 for non-negative targets."""
        rows = _origin_rows(self._work, self.recipe, self._feats, self._idx, int(h),
                            self._exog_specs, self._exog_arrays)
        sindex = rows["target_sindex"].to_numpy()
        alphas = sorted({a for lvl in levels
                         for a in ((100 - lvl) / 200.0, 1 - (100 - lvl) / 200.0)})
        tr, va, es = self._split
        preds = {}
        for a in alphas:
            if a not in self._qmodels:
                params = {k: v for k, v in self.params_.items() if k != "best_iteration"}
                self._qmodels[a], _ = _fit_quantile(
                    self.model_id, params, a,
                    tr[self._feat_all], tr["label"].to_numpy(dtype=float),
                    va[self._feat_all] if va is not None else None,
                    va["label"].to_numpy(dtype=float) if va is not None else None,
                    es_rounds=es or int(self.train_cfg.get("early_stopping_rounds", 80)))
            q_t = np.asarray(self._qmodels[a].predict(rows[self._feat_all]), dtype=float) * sindex
            preds[a] = invert_transform(q_t, self.transform)
        # non-crossing: sort the stacked quantile predictions per step
        stack = np.sort(np.vstack([preds[a] for a in alphas]), axis=0)
        if np.nanmin(self._y_raw) >= 0:
            stack = np.clip(stack, 0, None)
        by_alpha = {a: stack[i] for i, a in enumerate(alphas)}
        return {int(lvl): (by_alpha[(100 - lvl) / 200.0], by_alpha[1 - (100 - lvl) / 200.0])
                for lvl in levels}


# ── Hyperparameter tuning (Optuna) ────────────────────────────────────────────

def tune(model_id: str, hist: pd.DataFrame, recipe: dict, n_trials: int,
         train_cfg: dict | None = None):
    """Optuna search seeded around the temp.txt base params, on the SAME direct
    multi-horizon dataset the forecaster trains on; objective = P50 pinball (≡ MAE/2) on
    the chronological tail. Returns (best_params, summary)."""
    train_cfg = train_cfg or {}
    period = max(int(recipe.get("primary_period") or 1), 1)
    horizon = max(int(recipe.get("horizon") or 1), 1)
    h = hist.copy().reset_index(drop=True)
    y = apply_transform(h["y"].to_numpy(), recipe.get("transform", "none"))
    idx = seasonal_index(y, period)
    y_ds = y / idx[np.arange(len(y)) % len(idx)]
    work = h.assign(y=y_ds)
    sup = build_supervised(work, recipe, dropna=True)
    feats = list(sup.attrs.get("feature_cols") or [])
    if len(sup) < 12 or not feats or n_trials <= 0:
        return (_default_params(model_id, max(len(sup) * horizon, 4)),
                {"trials": 0, "note": "defaults (no tuning budget or too little data)"})
    train_df = _direct_dataset(sup, feats, y_ds, idx, horizon)
    feat_all = [*feats, "pos", *_DIRECT_FEATS]

    pos = train_df["pos"].to_numpy()
    cut = np.quantile(np.unique(pos), 0.8)
    vm = pos > cut
    if vm.sum() < 5:
        return (_default_params(model_id, len(train_df)),
                {"trials": 0, "note": "defaults (validation tail too small)"})
    Xtr, ytr = train_df.loc[~vm, feat_all], train_df.loc[~vm, "label"].to_numpy(dtype=float)
    Xva, yva = train_df.loc[vm, feat_all], train_df.loc[vm, "label"].to_numpy(dtype=float)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = _default_params(model_id, len(train_df))

    def objective(trial):
        p = dict(base)
        p.update(learning_rate=trial.suggest_float("learning_rate", 0.008, 0.1, log=True),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True))
        if model_id == "lightgbm":
            p.update(num_leaves=trial.suggest_int("num_leaves", 15, 127),
                     max_depth=trial.suggest_int("max_depth", 4, 10),
                     min_child_samples=trial.suggest_int("min_child_samples", 5, 40))
        else:
            p.update(max_depth=trial.suggest_int("max_depth", 3, 10),
                     min_child_weight=trial.suggest_int("min_child_weight", 1, 20))
        m = _make_regressor(model_id, _alpha_params(model_id, p, 0.5))
        m.fit(Xtr, ytr)
        pred = m.predict(Xva)
        return float(np.mean(np.abs(yva - pred)))  # MAE ≡ 2·pinball(0.5)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    best = dict(base)
    best.update(study.best_params)
    return best, {"trials": int(n_trials), "best_mae": round(float(study.best_value), 4),
                  "validation_points": int(vm.sum())}


# ── Global panel path (one model, all series) ─────────────────────────────────

def run_global(frame, recipe, model_id, cfg, horizon, freq_alias, transform, params=None,
               levels=(80, 95), train_cfg=None):
    """ONE direct multi-horizon model across the whole panel. Per-series seasonal indices,
    a `series_id` categorical, per-series level/scale scalars and a share-of-total lag-1
    feature; holdout = the last `horizon` periods of every series; stratified
    (series × horizon) row cap for huge panels."""
    from models_lib.base_model import future_index
    train_cfg = train_cfg or {}
    horizon = max(int(horizon), 1)
    period = max(int(recipe.get("primary_period") or 1), 1)
    es_rounds = int(train_cfg.get("early_stopping_rounds", 80))
    max_rows = int(train_cfg.get("max_train_rows") or 1_500_000)

    work = frame.copy()
    work["ds"] = pd.to_datetime(work["ds"])
    work = work.sort_values(["unique_id", "ds"])
    work["y_t"] = apply_transform(work["y"].to_numpy(), transform)
    # share-of-total lag-1: the series' share of the panel total one step earlier —
    # cheap cross-series signal a per-series model can never see.
    total = work.groupby("ds")["y_t"].transform("sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total.to_numpy() > 0, work["y_t"].to_numpy() / total.to_numpy(), 0.0)
    work["share_lag1"] = pd.Series(share, index=work.index).groupby(work["unique_id"]).shift(1)

    ids = work["unique_id"].astype("category")
    code_of = {uid: i for i, uid in enumerate(ids.cat.categories)}

    # Row budget applied PER SERIES inside the loop, not after concat: the old
    # post-concat cap materialized the full origins×horizon expansion first (~50M rows
    # × ~45 float64 cols on a Favorita-scale panel — several times physical RAM, and it
    # hard-crashed an 8GB box). Sampling stratified by horizon within each series is
    # equivalent to the old global (series_id, horizon) stratified sample, but the
    # pre-concat peak now stays at ~max_rows regardless of panel size.
    ser_budget = max(int(max_rows / max(len(code_of), 1)), horizon)

    def _budget(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) <= ser_budget:
            return df
        return (df.groupby("horizon", group_keys=False)
                .sample(frac=ser_budget / len(df), random_state=0))

    per_series, feats = {}, []
    train_parts, hold_parts, hold_region_parts = [], [], []
    for uid, g in work.groupby("unique_id", observed=True, sort=False):
        g = g.reset_index(drop=True)
        y = g["y_t"].to_numpy(dtype=float)
        idx = seasonal_index(y, period)
        y_ds = y / idx[np.arange(len(y)) % len(idx)]
        gsup_in = g.drop(columns=["y"]).rename(columns={"y_t": "y"}).assign(y=y_ds)
        sup = build_supervised(gsup_in, recipe, dropna=True)
        if len(sup) < 3:
            per_series[uid] = {"g": gsup_in, "idx": idx, "y_ds": y_ds, "trainable": False}
            continue
        feats = list(sup.attrs.get("feature_cols") or feats)
        dd = _direct_dataset(sup, [*feats, "share_lag1"], y_ds, idx, horizon)
        if dd.empty:
            per_series[uid] = {"g": gsup_in, "idx": idx, "y_ds": y_ds, "trainable": False}
            continue
        dd["series_id"] = code_of[uid]
        # train-only per-series level/scale scalars (first len-h points → no holdout leak)
        cut = max(len(y_ds) - horizon, 1)
        dd["ser_mean"] = float(np.nanmean(y_ds[:cut]))
        dd["ser_std"] = float(np.nanstd(y_ds[:cut]))
        # rows whose TARGET lands in the last `horizon` periods are the holdout
        n = len(y_ds)
        tgt_pos = dd["pos"].to_numpy(dtype=int) + dd["horizon"].to_numpy(dtype=int)
        hold_mask = tgt_pos >= n - horizon
        # holdout predictions come from the LAST pre-holdout origin only (a true
        # forecast-origin backtest, not every overlapping origin)
        origin_last = int(n - horizon - 1)
        hp = dd[hold_mask & (dd["pos"].to_numpy(dtype=int) == origin_last)]
        train_parts.append(_budget(dd[~hold_mask]))
        if not hp.empty:
            hold_parts.append((uid, hp))
        # holdout-region rows kept separately (small: targets in the last h periods
        # only) so the final refit can still see the most recent data without holding
        # every series' full expansion in memory
        hr = _budget(dd[hold_mask])
        if not hr.empty:
            hold_region_parts.append(hr)
        per_series[uid] = {"g": gsup_in, "idx": idx, "y_ds": y_ds, "trainable": True}
    if not train_parts or not feats:
        return {"cv": None, "forecast": None, "params": {}, "tuning": None,
                "strategy": "none", "error": "global ML: no series had enough history"}

    feat_all = [*feats, "share_lag1", "pos", *_DIRECT_FEATS,
                "series_id", "ser_mean", "ser_std"]
    train_df = pd.concat(train_parts, ignore_index=True)
    train_parts.clear()  # per-series budgeting already capped this near max_rows
    if len(train_df) > max_rows:
        frac = max_rows / len(train_df)
        train_df = (train_df.groupby(["series_id", "horizon"], group_keys=False)
                    .sample(frac=frac, random_state=0))

    params = params or _default_params(model_id, len(train_df))
    tr, va, es = _chrono_split(train_df, es_rounds)

    # optional Optuna pass on the global dataset (sampled) — `max` profile territory
    tuning = None
    n_trials = int(train_cfg.get("optuna_trials_global", 0))
    if n_trials > 0 and va is not None:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        samp = tr.sample(n=min(len(tr), 200_000), random_state=0)

        def objective(trial):
            p = dict(params)
            p.update(learning_rate=trial.suggest_float("learning_rate", 0.008, 0.1, log=True),
                     subsample=trial.suggest_float("subsample", 0.6, 1.0),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0))
            if model_id == "lightgbm":
                p["num_leaves"] = trial.suggest_int("num_leaves", 31, 255)
            m = _make_regressor(model_id, _alpha_params(model_id, p, 0.5))
            m.fit(samp[feat_all], samp["label"].to_numpy(dtype=float))
            return float(np.mean(np.abs(va["label"].to_numpy(dtype=float)
                                        - m.predict(va[feat_all]))))
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        params = {**params, **study.best_params}
        tuning = {"trials": n_trials, "best_mae": round(float(study.best_value), 4)}

    model, best_iter = _fit_quantile(
        model_id, params, 0.5, tr[feat_all], tr["label"].to_numpy(dtype=float),
        va[feat_all] if va is not None else None,
        va["label"].to_numpy(dtype=float) if va is not None else None,
        es_rounds=es or es_rounds)

    # Holdout backtest per series (last pre-holdout origin → h steps).
    cv_rows = []
    for uid, hp in hold_parts:
        pred_ds = model.predict(hp[feat_all])
        sindex = hp["target_sindex"].to_numpy()
        info = per_series[uid]
        g = info["g"]
        tgt_pos = (hp["pos"].to_numpy(dtype=int) + hp["horizon"].to_numpy(dtype=int))
        cv_rows.append(pd.DataFrame({
            "unique_id": uid,
            "ds": g["ds"].to_numpy()[tgt_pos],
            "step": hp["horizon"].to_numpy(dtype=int),
            "actual": invert_transform(info["y_ds"][tgt_pos] * info["idx"][tgt_pos % len(info["idx"])], transform),
            "predicted": invert_transform(pred_ds * sindex, transform)}))
    cv_df = pd.concat(cv_rows, ignore_index=True) if cv_rows else None

    # Final model: refit at the found tree count on all budgeted rows, holdout region
    # included (train_df + hold_region_parts together cover every series' full history
    # at the same per-series sampling rate).
    full_df = pd.concat([train_df, *hold_region_parts], ignore_index=True)
    if len(full_df) > max_rows:
        frac = max_rows / len(full_df)
        full_df = (full_df.groupby(["series_id", "horizon"], group_keys=False)
                   .sample(frac=frac, random_state=0))
    p_final = _alpha_params(model_id, params, 0.5)
    if best_iter:
        p_final["n_estimators"] = max(int(best_iter), 10)
    final = _make_regressor(model_id, p_final)
    final.fit(full_df[feat_all], full_df["label"].to_numpy(dtype=float))

    # Forecast every series from its last origin.
    fc_parts = []
    for uid, info in per_series.items():
        g, idx = info["g"], info["idx"]
        rows = _origin_rows(g, recipe, [*feats, "share_lag1"], idx, horizon)
        rows["series_id"] = code_of[uid]
        y_ds = info["y_ds"]
        rows["ser_mean"] = float(np.nanmean(y_ds))
        rows["ser_std"] = float(np.nanstd(y_ds))
        yhat_ds = final.predict(rows[feat_all])
        fut_ds = future_index(g["ds"].iloc[-1], horizon, freq_alias)
        fc_parts.append(pd.DataFrame({
            "unique_id": uid, "ds": fut_ds,
            "yhat": invert_transform(yhat_ds * rows["target_sindex"].to_numpy(), transform)}))
    fc_df = pd.concat(fc_parts, ignore_index=True)

    return {"cv": cv_df, "forecast": fc_df,
            "params": {**params, "best_iteration": best_iter}, "tuning": tuning,
            "strategy": "holdout", "error": None, "_global": True,
            "_fitted": {"backend": "ml_global_direct", "model": final,
                        "feature_cols": feat_all, "recipe": recipe, "transform": transform}}


def make(model_id: str, recipe: dict, params, exog_cols, transform: str,
         train_cfg: dict | None = None):
    return DirectMLForecaster(model_id, recipe, params=params, exog_cols=exog_cols,
                              transform=transform, train_cfg=train_cfg)
