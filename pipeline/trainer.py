"""Stage 7: Model Training — hardcoded, EDA-seeded, NO LLM.

Trains the user-selected models (Stage 5 is only a default suggestion), backtests each with
walk-forward cross-validation, and ranks them by MASE. A seasonal-naive baseline is always
trained as the MASE reference but the crowned/serialized ``best_model`` is the best of the
models the USER requested — the dropdown selection is the ultimate choice. Prediction
intervals prefer the backend's native quantile forecasts (direct-ML / nhits); conformal
intervals from CV residuals remain the fallback for the classical models. One CV harness
drives every family (classical / intermittent / ML / deep) through the uniform Forecaster
interface, with a time-vs-accuracy budget from ``training.accuracy_profile``.

Guardrails (the reason this is job-managed + cancellable): a big per-series panel fit
per-series-per-model can be expensive, so ``estimate_cost`` tiers the request fast/moderate/
heavy, the per-series fan-out is capped to the top-K series by volume, and a "heavy" request
must be explicitly confirmed. Both reference runs are aggregate scope, where the series is
already collapsed to ~60-73 points and training is seconds.

Reads:  runs/{id}/feature_report.json (recipe), runs/{id}/model_selection.json (hints +
        eligibility), runs/{id}/model_selection_payload.json (panel size), the Stage 6
        model_frame parquet.
Writes: runs/{id}/training_report.json, models/{id}/best_model.pkl + preprocessor.pkl
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from models_lib import ml_models, registry
from models_lib.base_model import conformal_intervals, future_index
from pipeline import evaluator, progress
from pipeline.forecasting_eda import _pandas_freq

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"
CONFIG_PATH = ROOT / "config" / "settings.yaml"

BASELINE = "seasonal_naive"
ENSEMBLE = "ensemble"   # weighted-ensemble row id (pre-2026-07-17 reports used "ensemble_top2")

_ENSEMBLE_DEFAULTS = {
    "enabled": True,
    "max_members": 5,
    "mase_factor": 1.5,
    "exclude_worse_than_baseline": True,
    "min_shared_points": 3,
    "min_points_per_member": 3,
    "shrink_points_per_member": 3,
}


def _ensemble_cfg(tcfg: dict) -> dict:
    """Merged ensemble config: defaults ← legacy `ensemble_top2` bool ← `training.ensemble`."""
    out = dict(_ENSEMBLE_DEFAULTS)
    if "ensemble_top2" in tcfg:   # pre-refactor settings files can still disable it
        out["enabled"] = bool(tcfg["ensemble_top2"])
    block = tcfg.get("ensemble")
    if isinstance(block, dict):
        out.update({k: block[k] for k in _ENSEMBLE_DEFAULTS if k in block})
    return out


class TrainingHeavyError(Exception):
    """Raised when a request is tiered 'heavy' and confirm_heavy is False. Carries the
    estimate so the route can return it for the UI's confirm prompt."""
    def __init__(self, estimate: dict):
        super().__init__("Training request is heavy — explicit confirmation required.")
        self.estimate = estimate


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    return {"training": cfg.get("training", {}) or {},
            "sarima_max_period": (cfg.get("model_selection", {}) or {}).get("sarima_max_period", 24)}


def _resolve_profile(tcfg: dict, override=None) -> dict:
    """The accuracy-vs-time budget passed to every backend: the selected profile's knobs
    (optuna trials, early-stopping rounds, nhits epochs) + the global caps. The user chose
    to trade longer training for accuracy — `balanced` is the default, `max` is the
    1-3h-class budget. `override` (the UI's per-run picker) wins only when it names a
    profile that actually exists in settings; anything else falls back silently."""
    name = str(override) if override in (tcfg.get("profiles") or {}) \
        else str(tcfg.get("accuracy_profile", "balanced"))
    prof = dict((tcfg.get("profiles") or {}).get(name) or {})
    prof.setdefault("optuna_trials", 30)
    prof.setdefault("early_stopping_rounds", 80)
    prof.setdefault("nhits_max_epochs", 200)
    prof.setdefault("patchtst_max_epochs", 150)
    prof["profile_name"] = name
    prof["max_train_rows"] = int(tcfg.get("max_train_rows", 1_500_000))
    prof["interval_levels"] = list(tcfg.get("interval_levels", [80, 95]))
    # global-panel Optuna only in the `max` profile (it is the 1-3h cost driver)
    prof["optuna_trials_global"] = prof["optuna_trials"] if name == "max" else 0
    return prof


def _round_list(arr, nd=4):
    out = []
    for v in np.asarray(arr, dtype=float):
        out.append(round(float(v), nd) if np.isfinite(v) else None)
    return out


def _json_safe_params(params) -> dict:
    """Numpy scalars (optuna/best_iteration) → native types for the JSON report."""
    out = {}
    for k, v in (params or {}).items():
        if isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
        else:
            out[k] = v
    return out


# ── Cost estimate ─────────────────────────────────────────────────────────────
# Wall-clock priors (seconds) for this machine class, self-corrected over time by
# models/timing_calibration.json (median observed/predicted per model+route, appended
# by record_timings after every training run). Tier thresholds are MINUTES:
# settings.yaml → training.tier_minutes {moderate, heavy}. Prior anchors: Favorita
# fast-profile run (lightgbm global 467s @1.5M capped rows / 1,782 series) and the
# aggregate reference runs (max-profile xgboost 51s / 100 trials, fast lightgbm 3.5s).

_FIT_S = {"seasonal_naive": 0.02, "croston": 0.05, "tsb": 0.05, "auto_theta": 0.25,
          "mstl": 0.6, "auto_ets": 0.8, "prophet": 1.2, "auto_arima": 6.0}
_PER_SERIES_OVERHEAD_S = 0.12   # harness cost per series (groupby/frames/metrics)
_ML_GLOBAL_S_PER_MROW = 130.0   # per million capped direct-dataset rows (ES fit + refit)
_ML_SINGLE_FIT_S = 1.0          # one aggregate direct-ML fit (no tuning)
_OPTUNA_TRIAL_S = 0.4           # per tuning trial on a single series
_OPTUNA_GLOBAL_TRIAL_S = 26.0   # per tuning trial on the sampled global dataset
_NHITS_S_PER_EPOCH_MROW = 8.0   # per effective epoch per million frame rows
_NHITS_MIN_S = 20.0
_PATCHTST_S_PER_EPOCH_MROW = 16.0  # attention ~2× the NHITS MLP cost per epoch-row
_PATCHTST_MIN_S = 30.0
_CHRONOS_LOAD_S = 20.0          # HF weights load once per process
_CHRONOS_S_PER_KSERIES = 25.0   # one batched inference pass per 1000 series (h≈16)
CALIBRATION_PATH = MODELS_DIR / "timing_calibration.json"


def _load_calibration() -> dict:
    try:
        return json.loads(CALIBRATION_PATH.read_text())
    except (OSError, ValueError):
        return {}


def _route_of(mid: str, scope: str, policy: str) -> str:
    """Mirror of run()'s routing: which execution path a model takes."""
    if scope != "per_series":
        return "single"
    if registry.is_global_capable(mid) and (policy == "global" or registry.always_global(mid)):
        return "global"
    return "per_series"


def _model_base_seconds(mid: str, route: str, n_series: int, length: int, horizon: int,
                        period: int, tcfg: dict, knobs: dict) -> float:
    """Prior wall-clock seconds for one model, before calibration."""
    n_series = max(int(n_series or 1), 1)
    length = max(int(length or 100), 10)
    horizon = max(int(horizon or 1), 1)
    n_fits = (1 if n_series >= int(tcfg.get("large_panel_series", 500))
              else int(tcfg.get("cv_windows", 3))) + 1  # CV windows + final refit
    n_routed = (min(n_series, int(tcfg.get("per_series_cap", 200)))
                if route == "per_series" else n_series)

    if mid in registry.CHRONOS_IDS:
        n_eff = n_series if route == "global" else 1
        return (_CHRONOS_LOAD_S
                + 2 * _CHRONOS_S_PER_KSERIES * max(n_eff, 1) / 1000.0
                * max(horizon / 16.0, 0.5))
    if mid in registry.NHITS_IDS:
        rows_m = (n_series if route == "global" else 1) * length / 1e6
        epochs_eff = 0.5 * int(knobs.get("nhits_max_epochs", 150))
        # ×2: truncated-data backtest model + fresh full-data final model
        return max(_NHITS_MIN_S, 2 * epochs_eff * _NHITS_S_PER_EPOCH_MROW * max(rows_m, 0.01))
    if mid in registry.PATCHTST_IDS:
        rows_m = (n_series if route == "global" else 1) * length / 1e6
        epochs_eff = 0.5 * int(knobs.get("patchtst_max_epochs", 100))
        # same two-phase scheme as nhits; attention is ~2× heavier per epoch-row
        return max(_PATCHTST_MIN_S,
                   2 * epochs_eff * _PATCHTST_S_PER_EPOCH_MROW * max(rows_m, 0.01))
    if registry.is_ml(mid):
        if route == "global":
            expanded = n_series * max(length - period - 12, 10) * horizon
            rows = min(expanded, int(tcfg.get("max_train_rows") or 1_500_000))
            s = rows / 1e6 * _ML_GLOBAL_S_PER_MROW + n_series * _PER_SERIES_OVERHEAD_S
            s += int(knobs.get("optuna_trials_global", 0)) * _OPTUNA_GLOBAL_TRIAL_S
            return s
        if route == "per_series":
            return n_routed * (n_fits * _ML_SINGLE_FIT_S + _PER_SERIES_OVERHEAD_S)
        return (n_fits * _ML_SINGLE_FIT_S
                + int(knobs.get("optuna_trials", 0)) * _OPTUNA_TRIAL_S)
    # classical / intermittent — per-series fan-out or a single aggregate series
    fit_s = _FIT_S.get(mid, 1.0)
    if route == "per_series":
        return n_routed * (n_fits * fit_s + _PER_SERIES_OVERHEAD_S)
    return n_fits * fit_s + 1.0  # +1s import/setup


def estimate_cost(n_series: int, policy: str, model_ids, cfg: dict, *,
                  scope: str | None = None, length: int | None = None,
                  horizon: int | None = None, period: int | None = None,
                  profile: str | None = None, include_baseline: bool = True) -> dict:
    """Wall-clock estimate for a training request. Seconds per model from work-based
    priors × a per-(model, route) calibration factor learned from past runs on this
    machine; tier from estimated minutes. The always-trained baseline is included so
    the total honestly covers the whole job."""
    t = cfg["training"]
    scope = scope or ("per_series" if int(n_series or 1) > 1 else "aggregate")
    knobs = _resolve_profile(t, profile)
    cal = _load_calibration()

    ids = [m for m in dict.fromkeys(model_ids) if m in registry.KNOWN_IDS]
    if include_baseline and BASELINE not in ids:
        ids = [*ids, BASELINE]

    per_model, total = [], 0.0
    for mid in ids:
        route = _route_of(mid, scope, policy)
        base = _model_base_seconds(mid, route, n_series, length, horizon,
                                   int(period or 1), t, knobs)
        factors = cal.get(f"{mid}|{route}") or []
        factor = float(np.median(factors)) if factors else 1.0
        secs = base * factor
        per_model.append({"model": mid, "route": route,
                          "seconds": round(secs, 1), "base_seconds": round(base, 1),
                          "calibrated": bool(factors),
                          "is_baseline": mid == BASELINE and mid not in model_ids})
        total += secs

    tiers = t.get("tier_minutes") or {}
    mod_min = float(tiers.get("moderate", 2))
    heavy_min = float(tiers.get("heavy", 15))
    minutes = total / 60.0
    tier = "heavy" if minutes > heavy_min else ("moderate" if minutes > mod_min else "fast")

    rec = None
    slow_ps = [p for p in per_model
               if p["route"] == "per_series" and p["seconds"] > 120
               and not p["is_baseline"]]
    if slow_ps:
        names = ", ".join(p["model"] for p in slow_ps)
        rec = (f"{names}: fits each series individually (capped at "
               f"{int(t.get('per_series_cap', 200))} of {n_series}) — the slow path. "
               "Global models (LightGBM/XGBoost/NHITS/Chronos) cover every series in "
               "one pass.")
    return {"tier": tier, "n_series": int(n_series), "policy": policy, "scope": scope,
            "profile": knobs["profile_name"],
            "n_models": len([m for m in per_model if not m["is_baseline"]]),
            "est_seconds": round(total, 1),
            "est_low_seconds": round(total * 0.5, 1),
            "est_high_seconds": round(total * 2.0, 1),
            "per_model": per_model, "recommendation": rec}


def record_timings(estimate: dict, results: list) -> None:
    """After a run: store observed/predicted per (model, route) so the next estimate
    on this machine starts from reality. Factors clipped to [0.2, 5]; last 6 kept."""
    try:
        cal = _load_calibration()
        by_model = {p["model"]: p for p in estimate.get("per_model") or []}
        for r in results:
            secs = r.get("train_seconds")
            pm = by_model.get(r.get("model"))
            if r.get("error") or not secs or not pm or pm.get("base_seconds", 0) <= 0:
                continue
            f = min(max(secs / pm["base_seconds"], 0.2), 5.0)
            key = f"{r['model']}|{pm['route']}"
            cal[key] = ((cal.get(key) or [])[-5:]) + [round(f, 3)]
        CALIBRATION_PATH.parent.mkdir(exist_ok=True)
        CALIBRATION_PATH.write_text(json.dumps(cal, indent=1))
    except Exception:  # noqa: BLE001 — timing telemetry must never fail a training run
        pass


def estimate_for_run(run_id: str, model_ids, profile: str | None = None,
                     cfg: dict | None = None) -> dict:
    """Estimate from a run's persisted Stage 4/5 outputs — the one authoritative
    entry point for the /train gate and the GET /train-estimate endpoint."""
    if cfg is None:
        cfg = _load_config()
    run_dir = RUNS_DIR / run_id

    def _read(name):
        try:
            return json.loads((run_dir / name).read_text())
        except (OSError, ValueError):
            return {}

    payload = _read("model_selection_payload.json")
    sel = _read("forecast_user_selections.json")
    msel = _read("model_selection.json")
    panel = payload.get("panel") or {}
    n_series = int(panel.get("n_series") or 1)
    length = int((panel.get("series_length") or {}).get("median")
                 or payload.get("series_length") or 100)
    horizon = int(sel.get("horizon") or payload.get("horizon") or 1)
    scope = sel.get("scope") or payload.get("scope") or "aggregate"
    policy = (msel.get("training_hints") or {}).get("policy") or "aggregate"
    period = int(payload.get("seasonality_period") or 1)
    return estimate_cost(n_series, policy, model_ids, cfg, scope=scope, length=length,
                         horizon=horizon, period=period, profile=profile)


# ── Walk-forward CV plan ──────────────────────────────────────────────────────

def _cv_plan(length: int, period: int, horizon: int, cfg: dict, n_series: int) -> dict:
    t = cfg["training"]
    h = max(int(horizon), 1)
    period = max(int(period or 1), 1)
    min_train = max(period * int(t.get("min_train_multiple", 2)), period + 2, 8)
    max_windows = (t.get("cv_windows_large_panel", 1)
                   if n_series >= t.get("large_panel_series", 500)
                   else t.get("cv_windows", 3))
    step = h
    for k in range(int(max_windows), 0, -1):
        if length - h - (k - 1) * step >= min_train:
            return {"n_windows": k, "step_size": step, "h": h, "strategy": "cv"}
    # holdout fallback (relaxed floor)
    if length - h >= max(period + 2, 6):
        return {"n_windows": 1, "step_size": h, "h": h, "strategy": "holdout"}
    return {"n_windows": 0, "step_size": h, "h": h, "strategy": "none"}


# ── Per-series CV + final forecast (uniform harness) ──────────────────────────

def _walk_forward(series_df, make_fn, period, cv_plan, exog_cols):
    n = len(series_df)
    recs = []
    if cv_plan["n_windows"] < 1:
        return recs
    h, step, nw = cv_plan["h"], cv_plan["step_size"], cv_plan["n_windows"]
    for w in range(nw):
        train_end = n - h - (nw - 1 - w) * step
        if train_end < 3:
            continue
        hist = series_df.iloc[:train_end]
        fut = series_df.iloc[train_end:train_end + h]
        if len(fut) == 0:
            continue
        try:
            model = make_fn().fit(hist, period)
            fx = fut[exog_cols] if exog_cols else None
            yhat = model.predict(len(fut), fut["ds"].to_numpy(), fx)
        except Exception:
            continue
        for k in range(len(fut)):
            recs.append({"step": k + 1, "ds": fut["ds"].iloc[k],
                         "actual": float(fut["y"].iloc[k]), "predicted": float(yhat[k])})
    return recs


def _final_forecast(series_df, make_fn, period, horizon, freq_alias, exog_cols, levels=None):
    model = make_fn().fit(series_df, period)
    fut_ds = future_index(series_df["ds"].iloc[-1], horizon, freq_alias)
    yhat = model.predict(horizon, fut_ds.to_numpy(), None)
    # Native quantile intervals (direct-ML / nhits backends) — preferred over conformal.
    native_iv = None
    if levels and hasattr(model, "predict_quantiles"):
        try:
            native_iv = model.predict_quantiles(horizon, fut_ds.to_numpy(), levels)
        except Exception:  # noqa: BLE001 — intervals are best-effort, conformal covers
            native_iv = None
    return fut_ds, np.asarray(yhat, dtype=float), model, native_iv


def _run_per_series(frame, series_ids, make_fn, period, cv_plan, horizon, freq_alias,
                    exog_cols, levels=None):
    """CV + final forecast for each series. Per-series failures (a tiny/degenerate series in
    a panel) are isolated so one bad series can't wipe the whole model's results — for
    aggregate scope there's just one series, so a failure there surfaces to the caller.
    Native quantile intervals are only kept for the single-series case (quantiles cannot
    be summed across a panel — the per-series display falls back to conformal)."""
    cv_all, fc_parts, fitted, n_ok, native_iv = [], [], None, 0, None
    want_native = levels if len(series_ids) == 1 else None
    for uid in series_ids:
        g = frame[frame["unique_id"] == uid].sort_values("ds").reset_index(drop=True)
        try:
            for r in _walk_forward(g, make_fn, period, cv_plan, exog_cols):
                r["unique_id"] = uid
                cv_all.append(r)
            fut_ds, yhat, model, native_iv = _final_forecast(
                g, make_fn, period, horizon, freq_alias, exog_cols, want_native)
            fc_parts.append(pd.DataFrame({"unique_id": uid, "ds": fut_ds, "yhat": yhat}))
            fitted = model
            n_ok += 1
        except Exception:
            if len(series_ids) == 1:
                raise  # aggregate scope: no other series to fall back on
            continue
    if n_ok == 0 and series_ids:
        raise RuntimeError("all series failed to train")
    cv_df = pd.DataFrame(cv_all) if cv_all else None
    fc_df = pd.concat(fc_parts, ignore_index=True) if fc_parts else None
    return cv_df, fc_df, fitted, native_iv


# ── Display aggregation + intervals ───────────────────────────────────────────

def _display_backtest(cv_df, scope):
    if cv_df is None or cv_df.empty:
        return {"ds": [], "actual": [], "predicted": []}
    if scope == "per_series":
        g = cv_df.groupby("ds", as_index=False)[["actual", "predicted"]].sum().sort_values("ds")
    else:
        g = cv_df.sort_values("ds")
    return {"ds": [str(d) for d in g["ds"]], "actual": _round_list(g["actual"]),
            "predicted": _round_list(g["predicted"])}


def _display_forecast(fc_df, scope, cv_df, levels, native_iv=None):
    if fc_df is None or fc_df.empty:
        return {"ds": [], "yhat": []}
    if scope == "per_series":
        g = fc_df.groupby("ds", as_index=False)["yhat"].sum().sort_values("ds")
    else:
        g = fc_df.sort_values("ds")
    point = g["yhat"].to_numpy(dtype=float)

    # Backend-provided quantile intervals (direct-ML / nhits / chronos) beat conformal
    # where present; levels a backend can't produce natively (chronos-bolt's trained
    # quantile range stops at 90) are conformal-filled below so the UI's 95% band never
    # silently disappears.
    native_iv = native_iv or {}
    missing = [lvl for lvl in levels if int(lvl) not in native_iv]

    res_by_step = {}
    if cv_df is not None and not cv_df.empty and "step" in cv_df.columns and scope != "per_series":
        for s, grp in cv_df.groupby("step"):
            res_by_step[int(s)] = (grp["actual"] - grp["predicted"]).to_numpy()
    elif cv_df is not None and not cv_df.empty:
        bt = _display_backtest(cv_df, scope)
        resid = np.asarray(bt["actual"], dtype=float) - np.asarray(bt["predicted"], dtype=float)
        res_by_step = {k + 1: resid for k in range(len(point))}
    intervals = conformal_intervals(point, res_by_step, missing) if (missing and len(point)) else {}

    if not native_iv:
        method = "conformal"
    elif missing:
        method = "quantile+conformal"
    else:
        method = "quantile"
    out = {"ds": [str(d) for d in g["ds"]], "yhat": _round_list(point),
           "interval_method": method}
    for lvl in levels:
        if int(lvl) in native_iv:
            lo, hi = native_iv[int(lvl)]
        elif lvl in intervals:
            lo, hi = intervals[lvl]
        else:
            continue
        out[f"lo{lvl}"] = _round_list(lo)
        out[f"hi{lvl}"] = _round_list(hi)
    return out


def _collect_series_frames(series_frames: list, mid: str, cv_df, fc_df) -> None:
    """Accumulate one model's per-series backtest + forecast rows (long format:
    model, unique_id, ds, kind, actual, predicted) for the series-viewer parquet."""
    if cv_df is not None and not cv_df.empty and "unique_id" in cv_df.columns:
        p = cv_df.copy()
        p["model"], p["kind"] = mid, "backtest"
        series_frames.append(p[["model", "unique_id", "ds", "kind", "actual", "predicted"]])
    if fc_df is not None and not fc_df.empty and "unique_id" in fc_df.columns:
        p = fc_df.rename(columns={"yhat": "predicted"}).copy()
        p["model"], p["kind"], p["actual"] = mid, "forecast", np.nan
        series_frames.append(p[["model", "unique_id", "ds", "kind", "actual", "predicted"]])


def _pooled_mase_scale(frame, season):
    diffs = []
    m = max(int(season or 1), 1)
    for _, g in frame.groupby("unique_id", observed=True):
        y = g.sort_values("ds")["y"].to_numpy(dtype=float)
        mm = m if len(y) > m else 1
        if len(y) > mm:
            diffs.append(np.abs(y[mm:] - y[:-mm]))
    if not diffs:
        return None
    scale = float(np.mean(np.concatenate(diffs)))
    return scale if (np.isfinite(scale) and scale > 0) else None


def _target_insights(entry, hist: pd.Series, horizon: int) -> dict | None:
    """Target-level summary for the Training tab's insights strip: next-period forecast,
    horizon total vs the trailing same-length history total (% change), widest 95% band.
    `hist` is the exact frame's y indexed by ds (panel-summed on per-series scope, which
    matches _display_forecast's panel-summed yhat)."""
    fc = entry.get("forecast") or {}
    ds = fc.get("ds") or []
    yhat = [v for v in (fc.get("yhat") or []) if v is not None]
    if not ds or not yhat:
        return None
    total = float(np.nansum(yhat))
    trailing = float(hist.tail(horizon).sum()) if len(hist) >= horizon else None
    pct = (round((total - trailing) / abs(trailing) * 100, 2)
           if trailing not in (None, 0) else None)
    widest = None
    if fc.get("lo95") and fc.get("hi95"):
        widths = [h - l for l, h in zip(fc["lo95"], fc["hi95"])
                  if l is not None and h is not None]
        widest = round(max(widths), 4) if widths else None
    return {"next_ds": str(ds[0]), "next_value": round(float(yhat[0]), 4),
            "horizon_total": round(total, 4),
            "trailing_total": round(trailing, 4) if trailing is not None else None,
            "pct_change": pct, "widest_interval_95": widest}


def _solve_weights(act, P, mases, ecfg):
    """MAE-optimal convex weights over the member prediction matrix.

    minimize mean|act − P·w|  s.t.  w ≥ 0, Σw = 1  — SLSQP (deterministic, ≤5 dims).
    MAE is the right loss: the leaderboard ranks by MASE = MAE / scale with the pooled
    scale constant on this window, so this minimizes the exact ranking metric.
    Fallbacks: optimizer failure or no in-sample improvement over equal weights →
    inverse-MASE weights → equal weights. Weights are then shrunk toward equal when the
    shared window is small relative to the member count — a 5-member weight vector
    fitted on a dozen points is noise, not signal."""
    n, k = P.shape
    equal = np.full(k, 1.0 / k)

    def _mae(w):
        return float(np.mean(np.abs(act - P @ w)))

    w, method = None, None
    try:
        from scipy.optimize import minimize
        res = minimize(_mae, x0=equal, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                       constraints=[{"type": "eq",
                                     "fun": lambda w: float(np.sum(w) - 1.0)}])
        cand = np.clip(np.asarray(res.x, dtype=float), 0.0, None)
        if res.success and np.all(np.isfinite(cand)) and cand.sum() > 0:
            cand = cand / cand.sum()
            if _mae(cand) <= _mae(equal) + 1e-12:
                w, method = cand, "slsqp"
    except Exception:  # noqa: BLE001 — weight fitting must never kill the run
        pass
    if w is None:
        inv = np.array([1.0 / m if (m and np.isfinite(m) and m > 0) else 0.0
                        for m in mases], dtype=float)
        if inv.sum() > 0:
            w, method = inv / inv.sum(), "inverse_mase"
        else:
            w, method = equal, "equal"
    lam = min(1.0, n / float(max(1, int(ecfg["shrink_points_per_member"]) * k)))
    if lam < 1.0 and method != "equal":
        w = lam * w + (1.0 - lam) * equal
        w = w / w.sum()
        method += "+shrunk"
    return w, method


def _ensemble(results, requested, scale, levels, ecfg, scope=None, series_frames=None) -> dict | None:
    """Weighted-ensemble leaderboard row over the REQUESTED models (2026-07-17; replaces
    the fixed top-2 mean).

    Members = every requested model with a finite backtest MASE, minus low performers
    (not beating the seasonal-naive reference; worse than mase_factor × the best member),
    capped at max_members. Backtests are aligned on shared points, and members are pruned
    until the shared window supports the member count. Weights come from _solve_weights
    (constrained optimization with fallbacks). Metrics use the SAME pooled MASE scale as
    every other row; intervals are conformal from the ensemble's own backtest residuals.
    Returns None — no row at all — when fewer than two members or too few shared points
    survive.

    On per_series scope, alignment/weight-fitting/metrics use the FULL per-series backtest
    rows from `series_frames` (unique_id, ds) — not the per-model `backtest` field on
    `results`, which is already panel-summed-by-ds for chart display (~16 dates) via
    _display_backtest. Fitting weights against 16 aggregate points and then scoring them
    with the per-series-calibrated `scale` is a level mismatch (aggregate-level errors,
    per-series-level scale) that inflated MASE by ~700x on a real panel (2026-07-20 bug).
    Every other row's `metrics` is computed from the full-resolution cv_df before it gets
    downsampled for display — this makes the ensemble consistent with that. The chart
    `backtest` field is still panel-summed-by-ds afterward, matching every other row's
    display convention."""
    cands = [e for e in results
             if e["model"] in requested and not e.get("error")
             and (e.get("metrics") or {}).get("mase") is not None
             and (e.get("backtest") or {}).get("ds")]
    cands.sort(key=lambda e: e["metrics"]["mase"])
    if len(cands) < 2:
        return None

    excluded = {}
    best_mase = float(cands[0]["metrics"]["mase"])
    if ecfg["exclude_worse_than_baseline"]:
        base = next((e for e in results if e["model"] == BASELINE), None)
        bm = ((base or {}).get("metrics") or {}).get("mase")
        if bm is not None and np.isfinite(bm):
            keep = []
            for e in cands:
                if float(e["metrics"]["mase"]) >= float(bm) and e["model"] != BASELINE:
                    excluded[e["model"]] = (f"MASE {e['metrics']['mase']:.3f} not better "
                                            f"than baseline {bm:.3f}")
                else:
                    keep.append(e)
            cands = keep
    # Factor exclusion never cuts below two members — the old top-2 behavior is the
    # floor (a mediocre second member just gets a low weight from the optimizer).
    factor = float(ecfg["mase_factor"])
    keep = []
    for i, e in enumerate(cands):
        if i >= 2 and float(e["metrics"]["mase"]) > factor * best_mase:
            excluded[e["model"]] = (f"MASE {e['metrics']['mase']:.3f} > "
                                    f"{factor:g}x best ({best_mase:.3f})")
        else:
            keep.append(e)
    cands = keep
    for e in cands[int(ecfg["max_members"]):]:
        excluded[e["model"]] = "beyond max_members cap"
    cands = cands[:int(ecfg["max_members"])]
    if len(cands) < 2:
        return None

    # Per-series backtest lookup: model -> {(unique_id, ds): (actual, predicted)}, built
    # from the full-resolution rows (same source as series_forecasts.parquet), not the
    # panel-summed display arrays. Only meaningful on a per_series run.
    per_series_bt = None
    if scope == "per_series" and series_frames:
        bt_all = pd.concat(series_frames, ignore_index=True)
        bt_all = bt_all[bt_all["kind"] == "backtest"]
        per_series_bt = {
            mid: {(uid, ds): (a, p) for uid, ds, a, p
                  in zip(g["unique_id"], g["ds"], g["actual"], g["predicted"])
                  if pd.notna(p)}
            for mid, g in bt_all.groupby("model", observed=True)
        }

    def _bt_map(e):
        if per_series_bt is not None and e["model"] in per_series_bt:
            return {k: v[1] for k, v in per_series_bt[e["model"]].items()}
        return {ds: p for ds, p in zip(e["backtest"]["ds"], e["backtest"]["predicted"])
                if p is not None}

    def _shared(members):
        maps = [_bt_map(e) for e in members]
        shared = set(maps[0])
        for m in maps[1:]:
            shared &= set(m)
        return shared, maps

    members = list(cands)
    shared, maps = _shared(members)
    min_pts = int(ecfg["min_shared_points"])

    def _need(k):
        return max(min_pts, int(ecfg["min_points_per_member"]) * k)

    # Prune until the shared window supports the member count: drop the non-best member
    # whose removal grows the intersection most (>= keeps the later, worse-MASE member
    # as the drop on ties). Two members are always allowed min_shared_points, matching
    # the old top-2 behavior — shrinkage handles the small-sample risk there.
    while len(shared) < _need(len(members)) and len(members) > 2:
        best_gain, drop_i = -1, len(members) - 1
        for i in range(1, len(members)):
            s, _ = _shared(members[:i] + members[i + 1:])
            if len(s) >= best_gain:
                best_gain, drop_i = len(s), i
        excluded[members[drop_i]["model"]] = "insufficient shared backtest window"
        members.pop(drop_i)
        shared, maps = _shared(members)
    if len(members) < 2 or len(shared) < min_pts:
        return None

    a = members[0]
    if per_series_bt is not None and a["model"] in per_series_bt:
        act_map = {k: v[0] for k, v in per_series_bt[a["model"]].items() if pd.notna(v[0])}
    else:
        act_map = {ds: v for ds, v in zip(a["backtest"]["ds"], a["backtest"]["actual"])
                   if v is not None}
    # Tuple keys (per-series) sort by unique_id then ds — fine, order only needs to be
    # consistent between act/P, not chronological (MAE is order-invariant). Plain ISO
    # date-string keys (aggregate) sort chronologically as before.
    ds_c = [k for k in sorted(shared) if k in act_map]
    if len(ds_c) < min_pts:
        return None
    act = np.array([float(act_map[k]) for k in ds_c])
    P = np.column_stack([np.array([float(m[k]) for k in ds_c]) for m in maps])
    mases = [float(e["metrics"]["mase"]) for e in members]

    w, method = _solve_weights(act, P, mases, ecfg)
    pred = P @ w
    metrics = evaluator.compute_metrics(pd.Series(act), pd.Series(pred), scale)

    # Forecast: all members share the final horizon grid, so the inner join is
    # normally the full horizon; weighted mean with the same weights.
    fmaps = [{ds: v for ds, v in zip(e["forecast"]["ds"], e["forecast"]["yhat"])
              if v is not None} for e in members]
    f_shared = set(fmaps[0])
    for m in fmaps[1:]:
        f_shared &= set(m)
    f_ds = sorted(f_shared)
    point = (np.column_stack([np.array([float(m[ds]) for ds in f_ds]) for m in fmaps]) @ w
             if f_ds else np.array([]))
    resid = act - pred
    res_by_step = {k + 1: resid for k in range(len(point))}
    intervals = conformal_intervals(point, res_by_step, levels) if len(point) else {}
    forecast = {"ds": f_ds, "yhat": _round_list(point), "interval_method": "conformal"}
    for lvl in levels:
        if lvl in intervals:
            lo, hi = intervals[lvl]
            forecast[f"lo{lvl}"] = _round_list(lo)
            forecast[f"hi{lvl}"] = _round_list(hi)

    weights = [round(float(x), 4) for x in w]
    label_parts = [f"{w[i]:.2f}·{members[i]['label']}" for i in range(len(members))
                   if round(float(w[i]), 2) > 0]

    # Chart display: panel-sum the matched per-series points by ds, matching every other
    # row's `backtest` field convention (_display_backtest's per_series branch). Metrics
    # above stayed at native per-series resolution; only this display curve is aggregated.
    if per_series_bt is not None:
        disp = pd.DataFrame({"ds": [k[1] for k in ds_c], "actual": act, "predicted": pred})
        g = disp.groupby("ds", as_index=False)[["actual", "predicted"]].sum().sort_values("ds")
        backtest_out = {"ds": [str(d) for d in g["ds"]], "actual": _round_list(g["actual"]),
                        "predicted": _round_list(g["predicted"])}
    else:
        backtest_out = {"ds": ds_c, "actual": _round_list(act), "predicted": _round_list(pred)}

    return {
        "model": ENSEMBLE, "label": f"Ensemble ({' + '.join(label_parts)})",
        "category": "ensemble", "requested": True, "is_baseline": False,
        "is_ensemble": True, "ensemble_of": [e["model"] for e in members],
        "metrics": metrics,
        "params": {"components": [e["model"] for e in members], "weights": weights,
                   "weight_method": method, "n_shared_points": len(ds_c),
                   "loss": "mae", "excluded": excluded},
        "tuning": None,
        "strategy": f"weighted mean of {len(members)} requested forecasts ({method} weights)",
        "error": None, "train_seconds": 0.0,
        "backtest": backtest_out,
        "forecast": forecast,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def run(run_id: str, models, confirm_heavy: bool = False, profile: str | None = None) -> dict:
    run_t0 = time.perf_counter()
    cfg = _load_config()
    tcfg = cfg["training"]
    run_dir = RUNS_DIR / run_id
    report_path = run_dir / "feature_report.json"
    if not report_path.exists():
        raise FileNotFoundError("Stage 6 feature report missing — run feature engineering first.")
    freport = json.loads(report_path.read_text())
    recipe = freport["recipe"]
    scope = freport["scope"]
    policy = freport["policy"]
    horizon = int(freport["horizon"])
    freq_label = freport["frequency"]
    period = int(recipe.get("primary_period") or 1)

    msel = json.loads((run_dir / "model_selection.json").read_text()) \
        if (run_dir / "model_selection.json").exists() else {}
    eligible = {m["id"]: m for m in (msel.get("eligible_models") or [])}
    label_map = {mid: m.get("label", mid) for mid, m in eligible.items()}
    cat_map = {mid: m.get("category") for mid, m in eligible.items()}
    suggested = msel.get("model")

    frame = pd.read_parquet(FEATURES_DIR / f"{run_id}_model_frame.parquet")
    frame["ds"] = pd.to_datetime(frame["ds"])
    exog_cols = [c for c in frame.columns if c not in ("unique_id", "ds", "y")]
    alias = _pandas_freq(freq_label, pd.Series(frame["ds"]))

    # Model set: requested ∩ eligible ∩ implemented; dedup; cap; always add the baseline.
    def _is_eligible(mid):
        m = eligible.get(mid)
        return (m is None) or (m.get("available", True) and not m.get("excluded_reason"))
    requested = [m for m in (models or [])
                 if m in registry.KNOWN_IDS and _is_eligible(m)]
    requested = list(dict.fromkeys(requested))
    warnings_list = []
    max_models = int(tcfg.get("max_models_per_run", 4))
    if len(requested) > max_models:
        warnings_list.append(f"capped to {max_models} models (requested {len(requested)})")
        requested = requested[:max_models]
    if not requested:
        requested = [suggested] if (suggested and suggested in registry.KNOWN_IDS) else [BASELINE]
    run_ids = list(dict.fromkeys([*requested, BASELINE]))

    # Series set + per-series fan-out cap.
    all_series = frame["unique_id"].unique().tolist()
    n_series = len(all_series)
    series_ids, reduced_from = all_series, None
    per_series_cap = int(tcfg.get("per_series_cap", 200))
    if scope == "per_series" and n_series > per_series_cap:
        # Applies under EVERY policy: even with a global policy, non-global-capable
        # models (prophet/theta/arima/... and the baseline) fan out per series — an
        # uncapped prophet once ran 1,782 Stan fits for hours. Global-routed models
        # read the full frame and are unaffected by series_ids.
        vol = frame.groupby("unique_id", observed=True)["y"].sum().sort_values(ascending=False)
        series_ids = vol.head(per_series_cap).index.tolist()
        reduced_from = n_series
        warnings_list.append(
            f"per-series model fan-out capped to top {per_series_cap} of {n_series} "
            "series by volume (global models still train on all series)")

    # Cost gate.
    med_len = int(frame.groupby("unique_id", observed=True).size().median() or 0)
    estimate = estimate_cost(n_series, policy, requested, cfg, scope=scope,
                             length=med_len, horizon=horizon, period=period,
                             profile=profile)
    if estimate["tier"] == "heavy" and not confirm_heavy:
        raise TrainingHeavyError(estimate)

    cv_plan = _cv_plan(int(frame.groupby("unique_id").size().median()) if scope == "per_series"
                       else len(frame), period, horizon, cfg, n_series)
    levels = list(tcfg.get("interval_levels", [80, 95]))
    scale = _pooled_mase_scale(frame, period)
    profile = _resolve_profile(tcfg, override=profile)
    n_trials = (tcfg.get("optuna_trials_large", 8) if n_series >= tcfg.get("large_panel_series", 500)
                else int(profile.get("optuna_trials", 30)))

    # Per-model progress for the UI rail (estimate gives each model's ETA up front).
    eta_by_model = {p["model"]: p["seconds"] for p in estimate.get("per_model") or []}

    def _write_checkpoint():
        """Best-effort interim training_report.json after each model — if the worker dies
        partway through the loop (a bad wheel, a native crash, a preemption), every model
        that already finished keeps its full metrics/backtest/forecast instead of vanishing
        with the run (2026-07-20: a laptop prophet DLL crash lost an already-finished
        patchtst result because only the end-of-loop write existed). Never raises."""
        try:
            partial = {
                "run_id": run_id, "scope": scope, "policy": policy, "frequency": freq_label,
                "horizon": horizon, "season_length": period,
                "transform": recipe.get("transform", "none"), "metric": "MASE",
                "n_series": n_series, "series_trained": len(series_ids),
                "reduced_from": reduced_from, "baseline": BASELINE,
                "requested_models": requested, "suggested_model": suggested,
                "accuracy_profile": profile.get("profile_name"),
                "cost": estimate, "cv": cv_plan, "results": results,
                "warnings": warnings_list, "status": "training",
                "total_seconds": round(time.perf_counter() - run_t0, 2),
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            (run_dir / "training_report.json").write_text(json.dumps(partial, default=str))
        except Exception:  # noqa: BLE001 — a checkpoint write must never fail training
            pass

    def _backup_model(mid, fitted):
        """Best-effort per-model artifact backup the moment a model finishes — independent
        of which model eventually wins the crown, so a later model's failure (or a later
        step in run()) can't take an already-trained model's weights down with it."""
        if fitted is None:
            return
        try:
            out_dir = MODELS_DIR / run_id / "checkpoints"
            out_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(fitted, out_dir / f"{mid}.pkl")
            joblib.dump(recipe, out_dir / "preprocessor.pkl")
        except Exception:  # noqa: BLE001 — a checkpoint backup must never fail training
            pass

    results, fitted_objs, series_frames = [], {}, []
    for model_i, mid in enumerate(run_ids, start=1):
        progress.model_event(run_id, mid, model_i, len(run_ids), "running",
                             eta_seconds=eta_by_model.get(mid))
        # Ambient context for deep sub-loop heartbeats (torch epoch progress) — the label
        # prefixes each heartbeat with the model id in the live log.
        progress.set_train_context(run_id, "train", label=mid)
        model_t0 = time.perf_counter()
        try:
            params, tuning, native_iv = None, None, None
            # nhits/chronos are ALWAYS global on a panel (one network / one batched
            # inference pass); ML goes global only when Stage 5's policy says so.
            route_global = scope == "per_series" and registry.is_global_capable(mid) \
                and (policy == "global" or registry.always_global(mid))
            if route_global:
                if mid in registry.NHITS_IDS:
                    from models_lib import nhits_model  # lazy: torch is a heavy import
                    res = nhits_model.run_global(frame, recipe, cfg, horizon, alias,
                                                 recipe.get("transform", "none"),
                                                 levels, profile)
                elif mid in registry.PATCHTST_IDS:
                    from models_lib import patchtst_model  # lazy: torch is a heavy import
                    res = patchtst_model.run_global(frame, recipe, cfg, horizon, alias,
                                                    recipe.get("transform", "none"),
                                                    levels, profile)
                elif mid in registry.CHRONOS_IDS:
                    from models_lib import chronos_model  # lazy: transformers is heavy
                    res = chronos_model.run_global(frame, recipe, cfg, horizon, alias,
                                                   recipe.get("transform", "none"),
                                                   levels, profile)
                else:
                    res = ml_models.run_global(frame, recipe, mid, cfg, horizon, alias,
                                               recipe.get("transform", "none"),
                                               levels=levels, train_cfg=profile)
                if res.get("error"):
                    raise RuntimeError(res["error"])
                cv_df, fc_df = res["cv"], res["forecast"]
                fitted, strategy, params, tuning = res.get("_fitted"), res["strategy"], res["params"], res["tuning"]
            else:
                if registry.is_ml(mid) and scope != "per_series":
                    agg_df = frame[frame["unique_id"] == series_ids[0]].sort_values("ds").reset_index(drop=True)
                    params, tuning = ml_models.tune(mid, agg_df, recipe, n_trials, profile)
                make_fn = (lambda mid=mid, params=params: registry.make_forecaster(
                    mid, recipe, exog_cols, cfg["sarima_max_period"], params, freq_label,
                    train_cfg=profile))
                cv_df, fc_df, fitted, native_iv = _run_per_series(
                    frame, series_ids, make_fn, period, cv_plan, horizon, alias, exog_cols,
                    levels)
                strategy = cv_plan["strategy"]

            metrics = (evaluator.compute_metrics(cv_df["actual"], cv_df["predicted"], scale)
                       if cv_df is not None and not cv_df.empty
                       else {"mae": None, "rmse": None, "mape": None, "smape": None,
                             "mase": None, "r2": None, "n": 0})
            fitted_objs[mid] = fitted
            _backup_model(mid, fitted)
            if scope == "per_series":
                _collect_series_frames(series_frames, mid, cv_df, fc_df)
            results.append({
                "model": mid, "label": label_map.get(mid, mid), "category": cat_map.get(mid),
                "requested": mid in requested, "is_baseline": mid == BASELINE and mid not in requested,
                "metrics": metrics, "params": _json_safe_params(params), "tuning": tuning,
                "strategy": strategy, "error": None,
                "train_seconds": round(time.perf_counter() - model_t0, 2),
                "backtest": _display_backtest(cv_df, scope),
                "forecast": _display_forecast(fc_df, scope, cv_df, levels, native_iv),
            })
            progress.model_event(run_id, mid, model_i, len(run_ids), "done",
                                 seconds=time.perf_counter() - model_t0,
                                 mase=metrics.get("mase"))
            _write_checkpoint()
        except Exception as exc:  # noqa: BLE001 — isolate per-model failure
            progress.model_event(run_id, mid, model_i, len(run_ids), "failed",
                                 seconds=time.perf_counter() - model_t0)
            results.append({
                "model": mid, "label": label_map.get(mid, mid), "category": cat_map.get(mid),
                "requested": mid in requested, "is_baseline": mid == BASELINE and mid not in requested,
                "metrics": {"mase": None}, "params": {}, "tuning": None, "strategy": "none",
                "error": f"{type(exc).__name__}: {exc}",
                "train_seconds": round(time.perf_counter() - model_t0, 2),
                "backtest": {"ds": [], "actual": [], "predicted": []},
                "forecast": {"ds": [], "yhat": []},
            })
            _write_checkpoint()

    progress.clear_train_context()  # heartbeats off — the model loop is done

    # Weighted ensemble row (free: components already trained). Derived exclusively from
    # REQUESTED models, so it respects the "user's selection is ultimate" rule and is
    # itself crownable. Config: training.ensemble (legacy ensemble_top2 bool honored).
    ecfg = _ensemble_cfg(tcfg)
    if ecfg["enabled"] and len(requested) >= 2:
        ens = _ensemble(results, requested, scale, levels, ecfg,
                        scope=scope, series_frames=series_frames)
        if ens:
            results.append(ens)
            fitted_objs[ENSEMBLE] = {"ensemble_of": ens["ensemble_of"],
                                     "weights": ens["params"]["weights"],
                                     "components": {m: fitted_objs.get(m)
                                                    for m in ens["ensemble_of"]}}
            m_txt = ens["metrics"].get("mase")
            progress.stage_update(run_id, "train", f"ensemble: {ens['label']}"
                                  + (f" — MASE {m_txt}" if m_txt is not None else ""))
            if scope == "per_series" and series_frames:
                # Per-series viewer rows: weighted mean where ALL members cover the
                # (unique_id, ds, kind). dropna first is load-bearing — a NaN member
                # prediction would otherwise be skipped by the sum while n still counts
                # the model, silently mis-weighting the point instead of dropping it.
                wmap = dict(zip(ens["ensemble_of"], ens["params"]["weights"]))
                sf_all = pd.concat(series_frames, ignore_index=True)
                comp = sf_all[sf_all["model"].isin(wmap)].dropna(subset=["predicted"]).copy()
                comp["_w"] = comp["model"].map(wmap).astype(float)
                comp["_wp"] = comp["predicted"] * comp["_w"]
                g = comp.groupby(["unique_id", "ds", "kind"], as_index=False, observed=True) \
                    .agg(actual=("actual", "first"), _wp=("_wp", "sum"),
                         _ws=("_w", "sum"), n=("model", "nunique"))
                g = g[(g["n"] == len(wmap)) & (g["_ws"] > 0)].copy()
                g["predicted"] = g["_wp"] / g["_ws"]
                g["model"] = ENSEMBLE
                series_frames.append(
                    g[["model", "unique_id", "ds", "kind", "actual", "predicted"]])
        else:
            progress.stage_update(run_id, "train",
                                  "ensemble skipped — not enough comparable backtests "
                                  "among the requested models")

    # Rank by MASE (finite first), mark primary + best. The user's dropdown selection is
    # the ULTIMATE choice: only a REQUESTED model (or the ensemble derived from requested
    # models) can be crowned/serialized — the baseline is a reference row in the
    # leaderboard, never the winner (unless explicitly selected).
    def _mase(e):
        v = (e.get("metrics") or {}).get("mase")
        return v if v is not None else float("inf")
    results.sort(key=_mase)
    primary_model = requested[0]
    best_model = next((e["model"] for e in results
                       if (e["model"] in requested or e.get("is_ensemble"))
                       and np.isfinite(_mase(e))), primary_model)
    for e in results:
        e["is_primary"] = e["model"] == primary_model
        e["is_best"] = e["model"] == best_model

    # Target-level insights for the UI (next period, horizon total vs trailing history,
    # widest 95% band). Trailing history comes from the EXACT modeling frame — the store's
    # plot arrays are downsampled, so summing them would be silently wrong on long series.
    hist = (frame.groupby("ds")["y"].sum() if scope == "per_series"
            else frame.set_index("ds")["y"]).sort_index()
    for e in results:
        if not e.get("error"):
            e["target_insights"] = _target_insights(e, hist, horizon)

    # Serialize the best model (+ recipe as the preprocessor).
    serialized = _serialize(run_id, best_model, fitted_objs.get(best_model), recipe)

    # Per-series backtest + forecast rows → parquet, so the UI can chart any single
    # series (e.g. one taluka) on demand via /runs/{id}/series-forecast without bloating
    # the JSON report / browser store.
    series_ids_out, has_series_fc = [], False
    if series_frames:
        try:
            sf = pd.concat(series_frames, ignore_index=True)
            out_dir = MODELS_DIR / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            sf.to_parquet(out_dir / "series_forecasts.parquet", index=False)
            series_ids_out = sorted(sf["unique_id"].astype(str).unique().tolist())
            has_series_fc = True
        except Exception as exc:  # noqa: BLE001 — viewer is optional, never fail training
            warnings_list.append(f"per-series forecast parquet not written: {exc}")

    training_report = {
        "run_id": run_id, "scope": scope, "policy": policy, "frequency": freq_label,
        "horizon": horizon, "season_length": period, "transform": recipe.get("transform", "none"),
        "metric": "MASE", "n_series": n_series, "series_trained": len(series_ids),
        "reduced_from": reduced_from, "baseline": BASELINE,
        "primary_model": primary_model, "best_model": best_model,
        "suggested_model": suggested, "requested_models": requested,
        "accuracy_profile": profile.get("profile_name"),
        "has_series_forecasts": has_series_fc, "series_ids": series_ids_out,
        "cost": estimate, "cv": cv_plan, "results": results, "status": "done",
        "serialized": serialized, "warnings": warnings_list,
        "total_seconds": round(time.perf_counter() - run_t0, 2),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / "training_report.json").write_text(json.dumps(training_report, indent=2, default=str))
    # Feed observed wall-clock back into the estimator's calibration store so the next
    # cost estimate on this machine starts from reality instead of priors.
    record_timings(estimate, results)
    return {"run_id": run_id, "status": "completed", "report": training_report}


def _serialize(run_id, model_id, fitted, recipe) -> dict:
    """Best-effort: pickle the winning model + the recipe (preprocessor). Non-fatal."""
    try:
        if fitted is None:
            return {"saved": False, "reason": "no fitted object"}
        out_dir = MODELS_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted, out_dir / "best_model.pkl")
        joblib.dump(recipe, out_dir / "preprocessor.pkl")
        return {"saved": True, "model": model_id,
                "path": str((out_dir / "best_model.pkl").relative_to(ROOT))}
    except Exception as exc:  # noqa: BLE001
        return {"saved": False, "reason": f"{type(exc).__name__}: {exc}"}
