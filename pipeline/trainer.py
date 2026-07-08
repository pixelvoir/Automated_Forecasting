"""Stage 7: Model Training — hardcoded, EDA-seeded, NO LLM.

Trains the user-selected models (Stage 5 is only a default suggestion), backtests each with
walk-forward cross-validation, ranks them by MASE against an always-included seasonal-naive
baseline, refits the winner on all history, produces a horizon forecast with conformal
prediction intervals, and serializes the chosen model. One CV harness drives every model
family (classical / intermittent / ML) through the uniform Forecaster interface.

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
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from models_lib import ml_models, registry
from models_lib.base_model import conformal_intervals, future_index
from pipeline import evaluator
from pipeline.forecasting_eda import _pandas_freq

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"
CONFIG_PATH = ROOT / "config" / "settings.yaml"

BASELINE = "seasonal_naive"


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


def _round_list(arr, nd=4):
    out = []
    for v in np.asarray(arr, dtype=float):
        out.append(round(float(v), nd) if np.isfinite(v) else None)
    return out


# ── Cost estimate ─────────────────────────────────────────────────────────────

def estimate_cost(n_series: int, policy: str, model_ids, cfg: dict) -> dict:
    """fast / moderate / heavy from policy × n_series × models. Per-series-local classical
    is the multiplier that matters; global ML counts as one fit."""
    t = cfg["training"]
    units = 0.0
    for mid in model_ids:
        if registry.is_ml(mid):
            units += 1.5 if policy == "global" else (n_series * 1.5 if policy == "per_series_local" else 1.5)
        else:
            units += n_series if policy == "per_series_local" else 1
    moderate = t.get("moderate_tier_units", 200)
    heavy = t.get("heavy_tier_units", 2000)
    tier = "heavy" if units > heavy else ("moderate" if units > moderate else "fast")
    rec = None
    if tier == "heavy" and policy == "per_series_local":
        rec = ("Consider the global LightGBM policy or reduce the per-series fan-out "
               f"(currently {n_series} series).")
    return {"tier": tier, "units": round(units, 1), "n_series": int(n_series),
            "policy": policy, "n_models": len(list(model_ids)), "recommendation": rec}


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


def _final_forecast(series_df, make_fn, period, horizon, freq_alias, exog_cols):
    model = make_fn().fit(series_df, period)
    fut_ds = future_index(series_df["ds"].iloc[-1], horizon, freq_alias)
    yhat = model.predict(horizon, fut_ds.to_numpy(), None)
    return fut_ds, np.asarray(yhat, dtype=float), model


def _run_per_series(frame, series_ids, make_fn, period, cv_plan, horizon, freq_alias, exog_cols):
    """CV + final forecast for each series. Per-series failures (a tiny/degenerate series in
    a panel) are isolated so one bad series can't wipe the whole model's results — for
    aggregate scope there's just one series, so a failure there surfaces to the caller."""
    cv_all, fc_parts, fitted, n_ok = [], [], None, 0
    for uid in series_ids:
        g = frame[frame["unique_id"] == uid].sort_values("ds").reset_index(drop=True)
        try:
            for r in _walk_forward(g, make_fn, period, cv_plan, exog_cols):
                r["unique_id"] = uid
                cv_all.append(r)
            fut_ds, yhat, model = _final_forecast(g, make_fn, period, horizon, freq_alias, exog_cols)
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
    return cv_df, fc_df, fitted


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


def _display_forecast(fc_df, scope, cv_df, levels):
    if fc_df is None or fc_df.empty:
        return {"ds": [], "yhat": []}
    if scope == "per_series":
        g = fc_df.groupby("ds", as_index=False)["yhat"].sum().sort_values("ds")
    else:
        g = fc_df.sort_values("ds")
    point = g["yhat"].to_numpy(dtype=float)

    res_by_step = {}
    if cv_df is not None and not cv_df.empty and "step" in cv_df.columns and scope != "per_series":
        for s, grp in cv_df.groupby("step"):
            res_by_step[int(s)] = (grp["actual"] - grp["predicted"]).to_numpy()
    elif cv_df is not None and not cv_df.empty:
        bt = _display_backtest(cv_df, scope)
        resid = np.asarray(bt["actual"], dtype=float) - np.asarray(bt["predicted"], dtype=float)
        res_by_step = {k + 1: resid for k in range(len(point))}
    intervals = conformal_intervals(point, res_by_step, levels) if len(point) else {}

    out = {"ds": [str(d) for d in g["ds"]], "yhat": _round_list(point)}
    for lvl in levels:
        if lvl in intervals:
            lo, hi = intervals[lvl]
            out[f"lo{lvl}"] = _round_list(lo)
            out[f"hi{lvl}"] = _round_list(hi)
    return out


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


# ── Entry point ───────────────────────────────────────────────────────────────

def run(run_id: str, models, confirm_heavy: bool = False) -> dict:
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
    if scope == "per_series" and policy != "global" and n_series > per_series_cap:
        vol = frame.groupby("unique_id", observed=True)["y"].sum().sort_values(ascending=False)
        series_ids = vol.head(per_series_cap).index.tolist()
        reduced_from = n_series
        warnings_list.append(
            f"per-series fan-out capped to top {per_series_cap} of {n_series} series by volume")

    # Cost gate.
    estimate = estimate_cost(n_series, policy, requested, cfg)
    if estimate["tier"] == "heavy" and not confirm_heavy:
        raise TrainingHeavyError(estimate)

    cv_plan = _cv_plan(int(frame.groupby("unique_id").size().median()) if scope == "per_series"
                       else len(frame), period, horizon, cfg, n_series)
    levels = list(tcfg.get("interval_levels", [80, 95]))
    scale = _pooled_mase_scale(frame, period)
    n_trials = (tcfg.get("optuna_trials_large", 8) if n_series >= tcfg.get("large_panel_series", 500)
                else tcfg.get("optuna_trials", 20))

    results, fitted_objs = [], {}
    for mid in run_ids:
        try:
            params, tuning = None, None
            if registry.is_ml(mid) and policy == "global" and scope == "per_series":
                res = ml_models.run_global(frame, recipe, mid, cfg, horizon, alias,
                                           recipe.get("transform", "none"))
                if res.get("error"):
                    raise RuntimeError(res["error"])
                cv_df, fc_df = res["cv"], res["forecast"]
                fitted, strategy, params, tuning = res.get("_fitted"), res["strategy"], res["params"], res["tuning"]
            else:
                if registry.is_ml(mid) and scope != "per_series":
                    agg_df = frame[frame["unique_id"] == series_ids[0]].sort_values("ds").reset_index(drop=True)
                    params, tuning = ml_models.tune(mid, agg_df, recipe, cfg, n_trials)
                make_fn = (lambda mid=mid, params=params: registry.make_forecaster(
                    mid, recipe, exog_cols, cfg["sarima_max_period"], params, freq_label))
                cv_df, fc_df, fitted = _run_per_series(
                    frame, series_ids, make_fn, period, cv_plan, horizon, alias, exog_cols)
                strategy = cv_plan["strategy"]

            metrics = (evaluator.compute_metrics(cv_df["actual"], cv_df["predicted"], scale)
                       if cv_df is not None and not cv_df.empty
                       else {"mae": None, "rmse": None, "mape": None, "smape": None,
                             "mase": None, "r2": None, "n": 0})
            fitted_objs[mid] = fitted
            results.append({
                "model": mid, "label": label_map.get(mid, mid), "category": cat_map.get(mid),
                "requested": mid in requested, "is_baseline": mid == BASELINE and mid not in requested,
                "metrics": metrics, "params": params or {}, "tuning": tuning, "strategy": strategy,
                "error": None,
                "backtest": _display_backtest(cv_df, scope),
                "forecast": _display_forecast(fc_df, scope, cv_df, levels),
            })
        except Exception as exc:  # noqa: BLE001 — isolate per-model failure
            results.append({
                "model": mid, "label": label_map.get(mid, mid), "category": cat_map.get(mid),
                "requested": mid in requested, "is_baseline": mid == BASELINE and mid not in requested,
                "metrics": {"mase": None}, "params": {}, "tuning": None, "strategy": "none",
                "error": f"{type(exc).__name__}: {exc}",
                "backtest": {"ds": [], "actual": [], "predicted": []},
                "forecast": {"ds": [], "yhat": []},
            })

    # Rank by MASE (finite first), mark primary + best.
    def _mase(e):
        v = (e.get("metrics") or {}).get("mase")
        return v if v is not None else float("inf")
    results.sort(key=_mase)
    primary_model = requested[0]
    best_model = next((e["model"] for e in results if np.isfinite(_mase(e))), primary_model)
    for e in results:
        e["is_primary"] = e["model"] == primary_model
        e["is_best"] = e["model"] == best_model

    # Serialize the best model (+ recipe as the preprocessor).
    serialized = _serialize(run_id, best_model, fitted_objs.get(best_model), recipe)

    training_report = {
        "run_id": run_id, "scope": scope, "policy": policy, "frequency": freq_label,
        "horizon": horizon, "season_length": period, "transform": recipe.get("transform", "none"),
        "metric": "MASE", "n_series": n_series, "series_trained": len(series_ids),
        "reduced_from": reduced_from, "baseline": BASELINE,
        "primary_model": primary_model, "best_model": best_model,
        "suggested_model": suggested, "requested_models": requested,
        "cost": estimate, "cv": cv_plan, "results": results,
        "serialized": serialized, "warnings": warnings_list,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / "training_report.json").write_text(json.dumps(training_report, indent=2, default=str))
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
