"""Stage 6: Feature Engineering — deterministic, EDA-seeded. NO LLM.

Two products, both derived entirely from Stage 4 evidence (never the LLM):

1. The **modeling frame** every model trains on — long format ``unique_id, ds, y[, exog]``.
   For aggregate scope this is the *identical* series Stage 4/5 analyzed (we reuse
   ``forecasting_eda._build_series``); for per-series scope it is the per-series panel
   built with one vectorized groupby-resample (same grid rules as Stage 4).

2. An **EDA-seeded feature recipe** (+ a materialized matrix for the ML models). The recipe
   is read straight off ``forecasting_eda_full.json``: significant ACF lags → lag features,
   primary seasonal period → rolling windows + Fourier terms, ndiffs/nsdiffs → momentum
   (past-difference) features, the heteroskedastic log recommendation → target transform,
   and exogenous cross-correlation lags → leading-driver features. Only lightgbm/xgboost
   consume the matrix; the statsforecast/classical models train on the raw ``y`` series.

The feature builders (``build_supervised`` etc.) are shared with ``models_lib/ml_models.py``
so the SAME transformation drives both training and the recursive multi-step forecast — the
only way to keep them leakage-free and consistent. Every engineered feature is computable
from PAST values only (lags/rolling/diffs all operate on shifted ``y``), so a row whose own
target is still unknown (the forecast row) is fully featurizable.

Reads:
  data/cleaned/{id}_cleaned.parquet
  runs/{id}/forecast_user_selections.json   (via forecasting_eda._resolve_selections)
  runs/{id}/forecasting_eda_full.json        (EDA evidence that seeds the recipe)
  runs/{id}/model_selection.json             (training_hints: transform/seasonal_periods/policy)

Writes:
  data/features/{id}_model_frame.parquet     (long: unique_id, ds, y[, exog]) — universal input
  data/features/{id}_features.parquet         (ML feature matrix — only when build_ml_matrix)
  runs/{id}/feature_report.json               (the EDA-seeded recipe + rationale)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.forecasting_eda import (
    _build_series, _f, _pandas_freq, _resolve_selections, apply_train_filter,
)

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
CLEANED_DIR = ROOT / "data" / "cleaned"
FEATURES_DIR = ROOT / "data" / "features"
CONFIG_PATH = ROOT / "config" / "settings.yaml"

# Which calendar parts are meaningful at each analysis frequency (no hour-of-day for
# monthly data, no month for yearly, etc). `year` is always included as a trend proxy.
_CALENDAR_BY_FREQ = {
    "hourly": ["year", "month", "day", "dayofweek", "hour", "is_weekend"],
    "daily": ["year", "month", "day", "dayofweek", "weekofyear", "quarter",
              "is_weekend", "is_month_start", "is_month_end"],
    "weekly": ["year", "month", "weekofyear", "quarter"],
    "monthly": ["year", "month", "quarter"],
    "quarterly": ["year", "quarter"],
    "yearly": ["year"],
}


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    return {"fe": cfg.get("feature_engineering", {}) or {},
            "training": cfg.get("training", {}) or {}}


def _paths(run_id: str) -> tuple[Path, Path]:
    run_dir = RUNS_DIR / run_id
    parquet = CLEANED_DIR / f"{run_id}_cleaned.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"Cleaned parquet not found: {parquet}. Run the Cleaning stage first.")
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}.")
    return run_dir, parquet


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


# ── EDA-seeded recipe ─────────────────────────────────────────────────────────

def _build_recipe(eda: dict, hints: dict, selections: dict, n_points: int, cfg: dict,
                  exog_cols: list[str]) -> dict:
    """The deterministic feature recipe, seeded from the Stage 4 EDA output + Stage 5
    training hints. Every choice traces back to a computed statistic — no free parameters
    beyond the caps in settings.yaml. `exog_cols` = the columns actually available in the
    modeling frame (the user-confirmed exogenous set); a leading driver Stage 5 liked but
    the user didn't include can't be used, so it's dropped from the recipe."""
    fe = cfg["fe"]
    agg = eda.get("aggregate") or {}
    freq = selections["forecast_frequency"]

    periods = [int(p) for p in (hints.get("seasonal_periods") or []) if p and p > 1]
    primary_period = periods[0] if periods else (
        (agg.get("seasonality") or {}).get("primary_period") or 1)
    primary_period = int(primary_period) if primary_period and primary_period > 1 else None

    # Lags: the significant ACF lags Stage 4 found, plus lag-1 and the seasonal lag, kept
    # short enough to leave training rows after the warmup, and capped.
    sig = list((agg.get("autocorrelation") or {}).get("significant_lags") or [])
    lag_set = {int(k) for k in sig if 1 <= int(k) < max(n_points - 2, 2)}
    lag_set.add(1)
    if primary_period and primary_period < n_points - 2:
        lag_set.add(primary_period)
    max_lags = int(fe.get("max_lags", 12) or 12)
    lags = sorted(lag_set)[:max_lags]

    # Rolling windows: the primary period (a full seasonal window) + any configured extras.
    windows = []
    if primary_period and primary_period >= 2:
        windows.append(primary_period)
    for w in fe.get("rolling_windows_extra") or []:
        if 2 <= int(w) < n_points and int(w) not in windows:
            windows.append(int(w))

    # Momentum (past-difference) features when Stage 4 said the series is non-stationary.
    st = agg.get("stationarity") or {}
    diffs = {
        "first": bool((st.get("ndiffs") or 0) >= 1),
        "seasonal": bool((st.get("nsdiffs") or 0) >= 1 and primary_period and primary_period > 1),
    }

    fourier = {int(p): int(fe.get("fourier_order", 3) or 3)
               for p in ([primary_period, *periods] if primary_period else periods)
               if p and p > 1 and n_points >= 2 * p}

    calendar = list(_CALENDAR_BY_FREQ.get(freq, ["year", "month"]))

    # Exogenous leading-driver lags (Stage 5's usable list — only drivers that LEAD, and
    # only those the user actually included in the modeling frame).
    exog_set = set(exog_cols or [])
    exog_lags = []
    for e in hints.get("exog_usable") or []:
        col, lag = e.get("col"), int(e.get("best_lag") or 0)
        if col in exog_set and lag >= 1:
            exog_lags.append({"col": col, "lag": lag})

    transform = "log" if (hints.get("transform") == "log") else "none"

    return {
        "target_col": selections["target_col"],
        "agg": selections["agg"],
        "frequency": freq,
        "horizon": selections["horizon"],
        "primary_period": primary_period,
        "seasonal_periods": periods or ([primary_period] if primary_period else []),
        "lags": lags,
        "rolling_windows": windows,
        "diffs": diffs,
        "fourier": fourier,
        "calendar": calendar,
        "exog_lags": exog_lags,
        "transform": transform,
        "rationale": {
            "lags": f"target at significant ACF lags {lags} identified in Stage 4",
            "rolling": (f"rolling mean/std over the {primary_period}-period seasonal window"
                        if windows else "no rolling window (no seasonal period)"),
            "calendar": f"calendar parts meaningful at {freq} frequency",
            "fourier": (f"sin/cos harmonics for seasonal period(s) {list(fourier)}"
                        if fourier else "no Fourier terms"),
            "diffs": ("first/seasonal momentum features (Stage 4 flagged non-stationary)"
                      if (diffs["first"] or diffs["seasonal"]) else "series is stationary — no diffs"),
            "exog": (f"leading exogenous drivers at their cross-correlation lags: "
                     f"{[(e['col'], e['lag']) for e in exog_lags]}" if exog_lags
                     else "no usable leading exogenous drivers"),
            "transform": ("log1p(target) — Stage 4 found level-dependent variance"
                          if transform == "log" else "no transform (variance is stable)"),
        },
    }


# ── Shared feature builders (used at train AND recursive-forecast time) ────────

def add_calendar_features(df: pd.DataFrame, ds_col: str, feats: list[str]) -> pd.DataFrame:
    """Deterministic datetime parts. Leakage-free (a timestamp is known for any row)."""
    ts = pd.to_datetime(df[ds_col])
    out = {}
    for f in feats:
        if f == "year":
            out["cal_year"] = ts.dt.year
        elif f == "month":
            out["cal_month"] = ts.dt.month
        elif f == "day":
            out["cal_day"] = ts.dt.day
        elif f == "dayofweek":
            out["cal_dow"] = ts.dt.dayofweek
        elif f == "hour":
            out["cal_hour"] = ts.dt.hour
        elif f == "weekofyear":
            out["cal_week"] = ts.dt.isocalendar().week.astype("int64")
        elif f == "quarter":
            out["cal_quarter"] = ts.dt.quarter
        elif f == "is_weekend":
            out["cal_is_weekend"] = (ts.dt.dayofweek >= 5).astype("int64")
        elif f == "is_month_start":
            out["cal_is_month_start"] = ts.dt.is_month_start.astype("int64")
        elif f == "is_month_end":
            out["cal_is_month_end"] = ts.dt.is_month_end.astype("int64")
    for k, v in out.items():
        df[k] = np.asarray(v)
    return df


def add_fourier_terms(df: pd.DataFrame, pos: np.ndarray, fourier: dict) -> pd.DataFrame:
    """sin/cos harmonics of the position-within-series index for each seasonal period.
    Position (not calendar month) keeps recursion trivial: the forecast row's index just
    continues the count, so future Fourier values need no lookup."""
    for period, order in fourier.items():
        # JSON round-trips dict keys to strings ({12: 3} → {"12": 3}); coerce back to int
        # so the division is numeric whether the recipe came from memory or from disk.
        period = int(period)
        for k in range(1, int(order) + 1):
            ang = 2.0 * np.pi * k * pos / period
            df[f"fourier_sin_{period}_{k}"] = np.sin(ang)
            df[f"fourier_cos_{period}_{k}"] = np.cos(ang)
    return df


def build_supervised(frame: pd.DataFrame, recipe: dict, *, dropna: bool = True,
                     ycol: str = "y") -> pd.DataFrame:
    """Turn one regular series (columns: ds, y[, exog...], optional pos) into a supervised
    feature matrix. Every feature uses PAST values only, so the trailing row (whose own y
    may be NaN) is still fully featurized — exactly what recursive forecasting needs.

    Expects `frame` sorted by ds with a clean RangeIndex. Adds a `pos` positional index if
    absent. Returns the frame with feature columns; when `dropna`, warmup rows with any NaN
    lag/rolling feature are dropped. The chosen feature columns are recorded on
    ``df.attrs['feature_cols']``."""
    df = frame.copy().reset_index(drop=True)
    if "pos" not in df.columns:
        df["pos"] = np.arange(len(df), dtype=float)
    y = df[ycol].astype("float64")

    feat_cols: list[str] = []
    for k in recipe["lags"]:
        col = f"lag_{k}"
        df[col] = y.shift(k)
        feat_cols.append(col)
    for w in recipe["rolling_windows"]:
        shifted = y.shift(1)
        df[f"rollmean_{w}"] = shifted.rolling(w).mean()
        df[f"rollstd_{w}"] = shifted.rolling(w).std()
        feat_cols += [f"rollmean_{w}", f"rollstd_{w}"]
    if recipe["diffs"].get("first"):
        df["diff1"] = y.shift(1) - y.shift(2)
        feat_cols.append("diff1")
    if recipe["diffs"].get("seasonal") and recipe.get("primary_period"):
        m = int(recipe["primary_period"])
        df[f"sdiff_{m}"] = y.shift(1) - y.shift(1 + m)
        feat_cols.append(f"sdiff_{m}")

    add_calendar_features(df, "ds", recipe["calendar"])
    feat_cols += [c for c in df.columns if c.startswith("cal_")]

    before = set(df.columns)
    add_fourier_terms(df, df["pos"].to_numpy(dtype=float), recipe.get("fourier") or {})
    feat_cols += [c for c in df.columns if c.startswith("fourier_") and c not in before]

    for e in recipe.get("exog_lags") or []:
        col, lag = e["col"], int(e["lag"])
        if col in df.columns:
            name = f"exoglag_{_safe(col)}_{lag}"
            df[name] = df[col].shift(lag)
            feat_cols.append(name)

    if dropna:
        # Only drop rows missing a lag/rolling/diff/exog feature (warmup); calendar/fourier
        # are always present.
        warmup = [c for c in feat_cols
                  if c.startswith(("lag_", "rollmean_", "rollstd_", "diff1", "sdiff_", "exoglag_"))]
        if warmup:
            df = df.dropna(subset=warmup).reset_index(drop=True)
    df.attrs["feature_cols"] = feat_cols
    return df


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name))


# ── Modeling frame construction ───────────────────────────────────────────────

def _aggregate_frame(df, ts_col, target, exog_cols, agg, freq_label):
    """One aggregate series as a long frame (unique_id='aggregate'), identical to the
    series Stage 4/5 analyzed (reuses forecasting_eda._build_series)."""
    y, exog, report = _build_series(df, ts_col, target, exog_cols, agg, freq_label)
    frame = pd.DataFrame({"unique_id": "aggregate", "ds": y.index, "y": y.to_numpy()})
    for col in exog.columns:
        frame[col] = exog[col].to_numpy()
    return frame.reset_index(drop=True), report


def _build_panel_frame(df, ts_col, target, group_cols, exog_cols, agg, freq_label):
    """Per-series panel as a long frame. One vectorized groupby-resample onto each series'
    own regular grid; empty bins filled with the same rules as Stage 4 (count/nunique → 0,
    zero-inflated sum → 0, otherwise per-series time interpolation)."""
    alias = _pandas_freq(freq_label, df[ts_col])
    grouper = [*group_cols, pd.Grouper(key=ts_col, freq=alias)]
    if agg == "nunique":
        s = df.groupby(grouper, observed=True)[target].nunique()
    elif agg == "count":
        s = df.groupby(grouper, observed=True)[target].count()
    elif agg == "mean":
        s = df.groupby(grouper, observed=True)[target].mean()
    else:
        s = df.groupby(grouper, observed=True)[target].sum(min_count=1)
    long = s.reset_index().rename(columns={ts_col: "ds", target: "y"})
    long["unique_id"] = long[group_cols].astype(str).agg(" / ".join, axis=1)

    observed = long["y"].dropna()
    zero_pct = float((observed == 0).mean() * 100) if len(observed) else 0.0
    if agg in ("nunique", "count"):
        long["y"] = long["y"].fillna(0.0)
        fill_method = "count_zero"
    elif agg == "sum" and zero_pct > 10:
        long["y"] = long["y"].fillna(0.0)
        fill_method = "zero"
    else:
        long = long.sort_values(["unique_id", "ds"])
        long["y"] = (long.groupby("unique_id", observed=True)["y"]
                     .transform(lambda g: g.interpolate(limit_direction="both")))
        fill_method = "interpolate"

    if exog_cols:
        ex = df.groupby(grouper, observed=True)[exog_cols].mean().reset_index()
        ex = ex.rename(columns={ts_col: "ds"})
        ex["unique_id"] = ex[group_cols].astype(str).agg(" / ".join, axis=1)
        long = long.merge(ex[["unique_id", "ds", *exog_cols]], on=["unique_id", "ds"], how="left")
        long = long.sort_values(["unique_id", "ds"])
        for c in exog_cols:
            long[c] = (long.groupby("unique_id", observed=True)[c]
                       .transform(lambda g: g.interpolate(limit_direction="both")))

    keep = ["unique_id", "ds", "y", *exog_cols]
    long = long[keep].sort_values(["unique_id", "ds"]).reset_index(drop=True)
    n_series = int(long["unique_id"].nunique())
    lengths = long.groupby("unique_id", observed=True).size()
    report = {
        "grid_points": int(len(long)),
        "n_series": n_series,
        "series_length": {"min": int(lengths.min()), "median": int(lengths.median()),
                          "max": int(lengths.max())},
        "fill_method": fill_method,
        "zero_pct": _f(float((long["y"] == 0).mean() * 100), 2),
    }
    return long, report


def _materialize_ml_matrix(frame: pd.DataFrame, recipe: dict) -> tuple[pd.DataFrame, list[str]]:
    """Full-history supervised matrix for the ML models, per series (leakage-free lags).
    Snapshot for transparency + the 'Features used' UI; MLBackend rebuilds via
    build_supervised so its CV splits stay clean."""
    parts, feature_cols = [], []
    for uid, g in frame.groupby("unique_id", observed=True, sort=False):
        g = g.sort_values("ds").reset_index(drop=True)
        sup = build_supervised(g, recipe, dropna=True)
        feature_cols = sup.attrs.get("feature_cols") or feature_cols
        sup.insert(0, "series_id", uid)
        parts.append(sup)
    if not parts:
        return pd.DataFrame(), []
    return pd.concat(parts, ignore_index=True), feature_cols


# ── Entry point ───────────────────────────────────────────────────────────────

def run(run_id: str, build_ml_matrix: bool | None = None) -> dict:
    """Stage 6 entry point. `build_ml_matrix` materializes the ML feature parquet (the
    train task sets it True whenever a lightgbm/xgboost model is selected)."""
    cfg = _load_config()
    run_dir, parquet = _paths(run_id)
    selections = _resolve_selections(run_dir)
    eda = _load_json(run_dir / "forecasting_eda_full.json")
    msel = _load_json(run_dir / "model_selection.json")
    hints = msel.get("training_hints") or {}

    ts_col = selections["timestamp_col"]
    target = selections["target_col"]
    scope = selections["scope"]
    group_cols = selections["group_cols"] if scope == "per_series" else []
    exog_cols = [c for c in (selections["exog_cols"] or []) if c != target]
    agg = selections["agg"]
    freq_label = selections["forecast_frequency"]

    use_cols = list(dict.fromkeys([ts_col, target, *group_cols, *exog_cols]))
    df = pd.read_parquet(parquet, columns=[c for c in use_cols if c])
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.loc[df[ts_col].notna() & df[target].notna()]
    if df.empty:
        raise ValueError(f"No usable rows for target '{target}'.")
    # Same training-window cut as Stage 4 — the trained series must be identical to
    # the one Stages 4/5 analyzed and decided on.
    df, train_filter = apply_train_filter(df, ts_col, selections)
    if agg in ("sum", "mean") and not pd.api.types.is_numeric_dtype(df[target]):
        agg = "nunique"
        selections["agg"] = "nunique"

    # Exogenous columns must be numeric to be usable as drivers.
    exog_cols = [c for c in exog_cols
                 if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    if scope == "per_series" and group_cols:
        frame, frame_report = _build_panel_frame(
            df, ts_col, target, group_cols, exog_cols, agg, freq_label)
    else:
        frame, frame_report = _aggregate_frame(
            df, ts_col, target, exog_cols, agg, freq_label)
        frame_report["n_series"] = 1

    if len(frame) < 5:
        raise ValueError(f"Only {len(frame)} points at {freq_label} — too short to model.")

    # Recipe is seeded from the AGGREGATE EDA stats; n_points for the caps uses the median
    # per-series length in panel mode, else the aggregate length.
    if scope == "per_series":
        n_points = int(frame_report.get("series_length", {}).get("median") or 5)
    else:
        n_points = len(frame)
    recipe = _build_recipe(eda, hints, selections, max(n_points, 5), cfg, exog_cols)

    frame_path = FEATURES_DIR / f"{run_id}_model_frame.parquet"
    frame.to_parquet(frame_path, index=False)

    ml_matrix_path = None
    feature_names: list[str] = []
    if build_ml_matrix:
        matrix, feature_names = _materialize_ml_matrix(frame, recipe)
        ml_matrix_path = FEATURES_DIR / f"{run_id}_features.parquet"
        matrix.to_parquet(ml_matrix_path, index=False)

    report = {
        "run_id": run_id,
        "scope": scope,
        "policy": hints.get("policy") or ("per_series_local" if scope == "per_series" else "aggregate"),
        "frequency": freq_label,
        "horizon": selections["horizon"],
        "target": target,
        "agg": agg,
        "n_series": frame_report.get("n_series", 1),
        "train_filter": train_filter,
        "frame": frame_report,
        "recipe": recipe,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "model_frame_path": str(frame_path.relative_to(ROOT)),
        "ml_matrix_path": str(ml_matrix_path.relative_to(ROOT)) if ml_matrix_path else None,
    }
    (run_dir / "feature_report.json").write_text(json.dumps(report, indent=2, default=str))
    return {"run_id": run_id, "status": "completed", "report": report}
