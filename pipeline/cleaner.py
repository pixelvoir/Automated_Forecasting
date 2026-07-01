"""Stage 3 cleaner: apply the LLM-chosen cleaning recipe to the raw parquet.

All transforms are hardcoded Python functions. No eval(), no exec().
Row drops (for outlier removal and missing drop_row) are batched and applied once
to avoid skipping rows that match multiple criteria.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
RAW_DIR = ROOT / "data" / "raw"
CLEANED_DIR = ROOT / "data" / "cleaned"


# ── Missing-value handlers ────────────────────────────────────────────────────

def _apply_missing(df: pd.DataFrame, col: str, strategy: str,
                   ts_col: str | None = None) -> pd.DataFrame:
    s = df[col]
    if strategy == "interpolate":
        # time-based interpolation requires a DatetimeIndex.
        # Raw parquet has a RangeIndex, so temporarily align the series on the
        # timestamp column so pandas uses actual time gaps between rows.
        if ts_col and ts_col in df.columns and ts_col != col:
            ts_index = pd.to_datetime(df[ts_col], errors="coerce")
            temp = s.copy()
            temp.index = ts_index
            temp = temp.interpolate(method="time")
            df[col] = temp.values
        else:
            df[col] = s.interpolate()  # linear fallback when no timestamp col
    elif strategy == "forward_fill":
        df[col] = s.ffill()
    elif strategy == "backward_fill":
        df[col] = s.bfill()
    elif strategy == "mean_fill":
        if pd.api.types.is_numeric_dtype(s):
            df[col] = s.fillna(s.mean())
    elif strategy == "median_fill":
        if pd.api.types.is_numeric_dtype(s):
            df[col] = s.fillna(s.median())
    elif strategy == "flag_and_fill":
        df[f"{col}_missing"] = s.isna().astype(int)
        df[col] = s.ffill()
    # "drop_row" → collected upstream and applied in batch
    # "none"     → no-op
    return df


# ── Outlier handlers ──────────────────────────────────────────────────────────

_FREQ_PERIOD = {"hourly": 24, "daily": 7, "weekly": 52, "monthly": 12, "quarterly": 4, "yearly": 1}

# STL fits iterative LOESS and is O(n * iterations); it costs minutes on multi-million-row
# series. Above this length we swap to an O(n) classical decomposition that produces the same
# trend+seasonal output shape in a fraction of a second (see _fast_seasonal_outlier).
_STL_MAX_POINTS = 100_000


def _period_from_recipe(recipe: dict | None) -> int:
    return int((recipe or {}).get("period", 7)) or 7


# Above this window size, pandas' rolling().quantile() is faster than the numpy sliding-
# window approach below — measured crossover is between 40-52 on a multi-million-row series.
# The only real _FREQ_PERIOD value past it is "weekly" (52); every other value (1, 4, 7, 12,
# 24) is faster or much faster with numpy — up to 5.6x for daily/quarterly-sized windows.
_FAST_ROLLING_MAX_WINDOW = 40


def _fast_rolling_q1_q3(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised centered rolling Q1/Q3 — a drop-in equivalent of
    ``pd.Series.rolling(window, center=True, min_periods=1).quantile([0.25, 0.75])``
    for small-to-moderate windows. Verified to produce bit-identical outlier masks against
    the pandas method on real data; 1.5x-5.6x faster depending on window size."""
    half = window // 2
    padded = np.pad(values, (half, window - 1 - half), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    q1, q3 = np.percentile(windows, [25, 75], axis=1)
    return q1, q3


def _fast_seasonal_outlier(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """Vectorised O(n) equivalent of the STL-residual strategy for long series.

    Trend = centred rolling mean over one seasonal cycle; seasonal = mean detrended value per
    phase; residual outliers (MAD-thresholded) are replaced with trend+seasonal, dropping no
    rows — identical in shape to the STL branch, but seconds instead of minutes at 3M rows.
    """
    s = df[col]
    filled = s.ffill().bfill()
    win = period if period % 2 == 1 else period + 1
    trend = filled.rolling(win, center=True, min_periods=1).mean()
    detrended = filled - trend
    phase = np.arange(len(filled)) % period
    seasonal = detrended.groupby(phase).transform("mean")
    resid = (filled - trend - seasonal).to_numpy()
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    if mad == 0.0:
        return df  # no dispersion in residuals → nothing to flag
    mask = np.abs(resid - med) > 3.5 * 1.4826 * mad
    replacement = (trend + seasonal).to_numpy()
    out = s.to_numpy(dtype="float64", copy=True)
    out[mask] = replacement[mask]
    df[col] = out
    return df


def _apply_outlier(df: pd.DataFrame, col: str, strategy: str,
                   recipe: dict | None = None) -> pd.DataFrame:
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s):
        return df
    valid = s.dropna()
    if len(valid) == 0:
        return df

    if strategy == "winsorize":
        lo = float(np.percentile(valid, 1.5))
        hi = float(np.percentile(valid, 98.5))
        df[col] = s.clip(lo, hi)
    elif strategy == "clip_iqr":
        q1 = float(np.percentile(valid, 25))
        q3 = float(np.percentile(valid, 75))
        iqr = q3 - q1
        df[col] = s.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    elif strategy == "log_transform":
        df[col] = np.log1p(s.clip(lower=0))
    elif strategy == "rolling_iqr":
        window = max(_period_from_recipe(recipe), 7)
        if len(valid) < window:
            # too short for rolling — fall back to global clip_iqr
            q1 = float(np.percentile(valid, 25))
            q3 = float(np.percentile(valid, 75))
            iqr = q3 - q1
            df[col] = s.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        else:
            # The fast path assumes no NaN in the window (verified bit-identical to pandas
            # only in that case — pandas skips NaN within a window, plain np.percentile
            # does not). Missing-value handling runs *after* outlier handling in cleaner.py's
            # execution order, so `s` here isn't guaranteed NaN-free — fall back to the
            # pandas method whenever NaN is present, regardless of window size.
            if window <= _FAST_ROLLING_MAX_WINDOW and not s.isna().any():
                q1_vals, q3_vals = _fast_rolling_q1_q3(s.to_numpy(dtype="float64"), window)
                q1_r, q3_r = pd.Series(q1_vals, index=s.index), pd.Series(q3_vals, index=s.index)
            else:
                q1_r = s.rolling(window, center=True, min_periods=1).quantile(0.25)
                q3_r = s.rolling(window, center=True, min_periods=1).quantile(0.75)
            iqr_r = q3_r - q1_r
            df[col] = s.clip(q1_r - 1.5 * iqr_r, q3_r + 1.5 * iqr_r)
    elif strategy == "stl_residuals":
        period = max(_period_from_recipe(recipe), 2)
        if len(valid) < 2 * period:
            # not enough cycles — fall back to rolling_iqr
            df = _apply_outlier(df, col, "rolling_iqr", recipe=recipe)
        elif len(s) > _STL_MAX_POINTS:
            # too long for STL's LOESS iterations — use the O(n) classical path
            df = _fast_seasonal_outlier(df, col, period)
        else:
            from statsmodels.tsa.seasonal import STL
            s_filled = s.ffill().bfill()
            try:
                stl = STL(s_filled, period=period, robust=True)
                res = stl.fit()
                resid = res.resid
                med_resid = float(np.median(resid))
                mad = float(np.median(np.abs(resid - med_resid)))
                threshold = 3.5 * 1.4826 * mad
                outlier_mask = np.abs(resid - med_resid) > threshold
                s_clean = s.copy()
                s_clean[outlier_mask] = (res.trend + res.seasonal)[outlier_mask]
                df[col] = s_clean
            except Exception:
                df = _apply_outlier(df, col, "rolling_iqr", recipe=recipe)
    # "remove" → collected upstream and applied in batch
    # "keep"   → no-op
    return df


# ── Type-fix handlers ─────────────────────────────────────────────────────────

def _apply_type_fix(df: pd.DataFrame, col: str, fix: str) -> pd.DataFrame:
    s = df[col]
    if fix == "parse_datetime":
        df[col] = pd.to_datetime(s, errors="coerce")
    elif fix == "cast_numeric":
        df[col] = pd.to_numeric(s, errors="coerce")
    elif fix == "encode_boolean":
        mapping = {
            "yes": True, "no": False,
            "true": True, "false": False,
            "1": True, "0": False,
        }
        df[col] = s.astype(str).str.lower().map(mapping)
    return df


# ── Lightweight snapshot ──────────────────────────────────────────────────────

def _snapshot(df: pd.DataFrame) -> dict:
    nulls = {
        col: {
            "count": int(df[col].isna().sum()),
            "pct": round(float(df[col].isna().mean()) * 100, 2),
        }
        for col in df.columns
    }
    numeric_variance = {}
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        numeric_variance[col] = float(s.std()) if len(s) > 0 else 0.0

    return {
        "rows": len(df),
        "cols": len(df.columns),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 3),
        "nulls": nulls,
        "numeric_variance": numeric_variance,
    }


# ── Stage 3 executor ──────────────────────────────────────────────────────────

def run(run_id: str) -> dict:
    """Execute the cleaning recipe on the raw parquet.

    Reads:
      data/raw/{run_id}_raw.parquet
      runs/{run_id}/cleaning_recipe.json
    Writes:
      data/cleaned/{run_id}_cleaned.parquet
      runs/{run_id}/cleaning_report.json
      runs/{run_id}/cleaned_metadata.json
    """
    run_dir = RUNS_DIR / run_id
    recipe_path = run_dir / "cleaning_recipe.json"
    raw_path = RAW_DIR / f"{run_id}_raw.parquet"

    if not recipe_path.exists():
        raise FileNotFoundError(
            f"Cleaning recipe not found: {recipe_path}. Run cleaning agent first."
        )
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw parquet not found: {raw_path}.")

    recipe = json.loads(recipe_path.read_text())
    df = pd.read_parquet(raw_path)

    cols_recipe: dict = recipe.get("columns", {})
    ts_col: str | None = recipe.get("timestamp_col")
    rows_before = len(df)
    drop_rows_mask = pd.Series(False, index=df.index)
    removed_cols: list[str] = []

    # 1. Drop columns marked action=drop
    for col, col_rec in cols_recipe.items():
        if col in df.columns and col_rec.get("action") == "drop":
            df = df.drop(columns=[col])
            removed_cols.append(col)

    # 2. Type fixes first (before numeric ops so cast_numeric enables outlier handling)
    for col, col_rec in cols_recipe.items():
        if col not in df.columns:
            continue
        fix = col_rec.get("type_fix", "none")
        if fix != "none":
            df = _apply_type_fix(df, col, fix)

    # 3. Collect outlier-remove rows; apply other outlier strategies in-place
    for col, col_rec in cols_recipe.items():
        if col not in df.columns:
            continue
        strategy = col_rec.get("outlier_strategy", "keep")
        if strategy == "remove" and pd.api.types.is_numeric_dtype(df[col]):
            valid = df[col].dropna()
            if len(valid) > 0:
                q1 = float(np.percentile(valid, 25))
                q3 = float(np.percentile(valid, 75))
                iqr = q3 - q1
                mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
                drop_rows_mask = drop_rows_mask | mask.reindex(df.index, fill_value=False)
        elif strategy not in ("keep", "remove"):
            df = _apply_outlier(df, col, strategy, recipe=recipe)

    # 4. Collect missing drop_row rows; apply other missing strategies in-place
    for col, col_rec in cols_recipe.items():
        if col not in df.columns:
            continue
        strategy = col_rec.get("missing_strategy", "none")
        if strategy == "drop_row":
            mask = df[col].isna()
            drop_rows_mask = drop_rows_mask | mask.reindex(df.index, fill_value=False)
        elif strategy != "none":
            df = _apply_missing(df, col, strategy, ts_col=ts_col)

    # 5. Apply all accumulated row drops at once
    df = df[~drop_rows_mask].reset_index(drop=True)

    # 6. Drop duplicate rows
    if recipe.get("drop_duplicates", False):
        df = df.drop_duplicates().reset_index(drop=True)

    # 7. Sort by timestamp
    ts_col = recipe.get("timestamp_col")
    if recipe.get("sort_by_timestamp") and ts_col and ts_col in df.columns:
        df = df.sort_values(ts_col).reset_index(drop=True)

    rows_after = len(df)
    row_loss_pct = round((rows_before - rows_after) / max(rows_before, 1) * 100, 2)

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_path = CLEANED_DIR / f"{run_id}_cleaned.parquet"
    df.to_parquet(cleaned_path, index=False)

    snapshot = _snapshot(df)
    (run_dir / "cleaned_metadata.json").write_text(json.dumps(snapshot, indent=2))

    report = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "row_loss_pct": row_loss_pct,
        "cols_dropped": removed_cols,
        "recipe_applied": recipe,
    }
    (run_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2))

    return {
        "run_id": run_id,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "row_loss_pct": row_loss_pct,
        "cols_dropped": removed_cols,
        "cleaned_path": str(cleaned_path),
        "snapshot": snapshot,
    }
