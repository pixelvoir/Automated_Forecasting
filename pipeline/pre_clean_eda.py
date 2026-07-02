"""Stage 2: Pre-cleaning EDA.

Reads the local parquet from Stage 1 and Stage 1's metadata.json.
Answers: 'What quality problems exist before we can do forecasting analysis?'
Zero DB involvement — all computation is local.

Outputs:
  runs/{run_id}/pre_clean_eda_full.json        — full stats for human review
  runs/{run_id}/cleaning_decision_payload.json  — small JSON sent to LLM cleaning call
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
RAW_DIR = ROOT / "data" / "raw"
CONFIG_PATH = ROOT / "config" / "settings.yaml"

# Maps frequency label → seasonality period (used for rolling window size)
_FREQ_PERIOD = {
    "hourly": 24, "daily": 7, "weekly": 52,
    "monthly": 12, "quarterly": 4, "yearly": 1,
}


def _pick_ts_col(frequency: dict) -> str | None:
    """Return the most granular (highest-frequency) datetime column name."""
    if not frequency:
        return None
    order = {"hourly": 0, "daily": 1, "weekly": 2, "monthly": 3, "quarterly": 4, "yearly": 5}
    return min(frequency, key=lambda c: order.get(frequency[c], 99))


# Above this window size, pandas' rolling().quantile() is faster than the numpy sliding-
# window approach below (measured crossover is between 40-52 on a 3.4M-row series — the
# only real _FREQ_PERIOD value past it is "weekly" (52); every other value (1, 4, 7, 12, 24)
# is faster or much faster with numpy — up to 5.6x for daily/quarterly-sized windows).
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


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("pre_clean_eda", {})


# ── Missing data analysis ────────────────────────────────────────────────────

def _extended_missing(df: pd.DataFrame, stage1_nulls: dict) -> dict:
    """Extend Stage 1 null counts with consecutive-block analysis and pattern classification."""
    result = {}
    for col, info in stage1_nulls.items():
        if col not in df.columns:
            continue
        pct = info["pct"]
        count = info["count"]
        if count == 0:
            result[col] = {"pct": pct, "count": count, "max_consecutive": 0, "pattern": "none"}
            continue

        mask = df[col].isna()
        # Max consecutive nulls via cumsum trick
        groups = mask.ne(mask.shift()).cumsum()
        max_consec = int(mask.groupby(groups).sum().max())

        # Pattern classification
        n = len(df)
        if max_consec > 0.4 * count:
            pattern = "block"       # one big gap dominates
        elif mask.tail(max(1, n // 10)).mean() > 0.5:
            pattern = "trailing"    # most nulls at the end
        else:
            pattern = "random"

        result[col] = {
            "pct": pct,
            "count": count,
            "max_consecutive": max_consec,
            "pattern": pattern,
        }
    return result


# ── Outlier analysis (reuses Stage 1 bounds — no redundant quantile computation) ──

def _outlier_counts(
    df: pd.DataFrame,
    stage1_numeric: dict,
    iqr_mult: float = 1.5,
    z_thresh: float = 3.0,
    mad_thresh: float = 3.0,
    frequency: dict | None = None,
) -> dict:
    """Count outliers per numeric column using three global methods + one temporal method.
    Uses Stage 1's pre-computed Q1/Q3/IQR/mean/std to avoid recomputation.
    temporal_pct uses rolling IQR aligned on the most granular datetime column — a much
    lower temporal_pct vs iqr_pct signals seasonal inflation, not real errors."""
    ts_col = _pick_ts_col(frequency or {})
    # Parse the timestamp column once — re-parsing it inside the loop below (once per
    # numeric column) turns into a serious bottleneck on wide datasets: if the column's
    # format isn't cleanly inferable, pandas falls back to a slow per-row dateutil parse,
    # and repeating that N times (once per numeric column) can take minutes instead of
    # seconds on a multi-million-row series.
    ts_parsed = pd.to_datetime(df[ts_col], errors="coerce") if ts_col and ts_col in df.columns else None
    result = {}
    for col, stats in stage1_numeric.items():
        if col not in df.columns:
            continue
        # to_numeric is a no-op for already-numeric dtypes; it's required for columns
        # Stage 1 classified "numeric" via string-coercion (see ingest.py::_infer_dtype) —
        # without it, comparisons below (s < lo) would compare strings instead of numbers.
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        n = len(s)

        # IQR method — bounds from Stage 1 stats
        q1, q3, iqr = stats["q1"], stats["q3"], stats["iqr"]
        lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
        iqr_pct = round(float(((s < lo) | (s > hi)).sum()) / n * 100, 2)

        # Z-score method — mean/std from Stage 1
        mean, std = stats["mean"], stats["std"]
        if std > 0:
            z_pct = round(float((((s - mean) / std).abs() > z_thresh).sum()) / n * 100, 2)
        else:
            z_pct = 0.0

        # MAD method — median from Stage 1, MAD needs one pass
        median = stats["median"]
        mad = float((s - median).abs().median())
        if mad > 0:
            mad_pct = round(float(((s - median).abs() / mad > mad_thresh).sum()) / n * 100, 2)
        else:
            mad_pct = 0.0

        # Temporal rolling-IQR method — only when a timestamp col exists and series ≥ 30
        temporal_pct: float | None = None
        if ts_parsed is not None and ts_col != col and n >= 30:
            try:
                freq_str = (frequency or {}).get(ts_col, "daily")
                window = max(_FREQ_PERIOD.get(freq_str, 7), 7)
                # Align series on the timestamp so rolling uses sorted time order
                s_t = s.copy()
                s_t.index = ts_parsed.loc[s.index]
                s_t = s_t.sort_index()
                if window <= _FAST_ROLLING_MAX_WINDOW:
                    q1_vals, q3_vals = _fast_rolling_q1_q3(s_t.to_numpy(dtype="float64"), window)
                    q1_r, q3_r = pd.Series(q1_vals, index=s_t.index), pd.Series(q3_vals, index=s_t.index)
                else:
                    q1_r = s_t.rolling(window, center=True, min_periods=1).quantile(0.25)
                    q3_r = s_t.rolling(window, center=True, min_periods=1).quantile(0.75)
                iqr_r = q3_r - q1_r
                mask = (s_t < q1_r - 1.5 * iqr_r) | (s_t > q3_r + 1.5 * iqr_r)
                temporal_pct = round(float(mask.sum()) / len(s_t) * 100, 2)
            except Exception:
                temporal_pct = None

        result[col] = {
            "iqr_pct": iqr_pct, "zscore_pct": z_pct, "mad_pct": mad_pct,
            "temporal_pct": temporal_pct,
        }
    return result


# ── Timestamp integrity ──────────────────────────────────────────────────────

def _timestamp_integrity(df: pd.DataFrame, datetime_cols: list[str], frequency: dict) -> dict:
    result = {}
    for col in datetime_cols:
        if col not in df.columns:
            continue
        try:
            s = pd.to_datetime(df[col].dropna()).sort_values()
            is_monotonic = bool(pd.to_datetime(df[col].dropna()).is_monotonic_increasing)
            dup_count = int(s.duplicated().sum())

            # Infer expected gap from frequency string
            freq_str = frequency.get(col, "unknown")
            freq_map = {
                "hourly": pd.Timedelta("1h"),
                "daily": pd.Timedelta("1D"),
                "weekly": pd.Timedelta("7D"),
                "monthly": pd.Timedelta("30D"),
                "quarterly": pd.Timedelta("90D"),
                "yearly": pd.Timedelta("365D"),
            }
            expected_gap = freq_map.get(freq_str)
            gap_count = 0
            if expected_gap and len(s) > 1:
                diffs = s.diff().dropna()
                gap_count = int((diffs > expected_gap * 1.5).sum())

            result[col] = {
                "is_monotonic": is_monotonic,
                "duplicate_timestamps": dup_count,
                "gap_count": gap_count,
                "frequency": freq_str,
            }
        except Exception:
            result[col] = {"is_monotonic": None, "duplicate_timestamps": 0, "gap_count": 0}
    return result


# ── Dtype issue detection ────────────────────────────────────────────────────

def _dtype_issues(df: pd.DataFrame, schema: list[dict]) -> list[dict]:
    """Find object columns that are actually numeric or datetime (stored as strings)."""
    issues = []
    for item in schema:
        col = item["col"]
        if item["dtype_inferred"] != "categorical" or col not in df.columns:
            continue
        sample = df[col].dropna().head(200)
        if len(sample) == 0:
            continue

        # Try numeric parse
        try:
            pd.to_numeric(sample, errors="raise")
            numeric_ok = True
        except (ValueError, TypeError):
            numeric_ok = False

        if not numeric_ok:
            try:
                numeric_frac = pd.to_numeric(sample, errors="coerce").notna().mean()
                numeric_ok = numeric_frac > 0.8
            except Exception:
                numeric_ok = False

        if numeric_ok:
            issues.append({"col": col, "issue": "numeric_as_string"})
            continue

        # Try datetime parse
        try:
            pd.to_datetime(sample, errors="raise")
            issues.append({"col": col, "issue": "datetime_as_string"})
        except Exception:
            try:
                dt_frac = pd.to_datetime(sample, errors="coerce").notna().mean()
                if dt_frac > 0.8:
                    issues.append({"col": col, "issue": "datetime_as_string"})
            except Exception:
                pass

    return issues


# ── Constant column detection (from Stage 1 metadata — no parquet read needed) ──

def _constant_from_meta(stage1: dict, variance_threshold: float = 0.001) -> list[str]:
    constant = []
    # From cardinality: unique_count == 1
    for col, info in stage1.get("cardinality", {}).items():
        if info.get("unique_count", 2) <= 1:
            constant.append(col)
    # From numeric_stats: std is near zero
    for col, stats in stage1.get("numeric_stats", {}).items():
        if col not in constant:
            std = stats.get("std", 1.0)
            mean = abs(stats.get("mean", 1.0))
            if mean > 0 and std / mean < variance_threshold:
                constant.append(col)
            elif mean == 0 and std < 1e-9:
                constant.append(col)
    return constant


# ── Structural break detection ───────────────────────────────────────────────

def _structural_breaks(
    df: pd.DataFrame,
    datetime_cols: list[str],
    min_rows: int = 50,
    penalty: str = "bic",
) -> list[str]:
    """Detect structural breakpoints using ruptures PELT algorithm.
    Returns list of date strings (empty if < min_rows or ruptures unavailable)."""
    try:
        import ruptures as rpt
    except ImportError:
        return []

    # Find the first numeric column to analyse
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols or len(df) < min_rows:
        return []

    target = df[numeric_cols[0]].ffill().fillna(0).values

    try:
        model = rpt.Pelt(model="rbf").fit(target)
        breakpoints = model.predict(pen=_ruptures_penalty(target, penalty))
        breakpoints = [b for b in breakpoints if b < len(df)]  # remove sentinel

        # Convert integer indices to date strings if a datetime index exists
        if datetime_cols and datetime_cols[0] in df.columns:
            dt_series = pd.to_datetime(df[datetime_cols[0]])
            return [str(dt_series.iloc[b - 1].date()) for b in breakpoints if b - 1 < len(dt_series)]
        return [str(b) for b in breakpoints]
    except Exception:
        return []


def _ruptures_penalty(signal: np.ndarray, method: str) -> float:
    n = len(signal)
    var = np.var(signal) or 1.0
    if method == "bic":
        return np.log(n) * var
    if method == "aic":
        return 2 * var
    return var  # l2 fallback


# ── Column profile (statistics only — no raw values) ────────────────────────

def _column_profile(df: pd.DataFrame, stage1: dict) -> dict:
    """Compact per-column facts for the LLM payload (~10 tokens/column). This is what
    lets the model choose strategies from evidence instead of guessing:
    - zero_pct  → IQR-family outlier strategies are degenerate on zero-inflated columns
    - skew      → mean_fill vs median_fill
    - distinct_pct → ID-like columns (≈100 on non-timestamp columns carry no signal)
    - dtype/null_pct → drop candidates, fill feasibility
    """
    rows = max(stage1.get("shape", {}).get("rows", len(df)), 1)
    nulls = stage1.get("nulls", {})
    cardinality = stage1.get("cardinality", {})
    numeric_stats = stage1.get("numeric_stats", {})

    profile = {}
    for item in stage1.get("schema", []):
        col = item["col"]
        p: dict = {
            "dtype": item.get("dtype_inferred", "unknown"),
            "null_pct": nulls.get(col, {}).get("pct", 0),
        }
        unique = cardinality.get(col, {}).get("unique_count")
        if unique is None and col in df.columns:
            # Stage 1 only computes cardinality for categoricals — numeric/datetime ID
            # columns need it too (a per-row-unique "id" column is a drop candidate).
            try:
                unique = int(df[col].nunique())
            except Exception:
                unique = None
        if unique is not None:
            p["distinct_pct"] = round(unique / rows * 100, 2)
        stats = numeric_stats.get(col)
        if stats is not None and col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            p["zero_pct"] = round(float((s == 0).mean() * 100), 2)
            p["skew"] = round(stats.get("skew", 0), 2)
        profile[col] = p
    return profile


# ── Decision payload extractor ───────────────────────────────────────────────

def _build_decision_payload(full_stats: dict, stage1: dict) -> dict:
    """Extract the minimal JSON that gets sent to the LLM cleaning call.
    No raw values, no column samples — counts, percentages, flags only."""
    missing_payload = {}
    for col, info in full_stats["missing"].items():
        if info["pct"] > 0:
            missing_payload[col] = {
                "pct": info["pct"],
                "max_consecutive": info["max_consecutive"],
                "pattern": info["pattern"],
            }

    outlier_payload = {
        col: stats
        for col, stats in full_stats["outliers"].items()
        if any(v is not None and v > 0 for v in stats.values())
    }

    return {
        "n_rows": stage1.get("shape", {}).get("rows", 0),
        "column_profile": full_stats.get("column_profile", {}),
        "missing": missing_payload,
        "outliers": outlier_payload,
        "duplicates": {
            "rows": full_stats["duplicates"]["rows"],
            "timestamps": full_stats["duplicates"]["timestamps"],
        },
        "dtype_issues": full_stats["dtype_issues"],
        "constant_cols": full_stats["constant_cols"],
        "breakpoints": full_stats["breakpoints"],
        "frequency": stage1.get("frequency", {}),
    }


# ── Stage 2 entry point ──────────────────────────────────────────────────────

def run(run_id: str) -> dict:
    """Stage 2 entry point. Reads local parquet + Stage 1 metadata. Zero DB involvement."""
    cfg = _load_config()

    parquet_path = RAW_DIR / f"{run_id}_raw.parquet"
    meta_path = RUNS_DIR / run_id / "metadata.json"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Raw parquet not found: {parquet_path}. Run Stage 1 first.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Stage 1 metadata not found: {meta_path}. Run Stage 1 first.")

    df = pd.read_parquet(parquet_path)
    stage1 = json.loads(meta_path.read_text())

    datetime_cols = stage1.get("datetime_cols", [])
    frequency = stage1.get("frequency", {})

    ts_integrity = _timestamp_integrity(df, datetime_cols, frequency)

    full_stats = {
        "column_profile": _column_profile(df, stage1),
        "missing": _extended_missing(df, stage1.get("nulls", {})),
        "outliers": _outlier_counts(
            df,
            stage1.get("numeric_stats", {}),
            iqr_mult=cfg.get("outlier_iqr_multiplier", 1.5),
            z_thresh=cfg.get("outlier_zscore_threshold", 3.0),
            mad_thresh=cfg.get("outlier_mad_threshold", 3.0),
            frequency=frequency,
        ),
        "duplicates": {
            "rows": stage1.get("duplicates", {}).get("row_count", 0),
            "timestamps": sum(v.get("duplicate_timestamps", 0) for v in ts_integrity.values()),
        },
        "dtype_issues": _dtype_issues(df, stage1.get("schema", [])),
        "timestamp_integrity": ts_integrity,
        "constant_cols": _constant_from_meta(
            stage1,
            variance_threshold=cfg.get("constant_col_variance_threshold", 0.001),
        ),
        "breakpoints": _structural_breaks(
            df,
            datetime_cols,
            min_rows=cfg.get("structural_breaks_min_rows", 50),
            penalty=cfg.get("structural_breaks_penalty", "bic"),
        ),
        "frequency": frequency,
        "shape": stage1.get("shape", {}),
    }

    decision_payload = _build_decision_payload(full_stats, stage1)

    # Save outputs
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pre_clean_eda_full.json").write_text(
        json.dumps(full_stats, indent=2, default=str)
    )
    (run_dir / "cleaning_decision_payload.json").write_text(
        json.dumps(decision_payload, indent=2, default=str)
    )

    return {
        "run_id": run_id,
        "status": "completed",
        "decision_payload": decision_payload,
        "full_stats_path": str(run_dir / "pre_clean_eda_full.json"),
    }
