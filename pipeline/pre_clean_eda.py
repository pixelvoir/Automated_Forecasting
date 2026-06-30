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
) -> dict:
    """Count outliers per numeric column using three methods.
    Uses Stage 1's pre-computed Q1/Q3/IQR/mean/std to avoid recomputation."""
    result = {}
    for col, stats in stage1_numeric.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
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

        result[col] = {"iqr_pct": iqr_pct, "zscore_pct": z_pct, "mad_pct": mad_pct}
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
        if any(v > 0 for v in stats.values())
    }

    return {
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
        "missing": _extended_missing(df, stage1.get("nulls", {})),
        "outliers": _outlier_counts(
            df,
            stage1.get("numeric_stats", {}),
            iqr_mult=cfg.get("outlier_iqr_multiplier", 1.5),
            z_thresh=cfg.get("outlier_zscore_threshold", 3.0),
            mad_thresh=cfg.get("outlier_mad_threshold", 3.0),
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
