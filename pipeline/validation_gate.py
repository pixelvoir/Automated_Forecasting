"""Stage 3.5 validation gate: pass/fail quality checks on the cleaned dataset.

Compares Stage 1 metadata against post-cleaning metadata to catch regressions.
Does not call the LLM. Does not halt the pipeline — returns pass/fail to the API layer.
"""
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
CLEANED_DIR = ROOT / "data" / "cleaned"
CONFIG_PATH = ROOT / "config" / "settings.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("validation_gate", {})


def run(run_id: str) -> dict:
    """Stage 3.5 entry point: compare pre- and post-cleaning metadata.

    Reads:
      runs/{run_id}/metadata.json          (Stage 1 — original shape and nulls)
      runs/{run_id}/cleaned_metadata.json  (Stage 3 — cleaned data snapshot)
      runs/{run_id}/cleaning_recipe.json   (to know timestamp col and sort intent)
      data/cleaned/{run_id}_cleaned.parquet (for monotonicity check)
    Writes: runs/{run_id}/validation_gate.json
    Returns {run_id, passed, checks, row_delta_pct, rows_before, rows_after}.
    """
    cfg = _load_config()
    max_row_loss_pct = float(cfg.get("max_row_loss_pct", 15))
    min_series_length = int(cfg.get("min_series_length", 30))

    run_dir = RUNS_DIR / run_id
    meta_path = run_dir / "metadata.json"
    cleaned_meta_path = run_dir / "cleaned_metadata.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Stage 1 metadata not found: {meta_path}.")
    if not cleaned_meta_path.exists():
        raise FileNotFoundError(
            f"Cleaned metadata not found: {cleaned_meta_path}. Run Stage 3 (cleaning) first."
        )

    stage1 = json.loads(meta_path.read_text())
    cleaned = json.loads(cleaned_meta_path.read_text())

    recipe_path = run_dir / "cleaning_recipe.json"
    recipe = json.loads(recipe_path.read_text()) if recipe_path.exists() else {}
    ts_col = recipe.get("timestamp_col")

    rows_before = stage1.get("shape", {}).get("rows", 0)
    rows_after = cleaned.get("rows", 0)
    row_delta_pct = round((rows_before - rows_after) / max(rows_before, 1) * 100, 2)

    orig_nulls: dict = stage1.get("nulls", {})
    clean_nulls: dict = cleaned.get("nulls", {})
    numeric_variance: dict = cleaned.get("numeric_variance", {})

    # Every check carries a severity: "blocking" checks decide the gate outcome
    # (they catch data destruction); "warning" checks surface forecasting risks
    # without failing the run.
    checks: dict = {}

    # 1. Row loss within threshold (percentage-based, configurable)
    checks["row_loss"] = {
        "passed": row_delta_pct <= max_row_loss_pct,
        "severity": "blocking",
        "detail": f"{row_delta_pct}% rows removed (threshold: {max_row_loss_pct}%)",
    }

    # 2. Minimum series length (deliberately absolute — 30 rows is a statistical
    # floor for any seasonal/trend estimation, regardless of dataset size)
    checks["series_length"] = {
        "passed": rows_after >= min_series_length,
        "severity": "blocking",
        "detail": f"{rows_after} rows remaining (minimum: {min_series_length})",
    }

    # 3. No null regression — nulls must not increase for any column.
    # Columns with a type_fix are exempt: cast_numeric/parse_datetime legitimately
    # turn junk strings into NaN — that's a repair, not a regression.
    recipe_cols = recipe.get("columns", {})
    coerced_cols = {
        c for c, r in recipe_cols.items() if r.get("type_fix") not in (None, "none")
    }
    null_regressions = []
    for col, orig in orig_nulls.items():
        if col in coerced_cols:
            continue
        orig_pct = orig.get("pct", 0)
        cleaned_pct = clean_nulls.get(col, {}).get("pct", 0)
        if cleaned_pct > orig_pct + 0.1:  # 0.1% tolerance for float precision
            null_regressions.append(f"{col}: {orig_pct}% → {cleaned_pct}%")
    checks["no_null_regression"] = {
        "passed": len(null_regressions) == 0,
        "severity": "blocking",
        "detail": (
            f"Null increases detected: {null_regressions}"
            if null_regressions
            else "No null regressions"
            + (f" ({len(coerced_cols)} type-coerced column(s) exempt)" if coerced_cols else "")
        ),
    }

    # 4. Cleaning must not destroy variance. Relative check: only blame cleaning for
    # columns that HAD variance before (a column that was already constant is a
    # drop-candidate problem, not a cleaning regression).
    stage1_stats = stage1.get("numeric_stats", {})
    zero_variance = [
        col for col, var in numeric_variance.items()
        if var == 0.0 and stage1_stats.get(col, {}).get("std", 0) > 0
    ]
    checks["numeric_variance"] = {
        "passed": len(zero_variance) == 0,
        "severity": "blocking",
        "detail": (
            f"Cleaning collapsed variance to zero: {zero_variance}"
            if zero_variance
            else "No column had its variance destroyed by cleaning"
        ),
    }

    # 5. Timestamp completeness — rows without a timestamp are unusable for forecasting
    if ts_col:
        ts_null_count = clean_nulls.get(ts_col, {}).get("count", 0)
        checks["timestamp_nulls"] = {
            "passed": ts_null_count == 0,
            "severity": "blocking",
            "detail": (
                f"Timestamp '{ts_col}' has no missing values"
                if ts_null_count == 0
                else f"Timestamp '{ts_col}' still has {ts_null_count} missing value(s) after cleaning"
            ),
        }

    # 6. At least one usable numeric column must survive — no target, no forecast
    usable_numeric = [col for col, var in numeric_variance.items() if var > 0]
    checks["forecastable_columns"] = {
        "passed": len(usable_numeric) > 0,
        "severity": "blocking",
        "detail": (
            f"{len(usable_numeric)} numeric column(s) with variance available as forecast targets"
            if usable_numeric
            else "No numeric column with variance remains — nothing to forecast"
        ),
    }

    # 7. Timestamp monotonicity (only relevant if the recipe sorted by timestamp)
    # 8. Future-dated timestamps (warning) — they silently corrupt train/test splits
    ts_series = None
    if ts_col and recipe.get("sort_by_timestamp"):
        cleaned_path = CLEANED_DIR / f"{run_id}_cleaned.parquet"
        if cleaned_path.exists():
            try:
                ts_series = pd.to_datetime(
                    pd.read_parquet(cleaned_path, columns=[ts_col])[ts_col],
                    errors="coerce",
                )
            except Exception:
                ts_series = None
    if ts_series is not None:
        is_monotonic = bool(ts_series.is_monotonic_increasing)
        checks["timestamp_monotonic"] = {
            "passed": is_monotonic,
            "severity": "blocking",
            "detail": (
                f"Timestamp '{ts_col}' is monotonically increasing"
                if is_monotonic
                else f"Timestamp '{ts_col}' is NOT monotonic — sort did not apply"
            ),
        }
        future_count = int((ts_series > pd.Timestamp.now() + pd.Timedelta(days=1)).sum())
        future_pct = round(future_count / max(len(ts_series), 1) * 100, 2)
        checks["future_timestamps"] = {
            "passed": future_count == 0,
            "severity": "warning",
            "detail": (
                "No future-dated timestamps"
                if future_count == 0
                else f"{future_count} row(s) ({future_pct}%) are dated in the future — "
                     "verify before train/test splitting"
            ),
        }
    else:
        checks["timestamp_monotonic"] = {
            "passed": True,
            "severity": "blocking",
            "detail": "Not applicable (no timestamp column, sort not requested, or column unreadable)",
        }

    # 9. Enough history for seasonal models (warning) — needs ≥ 2 full cycles
    period = recipe.get("period")
    if period:
        enough = rows_after >= 2 * int(period)
        checks["seasonal_history"] = {
            "passed": enough,
            "severity": "warning",
            "detail": (
                f"History covers >= 2 seasonal cycles (period {period})"
                if enough
                else f"Fewer than 2 seasonal cycles of data (period {period}) — "
                     "seasonal models (STL, SARIMA) will be unreliable"
            ),
        }

    # Only blocking checks decide the gate; warnings inform the modeling stage.
    passed = all(c["passed"] for c in checks.values() if c["severity"] == "blocking")
    warnings_failed = [
        name for name, c in checks.items()
        if c["severity"] == "warning" and not c["passed"]
    ]

    result = {
        "run_id": run_id,
        "passed": passed,
        "warnings": warnings_failed,
        "checks": checks,
        "row_delta_pct": row_delta_pct,
        "rows_before": rows_before,
        "rows_after": rows_after,
    }
    (run_dir / "validation_gate.json").write_text(json.dumps(result, indent=2))
    return result
