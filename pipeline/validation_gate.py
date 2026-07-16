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

# frequency label → pandas Period alias, for counting distinct forecast periods
_FREQ_TO_PERIOD_ALIAS = {"hourly": "h", "daily": "D", "weekly": "W",
                         "monthly": "M", "quarterly": "Q", "yearly": "Y"}


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("validation_gate", {})


def _load_intent(run_dir: Path) -> dict | None:
    """User-confirmed forecast intent (Stage 2.5), when this run has one."""
    p = run_dir / "forecast_user_selections.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


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
    intent = _load_intent(run_dir)

    # The timestamp column is needed for several checks (period counting, monotonicity,
    # future dates) — read it once. Errors degrade the affected checks, never crash.
    cleaned_path = CLEANED_DIR / f"{run_id}_cleaned.parquet"
    ts_series = None
    if ts_col and cleaned_path.exists():
        try:
            ts_series = pd.to_datetime(
                pd.read_parquet(cleaned_path, columns=[ts_col])[ts_col],
                errors="coerce",
            )
        except Exception:
            ts_series = None

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

    # 2. Minimum series length (deliberately absolute — 30 points is a statistical
    # floor for any seasonal/trend estimation, regardless of dataset size).
    # Measured in DISTINCT PERIODS at the forecast frequency when a timestamp exists:
    # a 4.5M-row event log spanning 90 days is a 90-point series, not a 4.5M-point one.
    fcst_freq = (intent or {}).get("forecast_frequency") or recipe.get("frequency")
    period_alias = _FREQ_TO_PERIOD_ALIAS.get(fcst_freq)
    n_periods = None
    if ts_series is not None and period_alias:
        try:
            n_periods = int(ts_series.dropna().dt.to_period(period_alias).nunique())
        except Exception:
            n_periods = None
    if n_periods is not None:
        checks["series_length"] = {
            "passed": n_periods >= min_series_length,
            "severity": "blocking",
            "detail": (f"{n_periods} distinct {fcst_freq} period(s) in the data "
                       f"(minimum: {min_series_length})"),
        }
    else:
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
        if var == 0.0 and (stage1_stats.get(col, {}).get("std") or 0) > 0
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

    # 6. At least one usable numeric column must survive — no target, no forecast.
    # (Skipped when the confirmed target is a count-type event ID: an event log with
    # zero numeric measures is perfectly forecastable as counts per period.)
    usable_numeric = [col for col, var in numeric_variance.items() if var > 0]
    if not (intent and intent.get("agg") in ("nunique", "count")):
        checks["forecastable_columns"] = {
            "passed": len(usable_numeric) > 0,
            "severity": "blocking",
            "detail": (
                f"{len(usable_numeric)} numeric column(s) with variance available as forecast targets"
                if usable_numeric
                else "No numeric column with variance remains — nothing to forecast"
            ),
        }

    # 6b. The CONFIRMED target must survive cleaning intact (intent-aware runs only).
    # A measure target needs variance; a count target just needs non-null IDs left.
    if intent and intent.get("target_col"):
        target = intent["target_col"]
        agg = intent.get("agg", "sum")
        if target not in clean_nulls:
            passed_t, detail_t = False, f"Target '{target}' is missing from the cleaned data"
        elif agg in ("nunique", "count"):
            null_pct_t = clean_nulls.get(target, {}).get("pct", 0)
            passed_t = null_pct_t < 100
            detail_t = (f"Count target '{target}' retains non-null event IDs "
                        f"({null_pct_t}% null)")
            if not passed_t:
                detail_t = f"Count target '{target}' is entirely null after cleaning"
        else:
            var_t = numeric_variance.get(target, 0.0)
            passed_t = var_t > 0
            detail_t = (f"Target '{target}' retains variance ({var_t:.4g})" if passed_t
                        else f"Target '{target}' has zero variance / is non-numeric after cleaning")
        checks["target_survived"] = {
            "passed": passed_t, "severity": "blocking", "detail": detail_t,
        }

    # 6b-2. The target's LEVEL must survive too (sum/mean measure targets): a recipe
    # that clips/winsorizes the target compresses its top tail, silently shrinking every
    # aggregated total — and the forecast inherits the bias (observed real failure:
    # clip_iqr on a sum-target cut cleaned totals far below raw). Row loss legitimately
    # removes target value with the rows, so the allowance is threshold + row loss.
    report_path = run_dir / "cleaning_report.json"
    creport = json.loads(report_path.read_text()) if report_path.exists() else {}
    if intent and intent.get("agg") in ("sum", "mean") and creport.get("target_total_before"):
        before = float(creport["target_total_before"])
        after = creport.get("target_total_after")
        max_drift = float(cfg.get("max_target_sum_drift_pct", 5))
        if after is not None and before:
            drift_pct = round((float(after) - before) / abs(before) * 100, 2)
            allowed = max_drift + max(row_delta_pct, 0)
            passed_lvl = abs(drift_pct) <= allowed
            detail_lvl = (f"Target '{intent['target_col']}' total changed {drift_pct:+}% "
                          f"(allowed: ±{max_drift}% beyond the {row_delta_pct}% row loss)")
            if not passed_lvl:
                detail_lvl += (" — a level-mutating strategy (clip/winsorize/log) likely "
                               "compressed the values; re-run cleaning")
            checks["target_level_preserved"] = {
                "passed": passed_lvl, "severity": "blocking", "detail": detail_lvl,
            }

    # 6c. Series-key columns must survive when the user chose per-series scope
    if intent and intent.get("scope") == "per_series" and intent.get("group_cols"):
        missing_groups = [c for c in intent["group_cols"] if c not in clean_nulls]
        checks["group_cols_survived"] = {
            "passed": not missing_groups,
            "severity": "blocking",
            "detail": (f"Series-key column(s) missing after cleaning: {missing_groups}"
                       if missing_groups else
                       f"Series-key column(s) present: {intent['group_cols']}"),
        }

    # 7. Timestamp monotonicity (only relevant if the recipe sorted by timestamp)
    # 8. Future-dated timestamps (warning) — they silently corrupt train/test splits
    if ts_series is not None and recipe.get("sort_by_timestamp"):
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
    else:
        checks["timestamp_monotonic"] = {
            "passed": True,
            "severity": "blocking",
            "detail": "Not applicable (no timestamp column, sort not requested, or column unreadable)",
        }
    if ts_series is not None:
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

    # 8b. Per-series history depth (warning, per-series scope only): the panel median
    # series must span ≥ 2 seasonal cycles for per-series seasonal models to be viable.
    if (intent and intent.get("scope") == "per_series" and intent.get("group_cols")
            and ts_col and cleaned_path.exists() and period_alias):
        gcols = [c for c in intent["group_cols"] if c in clean_nulls]
        if gcols:
            try:
                sub = pd.read_parquet(cleaned_path, columns=[*gcols, ts_col])
                sub[ts_col] = pd.to_datetime(sub[ts_col], errors="coerce")
                per_series_len = (
                    sub.dropna(subset=[ts_col])
                    .assign(_p=lambda d: d[ts_col].dt.to_period(period_alias))
                    .groupby(gcols, observed=True)["_p"].nunique()
                )
                median_len = int(per_series_len.median())
                need = 2 * int(recipe.get("period") or 7)
                checks["per_series_history"] = {
                    "passed": median_len >= need,
                    "severity": "warning",
                    "detail": (f"Median series length {median_len} {fcst_freq} period(s) "
                               f"across {len(per_series_len):,} series "
                               f"(2 seasonal cycles = {need})"),
                }
            except Exception:
                pass

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
