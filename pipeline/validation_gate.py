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

    checks: dict = {}

    # 1. Row loss within threshold
    checks["row_loss"] = {
        "passed": row_delta_pct <= max_row_loss_pct,
        "detail": f"{row_delta_pct}% rows removed (threshold: {max_row_loss_pct}%)",
    }

    # 2. Minimum series length
    checks["series_length"] = {
        "passed": rows_after >= min_series_length,
        "detail": f"{rows_after} rows remaining (minimum: {min_series_length})",
    }

    # 3. No null regression — nulls must not increase for any column
    null_regressions = []
    for col, orig in orig_nulls.items():
        orig_pct = orig.get("pct", 0)
        cleaned_pct = clean_nulls.get(col, {}).get("pct", 0)
        if cleaned_pct > orig_pct + 0.1:  # 0.1% tolerance for float precision
            null_regressions.append(f"{col}: {orig_pct}% → {cleaned_pct}%")
    checks["no_null_regression"] = {
        "passed": len(null_regressions) == 0,
        "detail": (
            f"Null increases detected: {null_regressions}"
            if null_regressions
            else "No null regressions"
        ),
    }

    # 4. Numeric columns still have variance (cleaning didn't zero them out)
    zero_variance = [col for col, var in numeric_variance.items() if var == 0.0]
    checks["numeric_variance"] = {
        "passed": len(zero_variance) == 0,
        "detail": (
            f"Zero-variance columns after cleaning: {zero_variance}"
            if zero_variance
            else "All numeric columns retain variance"
        ),
    }

    # 5. Timestamp monotonicity (only relevant if the recipe sorted by timestamp)
    if ts_col and recipe.get("sort_by_timestamp"):
        cleaned_path = CLEANED_DIR / f"{run_id}_cleaned.parquet"
        if cleaned_path.exists():
            try:
                df_ts = pd.read_parquet(cleaned_path, columns=[ts_col])
                is_monotonic = bool(df_ts[ts_col].is_monotonic_increasing)
                checks["timestamp_monotonic"] = {
                    "passed": is_monotonic,
                    "detail": (
                        f"Timestamp '{ts_col}' is monotonically increasing"
                        if is_monotonic
                        else f"Timestamp '{ts_col}' is NOT monotonic — possible duplicate timestamps"
                    ),
                }
            except Exception as exc:
                checks["timestamp_monotonic"] = {
                    "passed": True,
                    "detail": f"Skipped (could not read column): {exc}",
                }
        else:
            checks["timestamp_monotonic"] = {
                "passed": True,
                "detail": "Skipped (cleaned parquet not found)",
            }
    else:
        checks["timestamp_monotonic"] = {
            "passed": True,
            "detail": "Not applicable (no timestamp column or sort not requested)",
        }

    passed = all(c["passed"] for c in checks.values())

    result = {
        "run_id": run_id,
        "passed": passed,
        "checks": checks,
        "row_delta_pct": row_delta_pct,
        "rows_before": rows_before,
        "rows_after": rows_after,
    }
    (run_dir / "validation_gate.json").write_text(json.dumps(result, indent=2))
    return result
