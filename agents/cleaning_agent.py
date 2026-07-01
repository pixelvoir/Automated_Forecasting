"""Stage 3 cleaning agent: uses LLM to decide column-level cleaning strategies.

Only the cleaning_decision_payload.json (statistics only, no raw data) is sent to the LLM.
If the LLM is unavailable or returns invalid output, a rule-based fallback is used instead.
"""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError, model_validator

from agents import llm_client
from agents.llm_client import LLMError

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

_FREQ_PERIOD = {
    "hourly": 24, "daily": 7, "weekly": 52,
    "monthly": 12, "quarterly": 4, "yearly": 1,
}


# ── Pydantic output schema ────────────────────────────────────────────────────

class ColumnRecipe(BaseModel):
    missing_strategy: Literal[
        "interpolate", "forward_fill", "backward_fill",
        "mean_fill", "median_fill", "drop_row", "flag_and_fill", "none"
    ]
    outlier_strategy: Literal[
        "winsorize", "clip_iqr", "log_transform", "remove", "keep",
        "rolling_iqr", "stl_residuals"
    ]
    type_fix: Literal["parse_datetime", "cast_numeric", "encode_boolean", "none"]
    action: Literal["keep", "drop"]


class CleaningRecipe(BaseModel):
    columns: dict[str, ColumnRecipe]
    drop_duplicates: bool
    sort_by_timestamp: bool
    timestamp_col: str | None
    frequency: str | None = None
    period: int | None = None

    @model_validator(mode="after")
    def coerce_timestamp_col(self):
        if self.timestamp_col == "":
            self.timestamp_col = None
        return self


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Task:
You are a data cleaning strategy expert for time-series forecasting pipelines.
You will receive a JSON object describing data quality issues in a dataset: missing values,
outliers, type problems, and duplicate counts. You must return a cleaning recipe.
Mandatory:
Use ONLY the options listed below. Do NOT invent new strategy names.
Instructions:
MISSING STRATEGY options:
  "interpolate"    - time-based interpolation (best for evenly-spaced numeric time series)
  "forward_fill"   - propagate last valid value forward (good for sparse or step data)
  "backward_fill"  - propagate next valid value backward (use rarely)
  "mean_fill"      - fill with column mean (for random, non-temporal missing)
  "median_fill"    - fill with column median (better when column is skewed)
  "drop_row"       - drop rows where this column is null (only if missingness is very rare < 1%)
  "flag_and_fill"  - add binary _missing indicator column then forward-fill (preserves information)
  "none"           - leave missing values as-is

OUTLIER STRATEGY options:
  "rolling_iqr"    - rolling window IQR clip (window = data frequency period).
                     USE when temporal_pct is significantly lower than iqr_pct (≥2x difference)
                     — this signals the high global rate is seasonal/trend inflation, not noise.
                     Requires: timestamp_col set, series length ≥ 30.
  "stl_residuals"  - STL seasonal decomposition. Outliers detected only in residual component
                     (after removing trend + seasonal). Outlier values are REPLACED with
                     trend+seasonal — rows are NOT dropped, dataset length is preserved.
                     USE when temporal_pct << iqr_pct AND data has clear seasonality with
                     ≥ 2 complete seasonal cycles.
  "clip_iqr"       - clip at Q1 - 1.5*IQR and Q3 + 1.5*IQR (global, non-temporal).
                     Use for non-temporal numerics, or when temporal_pct is null/unavailable,
                     or when series is too short for temporal methods.
  "winsorize"      - cap at 1.5th and 98.5th percentiles
  "log_transform"  - apply log1p (use only if values are positive and right-skewed)
  "remove"         - drop rows containing outliers in this column
  "keep"           - do not modify outliers

KEY SIGNAL: each column's outlier stats include "temporal_pct" (rolling-window IQR rate).
  - If temporal_pct is significantly lower than iqr_pct (e.g. iqr_pct=8, temporal_pct=1),
    the global rate is inflated by seasonal patterns → choose "rolling_iqr" or "stl_residuals".
  - If temporal_pct ≈ iqr_pct, outliers are real noise → choose "clip_iqr" or "remove".
  - If temporal_pct is null, the series is too short or has no timestamp → use "clip_iqr".

TYPE FIX options:
  "parse_datetime" - parse string column as datetime
  "cast_numeric"   - parse string column as float
  "encode_boolean" - convert yes/no, true/false, 1/0 strings to boolean
  "none"           - no type change needed

COLUMN ACTION options:
  "keep"  - include this column in the cleaned dataset
  "drop"  - exclude this column entirely (use for constant or near-constant columns)

REQUIRED OUTPUT FORMAT (strict JSON, no additional text):
{
  "columns": {
    "<column_name>": {
      "missing_strategy": "<option>",
      "outlier_strategy": "<option>",
      "type_fix": "<option>",
      "action": "keep" or "drop"
    }
  },
  "drop_duplicates": true or false,
  "sort_by_timestamp": true or false,
  "timestamp_col": "<column_name>" or null
}

Rules:
- Every column listed in the payload must appear in "columns".
- Use "none" for strategies that don't apply to a column.
- Set action to "drop" only for columns listed in constant_cols.
- Set drop_duplicates to true if duplicates.rows > 0.
- Set sort_by_timestamp to true and set timestamp_col if the frequency dict is non-empty.
- timestamp_col must be the business/event timestamp (e.g. transaction_date, sale_date),
  NOT a system/ETL audit column (e.g. loaded_at, created_at, updated_at, insert_time).
- Do not add explanations or markdown — return pure JSON only.
"""


# ── Rule-based fallback helpers ───────────────────────────────────────────────

def _pick_ts_col(frequency: dict) -> str | None:
    """Return the most granular datetime column (lowest period = highest frequency)."""
    if not frequency:
        return None
    order = {"hourly": 0, "daily": 1, "weekly": 2, "monthly": 3, "quarterly": 4, "yearly": 5}
    return min(frequency, key=lambda c: order.get(frequency[c], 99))


def _choose_outlier(col: str, outliers: dict, rows: int, frequency: dict,
                    ts_col: str | None) -> str:
    """Pick the appropriate outlier strategy based on data characteristics."""
    col_out = outliers.get(col, {})
    if not any(v is not None and v > 0 for v in col_out.values()):
        return "keep"
    if not ts_col or not frequency:
        return "clip_iqr"
    freq_str = frequency.get(ts_col, "daily")
    period = _FREQ_PERIOD.get(freq_str, 7)
    if rows >= 2 * period:
        return "stl_residuals"
    if rows >= 30:
        return "rolling_iqr"
    return "clip_iqr"


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _rule_based_fallback(payload: dict, all_cols: list[str], rows: int = 0) -> dict:
    """Safe default recipe — used when the LLM is unavailable or returns invalid output."""
    missing = payload.get("missing", {})
    outliers = payload.get("outliers", {})
    dtype_issues = {d["col"]: d["issue"] for d in payload.get("dtype_issues", [])}
    constant_cols = set(payload.get("constant_cols", []))
    duplicates = payload.get("duplicates", {})
    frequency = payload.get("frequency", {})

    ts_col = _pick_ts_col(frequency) if isinstance(frequency, dict) else None

    columns = {}
    for col in all_cols:
        if col in constant_cols:
            columns[col] = {
                "missing_strategy": "none",
                "outlier_strategy": "keep",
                "type_fix": "none",
                "action": "drop",
            }
            continue

        miss_strat = "none"
        if col in missing:
            miss_strat = "interpolate" if ts_col else "forward_fill"

        out_strat = _choose_outlier(col, outliers, rows, frequency, ts_col)

        issue = dtype_issues.get(col, "")
        type_fix = "none"
        if issue == "numeric_as_string":
            type_fix = "cast_numeric"
        elif issue == "datetime_as_string":
            type_fix = "parse_datetime"

        columns[col] = {
            "missing_strategy": miss_strat,
            "outlier_strategy": out_strat,
            "type_fix": type_fix,
            "action": "keep",
        }

    return {
        "columns": columns,
        "drop_duplicates": duplicates.get("rows", 0) > 0,
        "sort_by_timestamp": ts_col is not None,
        "timestamp_col": ts_col,
    }


# ── Stage 3 entry point ───────────────────────────────────────────────────────

def run(run_id: str, use_llm: bool = True) -> dict:
    """Stage 3 entry point: decide cleaning strategy via LLM (with rule-based fallback).

    Reads:  runs/{run_id}/cleaning_decision_payload.json  (Stage 2 output)
            runs/{run_id}/metadata.json                   (Stage 1 output, for column list)
            runs/{run_id}/user_selections.json            (optional: user-confirmed timestamp_col)
    Writes: runs/{run_id}/cleaning_recipe.json
            runs/{run_id}/cleaning_status.json            (recipe_source + recipe_error)
    Set use_llm=False to skip the LLM entirely and use the rule-based recipe directly.
    Returns {run_id, recipe, recipe_source: "llm" | "fallback", recipe_error: str | None}.
    """
    run_dir = RUNS_DIR / run_id
    payload_path = run_dir / "cleaning_decision_payload.json"
    meta_path = run_dir / "metadata.json"

    if not payload_path.exists():
        raise FileNotFoundError(
            f"Stage 2 output not found: {payload_path}. Run Stage 2 (pre-clean EDA) first."
        )
    if not meta_path.exists():
        raise FileNotFoundError(f"Stage 1 metadata not found: {meta_path}. Run Stage 1 first.")

    payload = json.loads(payload_path.read_text())
    meta = json.loads(meta_path.read_text())
    all_cols = [s["col"] for s in meta.get("schema", [])]
    rows = meta.get("shape", {}).get("rows", 0)

    recipe_source = "llm"
    recipe_error: str | None = None
    recipe: dict | None = None

    if not use_llm:
        # Deliberate rule-based-only run — not a failure, so no error is recorded.
        recipe_source = "fallback"
        recipe = _rule_based_fallback(payload, all_cols, rows=rows)
    else:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Here is the dataset quality report:\n\n"
                    + json.dumps(payload, indent=2)
                    + "\n\nPlease return the cleaning recipe JSON."
                ),
            },
        ]

        try:
            raw = llm_client.call(messages, require_json=True)

            # Fill in any columns the LLM may have omitted
            if "columns" in raw and isinstance(raw["columns"], dict):
                for col in all_cols:
                    if col not in raw["columns"]:
                        raw["columns"][col] = {
                            "missing_strategy": "none",
                            "outlier_strategy": "keep",
                            "type_fix": "none",
                            "action": "keep",
                        }

            validated = CleaningRecipe.model_validate(raw)
            recipe = validated.model_dump()
        except (LLMError, ValidationError, Exception) as exc:
            recipe_error = f"{type(exc).__name__}: {exc}"
            print(
                f"[cleaning_agent] LLM unavailable or invalid "
                f"({recipe_error}). Using rule-based fallback."
            )
            recipe_source = "fallback"
            recipe = _rule_based_fallback(payload, all_cols, rows=rows)

    # Safety net: force the correct type_fix for any column Stage 2 flagged as stored-as-
    # string, regardless of what the LLM chose (the rule-based fallback already gets this
    # right, but an LLM response isn't guaranteed to honor the dtype_issues signal). Without
    # the cast, cleaner.py's numeric/datetime operations would silently no-op on that column.
    _ISSUE_TO_FIX = {"numeric_as_string": "cast_numeric", "datetime_as_string": "parse_datetime"}
    for issue_item in payload.get("dtype_issues", []):
        col, issue = issue_item.get("col"), issue_item.get("issue")
        col_recipe = recipe.get("columns", {}).get(col)
        if col_recipe and col_recipe.get("type_fix") == "none" and issue in _ISSUE_TO_FIX:
            col_recipe["type_fix"] = _ISSUE_TO_FIX[issue]

    # Inject frequency + period into recipe so the cleaner can use temporal window sizes
    freq_dict = payload.get("frequency", {})
    ts_in_recipe = recipe.get("timestamp_col")
    freq_str = (
        freq_dict.get(ts_in_recipe) if ts_in_recipe and ts_in_recipe in freq_dict
        else next(iter(freq_dict.values()), None)
    )
    recipe["frequency"] = freq_str
    recipe["period"] = _FREQ_PERIOD.get(freq_str, 7) if freq_str else 7

    # User-confirmed timestamp always wins over LLM / fallback choice
    selections_path = run_dir / "user_selections.json"
    if selections_path.exists():
        sel = json.loads(selections_path.read_text())
        user_ts = sel.get("timestamp_col")
        if user_ts:
            recipe["timestamp_col"] = user_ts
            recipe["sort_by_timestamp"] = True
            # Re-derive frequency/period for the user-selected column
            user_freq_str = freq_dict.get(user_ts) or freq_str
            recipe["frequency"] = user_freq_str
            recipe["period"] = _FREQ_PERIOD.get(user_freq_str, 7) if user_freq_str else 7

    (run_dir / "cleaning_recipe.json").write_text(json.dumps(recipe, indent=2))
    # Persist status so a reloaded run (/summary) can show the real source + error.
    (run_dir / "cleaning_status.json").write_text(
        json.dumps({"recipe_source": recipe_source, "recipe_error": recipe_error}, indent=2)
    )

    return {
        "run_id": run_id,
        "recipe": recipe,
        "recipe_source": recipe_source,
        "recipe_error": recipe_error,
    }
