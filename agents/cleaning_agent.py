"""Stage 3 cleaning agent: uses LLM to decide column-level cleaning strategies.

Only the cleaning_decision_payload.json (statistics only, no raw data) is sent to the LLM.
If the LLM is unavailable or returns invalid output, a rule-based fallback is used instead.
"""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError, field_validator, model_validator

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

    # LLMs routinely swap the two no-op spellings ("none" vs "keep") across fields —
    # a full-recipe rejection over that (16 literal_errors → fallback) is self-inflicted.
    # Normalize instead of failing; real invalid strategies still raise.
    @field_validator("outlier_strategy", mode="before")
    @classmethod
    def _outlier_none_means_keep(cls, v):
        return "keep" if v == "none" else v

    @field_validator("missing_strategy", "type_fix", mode="before")
    @classmethod
    def _keep_means_none(cls, v):
        return "none" if v == "keep" else v


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

COLUMN PROFILE: the payload's "column_profile" gives per-column facts — use them:
  - "zero_pct" > 50 (zero-inflated column): quartiles collapse to zero, so clip_iqr,
    rolling_iqr, stl_residuals and remove would DESTROY the column's real values.
    Use "keep" (or "winsorize" only if extreme values are clearly errors).
  - "skew" with |skew| > 1: prefer "median_fill" over "mean_fill".
  - "distinct_pct" near 100 on a non-timestamp column = a row identifier, not a signal.
  - "null_pct" >= 95: nothing left to fill — set action "drop".
  - "n_rows" tells you the total series length for the strategy length requirements.

TYPE FIX options:
  "parse_datetime" - parse string column as datetime
  "cast_numeric"   - parse string column as float
  "encode_boolean" - convert yes/no, true/false, 1/0 strings to boolean
  "none"           - no type change needed

COLUMN ACTION options:
  "keep"  - include this column in the cleaned dataset
  "drop"  - exclude this column entirely. Use for columns that carry no forecasting
            signal: constant/near-constant columns, columns with null_pct >= 95,
            pure row identifiers (distinct_pct ≈ 100, e.g. "id"), and system/ETL audit
            columns (created_on, created_at, updated_at, loaded_at, inserted_at,
            source_file, batch/file metadata). Dropping is safe — raw data is preserved.

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
- No-op values differ per field: use "none" for missing_strategy and type_fix,
  but "keep" for outlier_strategy ("none" is NOT a valid outlier_strategy).
- Set action to "drop" for the no-signal column types listed under COLUMN ACTION —
  do not invent other reasons to drop a column.
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


# Column-name tokens that mark ETL/system metadata — safe to drop for forecasting.
_AUDIT_TOKENS = ("created", "updated", "loaded", "inserted", "modified",
                 "source_file", "audit", "batch")


def _is_audit_col(name: str) -> bool:
    n = name.lower()
    return any(t in n for t in _AUDIT_TOKENS)


def _sanitize_recipe(recipe: dict, meta: dict, payload: dict | None = None) -> None:
    """Recipe safety net applied to BOTH LLM and fallback output (in place).

    Fixes strategy/column mismatches that crash or corrupt the cleaner:
    - a column that is (almost) entirely null can't be filled from itself → drop it
    - rows without a timestamp are unusable for forecasting → drop_row on the ts col
      (interpolating the ts col itself, or time-interpolating anything else while the
      ts col has NaT, raised pandas' "NaNs in the index" NotImplementedError)
    - interpolate/mean/median are undefined for non-numeric columns → forward_fill
      (allowed when a cast_numeric type_fix makes the column numeric first)
    - quartile-based outlier strategies are degenerate on columns with IQR == 0 or
      heavy zero-inflation: the clip bounds collapse to a constant and the column's
      real values are destroyed (a 94%-zeros column failed the numeric_variance gate
      this way). Downgrade to "keep" — the gate should never see that again.
    """
    dtypes = {s["col"]: s.get("dtype_inferred") for s in meta.get("schema", [])}
    raw_dtypes = {s["col"]: s.get("dtype_raw", "") for s in meta.get("schema", [])}
    nulls = meta.get("nulls", {})
    numeric_stats = meta.get("numeric_stats", {})
    profile = (payload or {}).get("column_profile", {})
    constant_cols = set((payload or {}).get("constant_cols", []))
    ts_col = recipe.get("timestamp_col")

    _IQR_FAMILY = {"clip_iqr", "rolling_iqr", "stl_residuals", "remove"}

    for col, cr in recipe.get("columns", {}).items():
        null_pct = nulls.get(col, {}).get("pct", 0)
        if null_pct >= 95:
            if cr.get("action") == "keep":
                cr["action"] = "drop"
            continue

        # Columns whose INFERRED type differs from their STORED type must be cast —
        # dtype_issues alone doesn't cover these (it only flags categorical-inferred
        # columns). String-stored dates sort lexicographically (not chronologically);
        # string-stored numerics make fills/outlier math/downstream stages silently
        # no-op, leaving zero usable forecast targets in the cleaned parquet.
        raw = raw_dtypes.get(col, "")
        if cr.get("type_fix") == "none":
            if dtypes.get(col) == "datetime" and "datetime" not in raw:
                cr["type_fix"] = "parse_datetime"
            elif dtypes.get(col) == "numeric" and not any(t in raw for t in ("int", "float")):
                cr["type_fix"] = "cast_numeric"

        if col == ts_col:
            # drop_row is a no-op when there are no nulls, and also removes rows whose
            # timestamps fail to parse (NaT) — either way unusable for forecasting.
            cr["missing_strategy"] = "drop_row"
            cr["action"] = "keep"  # the timestamp itself is never droppable
            continue

        # A "drop" must be justified by a provable no-signal reason. LLM responses
        # sometimes drop measure columns (observed: an entire total_* family — the
        # would-be forecast targets); revert any drop that isn't clearly justified.
        if cr.get("action") == "drop":
            distinct_pct = profile.get(col, {}).get("distinct_pct", 0)
            justified = (
                col in constant_cols
                or distinct_pct >= 99      # per-row identifier
                or _is_audit_col(col)      # ETL/system metadata
            )
            if not justified:
                cr["action"] = "keep"

        numeric_ok = dtypes.get(col) == "numeric" or cr.get("type_fix") == "cast_numeric"
        if cr.get("missing_strategy") in ("interpolate", "mean_fill", "median_fill") and not numeric_ok:
            cr["missing_strategy"] = "forward_fill"

        iqr = numeric_stats.get(col, {}).get("iqr")
        zero_pct = profile.get(col, {}).get("zero_pct", 0)
        if cr.get("outlier_strategy") in _IQR_FAMILY and (iqr == 0 or zero_pct >= 50):
            cr["outlier_strategy"] = "keep"


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
    llm_response: dict | None = None   # raw parsed model output, kept for UI display
    llm_cfg = llm_client._load_llm_config()
    llm_model = f"{llm_cfg.get('provider', '?')}/{llm_cfg.get('model', '?')}"

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
            # Keep the model's own output (pre fill-in/validation) so the UI can show
            # exactly what came back — also useful when validation rejects it.
            llm_response = json.loads(json.dumps(raw))

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

    # Runs after the user timestamp override so the ts-col rule targets the final choice.
    _sanitize_recipe(recipe, meta, payload)

    (run_dir / "cleaning_recipe.json").write_text(json.dumps(recipe, indent=2))
    # Persist status so a reloaded run (/summary) can show the real source + error,
    # plus the model identity and its raw response for the Cleaning tab display.
    (run_dir / "cleaning_status.json").write_text(
        json.dumps({
            "recipe_source": recipe_source,
            "recipe_error": recipe_error,
            "llm_model": llm_model if use_llm else None,
            "llm_response": llm_response,
        }, indent=2)
    )

    return {
        "run_id": run_id,
        "recipe": recipe,
        "recipe_source": recipe_source,
        "recipe_error": recipe_error,
        "llm_model": llm_model if use_llm else None,
        "llm_response": llm_response,
    }
