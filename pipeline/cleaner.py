"""Stage 3 cleaner: apply the LLM-chosen cleaning recipe to the raw parquet.

All transforms are hardcoded Python functions. No eval(), no exec().
Row drops (for outlier removal and missing drop_row) are batched and applied once
to avoid skipping rows that match multiple criteria.

Columnar execution (2026-07-16): only the columns the recipe actually touches (plus
timestamp / series key / target) are materialized in pandas ("active"). Untouched
("passive") columns never leave Arrow — they are row-aligned to the final output
with a single take() at write time. The previous whole-frame pandas load was ~8 GB
for a 5.3M×58 dataset (more than the machine's RAM) and spent ~20 minutes swapping;
ingest and Stage 2 profiling were already columnar for exactly this reason —
cleaning was the last stage that wasn't.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
RAW_DIR = ROOT / "data" / "raw"
CLEANED_DIR = ROOT / "data" / "cleaned"
CONFIG_PATH = ROOT / "config" / "settings.yaml"


def _target_drift_threshold() -> float:
    """The validation gate's max_target_sum_drift_pct — shared so the execution-time
    target-level guard below and the gate can never disagree."""
    try:
        cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        return float(cfg.get("validation_gate", {}).get("max_target_sum_drift_pct", 5))
    except Exception:
        return 5.0


# ── Missing-value handlers ────────────────────────────────────────────────────
# When ``group_cols`` is set (user confirmed the data is many parallel series), every
# series-boundary-sensitive fill runs WITHIN each series: a forward-fill must never
# carry the last value of one store/society into the first row of the next. Rows are
# pre-sorted by (group, time) in run() before any of these execute.

def _apply_missing(df: pd.DataFrame, col: str, strategy: str,
                   ts_col: str | None = None,
                   group_cols: list[str] | None = None) -> pd.DataFrame:
    s = df[col]
    grouped = bool(group_cols)

    # dropna=False everywhere a series key groups rows: pandas' default (dropna=True)
    # EXCLUDES rows whose key is null from the result, and the index-aligned assignment
    # back then writes NaN into every excluded row — observed real failure: a 99.94%-null
    # series key (WARDNAME) turned per-series forward_fill into a mass ERASE of four
    # columns' real values (6% null → 99.94% null, no_null_regression gate failure).
    # Null-key rows form their own "(unknown)" series instead.
    def _gb():
        return df.groupby(group_cols, sort=False, dropna=False)

    if strategy == "interpolate":
        # Interpolation is only defined for numerics — a recipe (LLM or fallback) can
        # still assign it to a string/datetime column, so degrade to fill instead of
        # letting pandas raise.
        if not pd.api.types.is_numeric_dtype(s):
            if grouped:
                df[col] = _gb()[col].ffill()
                df[col] = _gb()[col].bfill()
            else:
                df[col] = s.ffill().bfill()
            return df
        if grouped:
            # Per-series positional interpolation (rows already time-ordered inside each
            # series). Time-weighted interpolation per series would need a reindex per
            # group for marginal gain on regular grids.
            df[col] = df.groupby(group_cols, sort=False, group_keys=False,
                                 dropna=False)[col].apply(lambda x: x.interpolate())
        # time-based interpolation requires a DatetimeIndex.
        # Raw parquet has a RangeIndex, so temporarily align the series on the
        # timestamp column so pandas uses actual time gaps between rows.
        elif ts_col and ts_col in df.columns and ts_col != col:
            ts_index = pd.to_datetime(df[ts_col], errors="coerce")
            if ts_index.isna().any():
                # pandas raises NotImplementedError on NaT in the index; positional
                # interpolation is the safe approximation when timestamps are missing.
                df[col] = s.interpolate().values
            else:
                temp = s.copy()
                temp.index = ts_index
                temp = temp.interpolate(method="time")
                df[col] = temp.values
        else:
            df[col] = s.interpolate()  # linear fallback when no timestamp col
    elif strategy == "forward_fill":
        df[col] = _gb()[col].ffill() if grouped else s.ffill()
    elif strategy == "backward_fill":
        df[col] = _gb()[col].bfill() if grouped else s.bfill()
    elif strategy == "mean_fill":
        if pd.api.types.is_numeric_dtype(s):
            # per-series mean beats the global mean on panels (levels differ per series)
            df[col] = s.fillna(_gb()[col].transform("mean") if grouped else s.mean())
    elif strategy == "median_fill":
        if pd.api.types.is_numeric_dtype(s):
            df[col] = s.fillna(_gb()[col].transform("median") if grouped else s.median())
    elif strategy == "flag_and_fill":
        df[f"{col}_missing"] = s.isna().astype(int)
        df[col] = _gb()[col].ffill() if grouped else s.ffill()
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


def _clip_iqr_series(s: pd.Series) -> pd.Series:
    valid = s.dropna()
    if len(valid) == 0:
        return s
    q1 = float(np.percentile(valid, 25))
    q3 = float(np.percentile(valid, 75))
    iqr = q3 - q1
    return s.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def _rolling_iqr_series(s: pd.Series, window: int) -> pd.Series:
    """Centered rolling-IQR clip of ONE series (the whole dataset, or one panel series)."""
    valid = s.dropna()
    if len(valid) < window:
        return _clip_iqr_series(s)  # too short for rolling — plain IQR clip within the series
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
    return s.clip(q1_r - 1.5 * iqr_r, q3_r + 1.5 * iqr_r)


def _seasonal_replace_series(s: pd.Series, period: int, allow_stl: bool) -> pd.Series:
    """STL-residual strategy for ONE series: residual outliers (MAD-thresholded) are
    replaced with trend+seasonal — no rows dropped.

    ``allow_stl`` gates statsmodels' iterative-LOESS STL (minutes-slow past
    ``_STL_MAX_POINTS`` and unaffordable × thousands of panel series); the alternative
    is the O(n) classical decomposition (centred rolling-mean trend + per-phase seasonal
    mean), verified equivalent in output shape."""
    valid = s.dropna()
    if len(valid) < 2 * period:
        return _rolling_iqr_series(s, max(period, 7))  # not enough cycles
    if allow_stl and len(s) <= _STL_MAX_POINTS:
        from statsmodels.tsa.seasonal import STL
        s_filled = s.ffill().bfill()
        try:
            res = STL(s_filled, period=period, robust=True).fit()
            resid = res.resid
            med_resid = float(np.median(resid))
            mad = float(np.median(np.abs(resid - med_resid)))
            threshold = 3.5 * 1.4826 * mad
            outlier_mask = np.abs(resid - med_resid) > threshold
            s_clean = s.copy()
            s_clean[outlier_mask] = (res.trend + res.seasonal)[outlier_mask]
            return s_clean
        except Exception:
            return _rolling_iqr_series(s, max(period, 7))
    # O(n) classical path
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
        return s  # no dispersion in residuals → nothing to flag
    mask = np.abs(resid - med) > 3.5 * 1.4826 * mad
    replacement = (trend + seasonal).to_numpy()
    out = s.to_numpy(dtype="float64", copy=True)
    out[mask] = replacement[mask]
    return pd.Series(out, index=s.index)


# STL per panel series is only affordable for a handful of series; past this many
# groups the O(n) classical decomposition is used inside every series instead.
_STL_MAX_GROUPS = 20


def _apply_outlier(df: pd.DataFrame, col: str, strategy: str,
                   recipe: dict | None = None,
                   group_cols: list[str] | None = None) -> pd.DataFrame:
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s):
        return df
    # Nullable-integer columns (pandas Int64, e.g. from DB ingest) reject fractional
    # values: s.clip(q3 + 1.5*iqr) raises "Invalid value ... for dtype 'Int64'". Every
    # strategy below can produce fractional replacements, so work in float64.
    if not pd.api.types.is_float_dtype(s):
        s = s.astype("float64")
        df[col] = s
    valid = s.dropna()
    if len(valid) == 0:
        return df
    grouped = bool(group_cols)

    def _per_series(fn):
        # rows are pre-sorted by (group, time) in run(); group_keys=False keeps the
        # original row index so the assignment aligns. dropna=False: a null series key
        # must not exclude its rows (the assignment would NaN them — see _apply_missing)
        return df.groupby(group_cols, sort=False, group_keys=False,
                          dropna=False)[col].apply(fn)

    if strategy == "winsorize":
        # distribution-level cap — deliberately global even on panels
        lo = float(np.percentile(valid, 1.5))
        hi = float(np.percentile(valid, 98.5))
        df[col] = s.clip(lo, hi)
    elif strategy == "clip_iqr":
        df[col] = _clip_iqr_series(s)  # global distribution clip
    elif strategy == "log_transform":
        df[col] = np.log1p(s.clip(lower=0))
    elif strategy == "rolling_iqr":
        window = max(_period_from_recipe(recipe), 7)
        if grouped:
            # temporal windows must never straddle series boundaries
            df[col] = _per_series(lambda x: _rolling_iqr_series(x, window))
        else:
            df[col] = _rolling_iqr_series(s, window)
    elif strategy == "stl_residuals":
        period = max(_period_from_recipe(recipe), 2)
        if grouped:
            n_groups = df.groupby(group_cols, sort=False, observed=True,
                                  dropna=False).ngroups
            allow_stl = n_groups <= _STL_MAX_GROUPS
            df[col] = _per_series(lambda x: _seasonal_replace_series(x, period, allow_stl))
        else:
            df[col] = _seasonal_replace_series(s, period, allow_stl=True)
    # "remove" → collected upstream and applied in batch
    # "keep"   → no-op
    return df


# ── Type-fix handlers ─────────────────────────────────────────────────────────

def _parse_datetime_fast(s: pd.Series) -> pd.Series:
    """``pd.to_datetime(errors="coerce")`` with pre-validated explicit formats.

    Format-less parsing of a string column pandas can't infer falls back to
    per-element dateutil — measured 92s for ONE 5.3M-row time-of-day column
    (``RECVDTIME``); an explicit-format parse is vectorized C (~2s). Real DB columns
    are also MIXED format (RECVDTIME stores both '17:24' and '19:38:42'), so no
    single format can cover them: candidate formats are guessed from a spread of
    sample values (plus bare-time formats the guesser never returns) and LAYERED —
    each strict vectorized parse claims the values it matches, majority format
    first, and only the residue pays the per-element dateutil price.

    The layered result is accepted only if, on the sample, it parses at least as
    many values as the lenient path — anything weirder keeps the old slow-but-
    lenient behavior. Note: bare time-of-day values get the deterministic
    1900-01-01 base date instead of dateutil's "today" (an improvement — re-running
    no longer shifts the dates); intent detection demotes such columns anyway.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    non_null = s.dropna()
    if non_null.empty:
        return pd.to_datetime(s, errors="coerce")
    try:
        from pandas.tseries.api import guess_datetime_format
    except ImportError:  # pandas < 2.1
        from pandas.core.tools.datetimes import guess_datetime_format
    import warnings

    step = max(1, len(non_null) // 1000)
    sample = non_null.iloc[::step].head(1000)

    seen: list[str] = []
    for v in sample.iloc[:: max(1, len(sample) // 25)]:
        g = guess_datetime_format(str(v))
        if g and g not in seen:
            seen.append(g)
    candidates = seen + [f for f in ("%H:%M:%S", "%H:%M") if f not in seen]

    def _hits(fmt: str, x: pd.Series) -> int:
        try:
            return int(pd.to_datetime(x, format=fmt, errors="coerce").notna().sum())
        except (ValueError, TypeError):
            return 0

    # Majority format first: for values several formats could claim (rare, and
    # inherently ambiguous data), the dominant convention wins — which is also how
    # dateutil resolves the bulk of such a column.
    hit_counts = {f: _hits(f, sample) for f in candidates}
    candidates = sorted((f for f in candidates if hit_counts[f] > 0),
                        key=lambda f: -hit_counts[f])
    if not candidates:
        return pd.to_datetime(s, errors="coerce")

    def _lenient(x: pd.Series) -> pd.Series:
        with warnings.catch_warnings():
            # intentionally format-less; its "could not infer format" advisory is
            # expected noise here
            warnings.simplefilter("ignore", UserWarning)
            return pd.to_datetime(x, errors="coerce")

    def _layered(x: pd.Series) -> pd.Series:
        out = pd.Series(pd.NaT, index=x.index, dtype="datetime64[ns]")
        remaining = x.notna()
        for fmt in candidates:
            if not remaining.any():
                break
            parsed = pd.to_datetime(x[remaining], format=fmt, errors="coerce")
            hit = parsed.dropna()
            out.loc[hit.index] = hit
            remaining.loc[hit.index] = False
        if remaining.any():
            out.loc[remaining] = _lenient(x[remaining])
        return out

    lenient_ok = int(_lenient(sample).notna().sum())
    if int(_layered(sample).notna().sum()) >= lenient_ok:
        return _layered(s)
    return pd.to_datetime(s, errors="coerce")


def _fix_series(s: pd.Series, fix: str) -> pd.Series:
    if fix == "parse_datetime":
        return _parse_datetime_fast(s)
    if fix == "cast_numeric":
        return pd.to_numeric(s, errors="coerce")
    if fix == "encode_boolean":
        mapping = {
            "yes": True, "no": False,
            "true": True, "false": False,
            "1": True, "0": False,
        }
        return s.astype(str).str.lower().map(mapping)
    return s


def _apply_type_fix(df: pd.DataFrame, col: str, fix: str) -> pd.DataFrame:
    df[col] = _fix_series(df[col], fix)
    return df


# ── Columnar-execution helpers ────────────────────────────────────────────────

# Strategies whose result depends on row ORDER (rolling windows, decomposition
# phase, directional fills). Only these force the expensive (series, time) pre-sort;
# distribution / groupby-transform strategies (clip_iqr, winsorize, remove,
# mean/median fill) are order-independent. A skip-cleaning recipe has neither, so it
# never pays for any sort beyond the final output ordering.
_ORDER_SENSITIVE_OUTLIER = {"rolling_iqr", "stl_residuals"}
_ORDER_SENSITIVE_MISSING = {"interpolate", "forward_fill", "backward_fill", "flag_and_fill"}


def _needs_time_order(cols_recipe: dict) -> bool:
    for cr in cols_recipe.values():
        if cr.get("action") == "drop":
            continue
        if cr.get("outlier_strategy", "keep") in _ORDER_SENSITIVE_OUTLIER:
            return True
        if cr.get("missing_strategy", "none") in _ORDER_SENSITIVE_MISSING:
            return True
    return False


def _is_active(col: str, cr: dict, ts_col, target_col, group_cols) -> bool:
    """Does this column need to live in pandas? True for columns whose VALUES the
    recipe changes, plus the columns the executor itself reads (timestamp for
    sorting/drop_row, series key for groupby, target for the level guard)."""
    return (col == ts_col or col == target_col or col in group_cols
            or cr.get("type_fix", "none") != "none"
            or cr.get("outlier_strategy", "keep") != "keep"
            or cr.get("missing_strategy", "none") != "none")


def _dedupe_keep_mask(df: pd.DataFrame, passive_cols: list[str],
                      pf: pq.ParquetFile, orig: np.ndarray) -> np.ndarray:
    """drop_duplicates across ALL columns without materializing the passive ones:
    combine per-column row hashes (active columns from the processed frame, passive
    columns taken from Arrow one at a time). 64-bit hashes over a few million rows
    make a false duplicate collision negligible (~1e-6), and pipeline duplicates
    are exact re-ingested rows anyway."""
    combined = pd.util.hash_pandas_object(df, index=False).to_numpy(dtype="uint64")
    if passive_cols:
        idx = pa.array(orig)
        for col in passive_cols:
            taken = pf.read(columns=[col]).column(0).take(idx)
            h = pd.util.hash_pandas_object(taken.to_pandas(),
                                           index=False).to_numpy(dtype="uint64")
            combined = (combined * np.uint64(0x9E3779B97F4A7C15)) ^ h
    return ~pd.Series(combined).duplicated().to_numpy()


def _target_total(df: pd.DataFrame, target_col: str | None) -> float | None:
    """Sum of the confirmed target column (numeric-coerced). Tracked before/after so a
    level-mutating recipe (e.g. clipping a sum-target) shows up as a visible drift in
    the report instead of silently shrinking every downstream forecast."""
    if not target_col or target_col not in df.columns:
        return None
    total = float(pd.to_numeric(df[target_col], errors="coerce").sum())
    return total if np.isfinite(total) else None


# ── Stage 3 executor ──────────────────────────────────────────────────────────

def run(run_id: str) -> dict:
    """Execute the cleaning recipe on the raw parquet — columnar.

    Reads:
      data/raw/{run_id}_raw.parquet
      runs/{run_id}/cleaning_recipe.json
    Writes:
      data/cleaned/{run_id}_cleaned.parquet
      runs/{run_id}/cleaning_report.json
      runs/{run_id}/cleaned_metadata.json

    Row identity is tracked through sort/drops/dedupe as ``orig`` — an index array
    into the raw parquet's row order. Active columns are processed in pandas exactly
    as before; passive columns are aligned to the final row selection/order with one
    Arrow take(orig) each at write time, so they never inflate into Python objects.
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
    cols_recipe: dict = recipe.get("columns", {})
    ts_col: str | None = recipe.get("timestamp_col")
    target_col: str | None = recipe.get("target_col")

    pf = pq.ParquetFile(raw_path)
    all_cols = list(pf.schema_arrow.names)
    rows_before = int(pf.metadata.num_rows)
    group_cols: list[str] = [c for c in (recipe.get("group_cols") or []) if c in all_cols]

    # 1. Column drops = exclusion from the output (nothing to load, mutate or copy).
    removed_cols = [c for c in all_cols if cols_recipe.get(c, {}).get("action") == "drop"]
    kept_cols = [c for c in all_cols if c not in removed_cols]
    active_cols = [c for c in kept_cols
                   if _is_active(c, cols_recipe.get(c, {}), ts_col, target_col, group_cols)]
    passive_cols = [c for c in kept_cols if c not in active_cols]

    # 2. Load active columns ONE at a time, applying the type fix immediately — a cast
    # column stores as 8-byte numeric/datetime instead of Python strings, so peak
    # memory is the compact frame plus a single raw object column in transit.
    data = {}
    for col in active_cols:
        s = pf.read(columns=[col]).column(0).to_pandas()
        fix = cols_recipe.get(col, {}).get("type_fix", "none")
        if fix != "none":
            s = _fix_series(s, fix)
        data[col] = s
    df = pd.DataFrame(data)
    orig = np.arange(rows_before, dtype=np.int64)

    target_total_before = _target_total(df, target_col)

    # 2.5 The timestamp must be real datetimes (string dates sort lexicographically) —
    # backstop for recipes that skipped its parse_datetime fix. The (series, time)
    # pre-sort runs ONLY when an order-sensitive strategy exists: rolling windows,
    # decomposition and directional fills assume time order and contiguous series;
    # distribution/groupby-transform strategies don't, and skip-cleaning recipes have
    # no strategies at all — those keep raw order until the final output sort,
    # saving a full-frame sort copy.
    if ts_col and ts_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = _parse_datetime_fast(df[ts_col])
    if ts_col and ts_col in df.columns and _needs_time_order(cols_recipe):
        order = df.sort_values([*group_cols, ts_col], kind="stable").index.to_numpy()
        df = df.take(order)
        df.index = pd.RangeIndex(len(df))
        orig = orig[order]
    drop_rows_mask = pd.Series(False, index=df.index)

    # 3. Collect outlier-remove rows; apply other outlier strategies in-place.
    # Target-level guard: the sanitizer allows nominally level-neutral repairs
    # (rolling_iqr / stl_residuals) on a sum/mean target, but "level-neutral" is an
    # assumption, not a guarantee — on a target riddled with huge junk sentinels the
    # repair can move the total enormously (observed: rolling_iqr shifted a sum-target
    # +182%, hard-failing the gate with no way forward). So VERIFY: measure the target
    # total around the treatment and revert it when the drift alone would breach the
    # gate's threshold — the run then proceeds with the target untouched, and the
    # reversion is surfaced in the cleaning report instead of a dead-end gate failure.
    target_agg = str(recipe.get("target_agg") or "sum")
    target_outlier_reverted: dict | None = None
    for col, col_rec in cols_recipe.items():
        if col not in df.columns:
            continue
        strategy = col_rec.get("outlier_strategy", "keep")
        if strategy == "remove" and pd.api.types.is_numeric_dtype(df[col]):
            if group_cols:
                # per-series fences — a small series' normal values are outliers only
                # relative to ITS OWN distribution, not the whole panel's
                gb = df.groupby(group_cols, sort=False, dropna=False)
                q1 = gb[col].transform("quantile", 0.25)
                q3 = gb[col].transform("quantile", 0.75)
                iqr = q3 - q1
                mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
                drop_rows_mask = drop_rows_mask | mask.fillna(False)
            else:
                valid = df[col].dropna()
                if len(valid) > 0:
                    q1 = float(np.percentile(valid, 25))
                    q3 = float(np.percentile(valid, 75))
                    iqr = q3 - q1
                    mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
                    drop_rows_mask = drop_rows_mask | mask.reindex(df.index, fill_value=False)
        elif strategy not in ("keep", "remove"):
            guarded = col == target_col and target_agg in ("sum", "mean")
            if guarded:
                before_vals = df[col].copy()
                before_tot = float(pd.to_numeric(before_vals, errors="coerce").sum())
            df = _apply_outlier(df, col, strategy, recipe=recipe, group_cols=group_cols)
            if guarded and np.isfinite(before_tot) and before_tot != 0:
                after_tot = float(pd.to_numeric(df[col], errors="coerce").sum())
                threshold = _target_drift_threshold()
                drift = (abs(after_tot - before_tot) / abs(before_tot) * 100
                         if np.isfinite(after_tot) else float("inf"))
                if drift > threshold:
                    df[col] = before_vals
                    target_outlier_reverted = {
                        "column": col,
                        "strategy": strategy,
                        "would_be_drift_pct": round(drift, 2) if np.isfinite(drift) else None,
                        "threshold_pct": threshold,
                    }

    # 4. Collect missing drop_row rows; apply other missing strategies in-place
    for col, col_rec in cols_recipe.items():
        if col not in df.columns:
            continue
        strategy = col_rec.get("missing_strategy", "none")
        if strategy == "drop_row":
            mask = df[col].isna()
            drop_rows_mask = drop_rows_mask | mask.reindex(df.index, fill_value=False)
        elif strategy != "none":
            df = _apply_missing(df, col, strategy, ts_col=ts_col, group_cols=group_cols)

    # 5. Apply all accumulated row drops at once
    keep = ~drop_rows_mask.to_numpy()
    if not keep.all():
        df = df.loc[keep]
        df.index = pd.RangeIndex(len(df))
        orig = orig[keep]

    # 6. Drop duplicate rows — full-row semantics via combined per-column hashes, so
    # the passive columns participate without being materialized.
    if recipe.get("drop_duplicates", False) and len(df):
        keep = _dedupe_keep_mask(df, passive_cols, pf, orig)
        if not keep.all():
            df = df.loc[keep]
            df.index = pd.RangeIndex(len(df))
            orig = orig[keep]

    # 7. Sort by timestamp. The column must be real datetimes first — string dates sort
    # lexicographically, which is NOT chronological order (caught by the validation
    # gate's monotonicity check). The sanitizer normally forces parse_datetime for the
    # ts col; this is the backstop for recipes that skipped it.
    if recipe.get("sort_by_timestamp") and ts_col and ts_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            df[ts_col] = _parse_datetime_fast(df[ts_col])
            keep = df[ts_col].notna().to_numpy()
            if not keep.all():
                df = df.loc[keep]
                df.index = pd.RangeIndex(len(df))
                orig = orig[keep]
        order = df[ts_col].sort_values(kind="stable", na_position="last").index.to_numpy()
        df = df.take(order)
        df.index = pd.RangeIndex(len(df))
        orig = orig[order]

    rows_after = len(df)
    row_loss_pct = round((rows_before - rows_after) / max(rows_before, 1) * 100, 2)
    target_total_after = _target_total(df, target_col)
    target_drift_pct = None
    if target_total_before and target_total_after is not None:
        target_drift_pct = round(
            (target_total_after - target_total_before) / abs(target_total_before) * 100, 2)

    # 8. Assemble the output: active columns from pandas, passive columns from Arrow
    # via one take() each — computing the cleaned-metadata snapshot on the way so no
    # second pass over the data is needed.
    idx = pa.array(orig)
    extra_cols = [c for c in df.columns if c not in kept_cols]  # flag_and_fill adds
    names, arrays = [], []
    nulls, numeric_variance = {}, {}
    mem_bytes = int(df.memory_usage(deep=True).sum())
    for col in [*kept_cols, *extra_cols]:
        if col in df.columns:
            s = df[col]
            arrays.append(pa.Array.from_pandas(s))
            n_null = int(s.isna().sum())
            if pd.api.types.is_numeric_dtype(s):
                sv = s.dropna()
                numeric_variance[col] = float(sv.std()) if len(sv) > 0 else 0.0
        else:
            arr = pf.read(columns=[col]).column(0).take(idx)
            arrays.append(arr)
            n_null = int(arr.null_count)
            mem_bytes += int(arr.nbytes)
            if pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type):
                try:
                    sd = pc.stddev(arr, ddof=1).as_py()
                    numeric_variance[col] = float(sd) if sd is not None else 0.0
                except pa.ArrowInvalid:
                    numeric_variance[col] = 0.0
        names.append(col)
        nulls[col] = {"count": n_null, "pct": round(n_null / max(rows_after, 1) * 100, 2)}

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_path = CLEANED_DIR / f"{run_id}_cleaned.parquet"
    pq.write_table(pa.table(arrays, names=names), cleaned_path)

    snapshot = {
        "rows": rows_after,
        "cols": len(names),
        "memory_mb": round(mem_bytes / 1e6, 3),
        "nulls": nulls,
        "numeric_variance": numeric_variance,
    }
    (run_dir / "cleaned_metadata.json").write_text(json.dumps(snapshot, indent=2))

    report = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "row_loss_pct": row_loss_pct,
        "target_col": target_col,
        "target_total_before": target_total_before,
        "target_total_after": target_total_after,
        "target_total_drift_pct": target_drift_pct,
        "target_outlier_reverted": target_outlier_reverted,
        "cols_dropped": removed_cols,
        "recipe_applied": recipe,
    }
    (run_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2))

    return {
        "run_id": run_id,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "row_loss_pct": row_loss_pct,
        "target_col": target_col,
        "target_total_before": target_total_before,
        "target_total_after": target_total_after,
        "target_total_drift_pct": target_drift_pct,
        "target_outlier_reverted": target_outlier_reverted,
        "cols_dropped": removed_cols,
        "cleaned_path": str(cleaned_path),
        "snapshot": snapshot,
    }
