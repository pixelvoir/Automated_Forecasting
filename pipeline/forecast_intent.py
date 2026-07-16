"""Stage 2.5: Forecast intent detection — the single user checkpoint's evidence pass.

Runs AFTER pre-clean EDA and BEFORE cleaning. Suggests every forecast-intent choice —
timestamp column, target (+aggregation, including event counts), aggregate-vs-per-series
scope, series group key, exogenous drivers, forecast frequency, horizon — each with a
confidence level. The user confirms everything in the Setup & Cleaning tab; cleaning,
validation and forecast EDA all consume the confirmed intent.

Rule-based only. The optional LLM refinement lives in agents/forecast_intent_agent.py
(same split as cleaning: pipeline computes, agent decides, sanitizer guarantees).

Evidence sources (no cleaned data exists yet):
  runs/{id}/metadata.json                   — Stage 1: schema + inferred dtypes, numeric
                                              stats (coercion-aware), frequency, cardinality
  runs/{id}/cleaning_decision_payload.json  — Stage 2: column_profile, dtype_issues (optional)
  data/raw/{id}_raw.parquet                 — targeted reads: nunique, (key, ts) uniqueness

Output: runs/{id}/forecast_intent.json
Confidence semantics (drives the UI): high — safe to run unattended; medium — pre-selected,
shown normally; low — pre-selected but flagged "please confirm".
"""
import json
import re
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
RAW_DIR = ROOT / "data" / "raw"
CONFIG_PATH = ROOT / "config" / "settings.yaml"

# Coarser frequencies the user may forecast at (data can be resampled down, never up).
_COARSER = {
    "hourly": ["daily", "weekly", "monthly"],
    "daily": ["weekly", "monthly"],
    "weekly": ["monthly", "quarterly"],
    "monthly": ["quarterly", "yearly"],
    "quarterly": ["yearly"],
    "yearly": [],
}
_DEFAULT_HORIZON = {"hourly": 48, "daily": 30, "weekly": 13, "monthly": 12, "quarterly": 8, "yearly": 3}
_FREQ_GRANULARITY = {"hourly": 0, "daily": 1, "weekly": 2, "monthly": 3, "quarterly": 4, "yearly": 5}

# Name heuristics. These only RANK suggestions — the user confirms everything before
# anything heavy runs, so a miss costs one dropdown click.
_MEASURE_HINTS = re.compile(
    r"sales|demand|qty|quantity|volume|revenue|amount|total|litre|liter|kilo|units|sold"
    r"|consumption|usage|load|orders|count|weight|\bkg\b|tonne|\bton\b|gallon|\bwt\b"
    r"|price|cost|value|spend", re.I)
_RATE_HINTS = re.compile(r"pct|percent|rate|ratio|avg|average|aggregated", re.I)
_ID_AUDIT_HINTS = re.compile(
    r"^id$|_id$|^index$|_key$|^key$|code|file_|_file|loaded|created|updated|_at$"
    r"|^year$|^month$|^day$|_year$|_month$", re.I)
_GROUP_HINTS = re.compile(
    r"code|name|nbr|family|category|store|region|society|branch|sku|item|product"
    r"|location|dept|group|type|segment|channel", re.I)
# Event-entity columns whose DISTINCT COUNT per period is a forecastable quantity
# ("number of vet visits per day" = nunique of VISIT ID resampled daily).
_EVENT_HINTS = re.compile(
    r"visit|appointment|appt|order|ticket|case|encounter|admission|booking|receipt"
    r"|claim|transaction|invoice|session|call|request|complaint|incident", re.I)
# Dimension columns that identify WHERE/WHO an event happened — group-key candidates
# for event logs, where (key, ts) uniqueness is meaningless (each row is one event).
# Two tiers: place/entity dimensions (the classic "per facility/region" ask) rank above
# attribute-ish ones (category/status/type), which in turn beat unhinted columns.
_DIMENSION_HINTS_STRONG = re.compile(
    r"center|centre|clinic|branch|store|taluka|region|district|zone|society|city"
    r"|state|area|facility|unit|location", re.I)
_DIMENSION_HINTS_WEAK = re.compile(
    r"category|type|dept|department|doctor|status|team|segment|channel|group", re.I)
# Never useful as drivers or dimensions: contact/free-text/tag identifier columns.
_JUNK_HINTS = re.compile(r"contact|phone|mobile|email|address|remark|comment|symptom|\btag\b", re.I)
# ETL/system timestamp names — never the business/event timestamp.
_AUDIT_TS_TOKENS = ("created", "updated", "loaded", "inserted", "modified", "audit", "batch")


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("forecast_intent", {})


def _f(x, nd: int = 4):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, nd)


def _is_audit_ts(name: str) -> bool:
    n = name.lower()
    return any(t in n for t in _AUDIT_TS_TOKENS)


# ── Timestamp candidates ─────────────────────────────────────────────────────

def _is_time_only(df: pd.DataFrame, col: str, sample_n: int = 5000) -> bool:
    """A 'timestamp' whose values all parse onto a single calendar date is a
    time-of-day-only column (e.g. RECVDTIME='11:05:33' stored beside RECVDDATE —
    a common split-date/time DB schema). pandas stamps bare times with TODAY's date,
    so the parsed span collapses to one day: useless as the forecast timeline (the
    gate's series_length check would count 1 distinct daily period). Detected here
    so it gets demoted and labeled instead of winning the granularity ranking."""
    if col not in df.columns:
        return False
    s = df[col].dropna()
    if len(s) == 0:
        return False
    if len(s) > sample_n:
        s = s.iloc[:: len(s) // sample_n]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, errors="coerce").dropna()
    if len(parsed) < 2:
        return False
    return int(parsed.dt.normalize().nunique()) <= 1


def _timestamp_candidates(meta: dict, df: pd.DataFrame) -> tuple[list[dict], str | None, str]:
    """Rank datetime columns: business/event timestamps above ETL audit ones, then by
    granularity; time-of-day-only columns rank below everything. Multiple business
    candidates (vet data has 3 event timestamps) → the user must pick, so confidence
    drops to low."""
    frequency = meta.get("frequency", {})
    cands = []
    for col in meta.get("datetime_cols", []):
        cands.append({
            "col": col,
            "frequency": frequency.get(col, "unknown"),
            "is_audit": _is_audit_ts(col),
            "is_time_only": _is_time_only(df, col),
        })
    cands.sort(key=lambda c: (c["is_time_only"], c["is_audit"],
                              _FREQ_GRANULARITY.get(c["frequency"], 9)))
    usable = [c for c in cands if not c["is_audit"] and not c["is_time_only"]]
    business = [c for c in cands if not c["is_audit"]]
    suggested = (usable or business or cands)[0]["col"] if cands else None
    if not cands:
        confidence = "low"
    elif len(usable) == 1:
        confidence = "high"
    elif len(usable) == 0:
        confidence = "low"      # only audit/time-only timestamps — user decides
    else:
        confidence = "low"      # several plausible event timestamps — only the user knows
    return cands, suggested, confidence


# ── Target candidates (measures + event counts) ──────────────────────────────

def _is_float_like(item: dict, stats: dict) -> bool:
    """Fractional values mark a continuous quantity (volume/weight) rather than an
    integer attribute/headcount. Covers object-stored numerics via the quartiles."""
    if "float" in str(item.get("dtype_raw", "")).lower():
        return True
    for k in ("median", "q1", "q3"):
        v = stats.get(k)
        try:
            if v is not None and float(v) % 1 != 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _measure_candidates(meta: dict, nunique: pd.Series, n_rows: int,
                        ts_cols: set[str]) -> list[dict]:
    numeric_stats = meta.get("numeric_stats", {})
    out = []
    for item in meta.get("schema", []):
        col = item["col"]
        if col in ts_cols or item.get("dtype_inferred") != "numeric":
            continue
        stats = numeric_stats.get(col, {})
        std = stats.get("std", 0) or 0
        if std <= 0:
            continue  # constant — nothing to forecast
        mean = stats.get("mean")
        cv = abs(std / mean) if mean else None  # scale-free variation
        nu = float(nunique.get(col, 0))
        distinct_pct = nu / max(n_rows, 1) * 100

        score = 0.0
        if _MEASURE_HINTS.search(col):
            score += 2
        if _RATE_HINTS.search(col):
            score -= 2
        if _ID_AUDIT_HINTS.search(col):
            score -= 2
        if _JUNK_HINTS.search(col):
            score -= 3  # contact/tag/free-text names are never the forecast quantity
        if distinct_pct > 95:
            score -= 3

        # Continuous-signal bonuses (all scale/unit-free), hard-capped at +1.75 so an
        # unhinted column can never cross the measure-keyword bar of 2 (which would flip
        # event_log_mode). This is the tiebreaker keyword ties need: a real volume
        # measure (~1M distinct fractional values, healthy CV) now outranks a
        # near-constant headcount attribute that happens to share the keyword — the
        # observed failure was `total_farmers` (distinct 0.04%, CV≈0) beating
        # `total_litre`/`total_kilo` purely by schema column order.
        score += min(1.0, float(np.log10(max(nu, 1))) / 6.0)   # cardinality: 1e6 → +1.0
        if cv is not None:
            score += min(0.5, cv / 2.0)                        # variation: CV ≥ 1 → +0.5
        if _is_float_like(item, stats):
            score += 0.25
        # Near-constant penalty: a column whose values barely vary carries no
        # forecastable flow — it is an attribute, not a measure.
        if cv is not None and cv < 0.05:
            score -= 1.5
        elif cv is not None and cv < 0.15:
            score -= 0.75

        out.append({
            "col": col, "kind": "measure", "score": _f(score, 3),
            "agg_hint": "mean" if _RATE_HINTS.search(col) else "sum",
            "distinct_pct": _f(distinct_pct, 2),
            "cv": _f(cv, 3),
        })
    out.sort(key=lambda c: (-c["score"], -(c["distinct_pct"] or 0), c["col"]))
    return out


def _count_candidates(meta: dict, nunique: pd.Series, n_rows: int,
                      ts_cols: set[str]) -> list[dict]:
    """Columns whose DISTINCT COUNT per period is the forecastable quantity. Any dtype —
    event IDs are routinely strings (vet data's VISIT ID)."""
    out = []
    for item in meta.get("schema", []):
        col = item["col"]
        if col in ts_cols or item.get("dtype_inferred") == "datetime":
            continue
        distinct_pct = float(nunique.get(col, 0)) / max(n_rows, 1) * 100
        event_named = bool(_EVENT_HINTS.search(col))
        if distinct_pct < 30:
            # Low-distinct + event-identity NAME = a RECYCLED serial (observed real
            # case: COMPLAINTNO cycles 1001-9999 daily, so nunique per month
            # saturates at exactly the serial range — a flat, wrong series). Each
            # row is still one event, so the countable aggregate is the ROW count.
            if not event_named:
                continue  # low-cardinality without identity naming — a dimension, not events
            agg_hint, score = "count", 2.5
        elif event_named:
            agg_hint, score = "nunique", 3.0
        elif re.search(r"\b(id|no|number)\b|_id$|_no$", col, re.I):
            agg_hint, score = "nunique", 1.5
        else:
            continue  # high-cardinality but no identity naming — not offered
        out.append({
            "col": col, "kind": "count", "score": score,
            "agg_hint": agg_hint,
            "distinct_pct": _f(distinct_pct, 2),
        })
    out.sort(key=lambda c: (-c["score"], -(c["distinct_pct"] or 0)))
    return out


# ── Group-key candidates ─────────────────────────────────────────────────────

def _panel_group_candidates(df: pd.DataFrame, ts_col: str, nunique: pd.Series,
                            rows_per_ts: float, cfg: dict) -> list[dict]:
    """Panel mode: a good key has ~one row per (group, timestamp). Singles first; pairs
    of the best singles when no single reaches the high bar (Favorita: store × family)."""
    hi = cfg.get("group_uniqueness_high", 0.995)
    singles = []
    for col in df.columns:
        if col == ts_col:
            continue
        # A measure/rate column is never a series key — without this, pair search
        # produced nonsense composites like (societyname, total_cows).
        if _MEASURE_HINTS.search(col) or _RATE_HINTS.search(col):
            continue
        nu = int(nunique.get(col, 0))
        if nu < 2 or nu > rows_per_ts * 5:
            continue
        closeness = abs(np.log(nu / rows_per_ts))
        bonus = -0.5 if _GROUP_HINTS.search(col) else 0.0
        singles.append({"cols": [col], "nunique": nu, "rank": closeness + bonus})
    singles.sort(key=lambda c: c["rank"])

    candidates = []
    for cand in singles[:6]:
        uniq = 1.0 - float(df.duplicated(subset=[*cand["cols"], ts_col]).mean())
        candidates.append({**cand, "uniqueness": round(uniq, 4)})
        if uniq >= hi:
            break

    if not any(c["uniqueness"] >= hi for c in candidates):
        for a, b in combinations(singles[:4], 2):
            cols = a["cols"] + b["cols"]
            uniq = 1.0 - float(df.duplicated(subset=[*cols, ts_col]).mean())
            candidates.append({
                "cols": cols, "nunique": int(df.groupby(cols, observed=True).ngroups),
                "rank": 0.0, "uniqueness": round(uniq, 4),
            })
            if uniq >= hi:
                break

    candidates.sort(key=lambda c: -c["uniqueness"])
    return [{k: v for k, v in c.items() if k != "rank"} for c in candidates[:5]]


def _dimension_group_candidates(meta: dict, nunique: pd.Series,
                                max_card: int) -> list[dict]:
    """Event-log mode: (key, ts) uniqueness is meaningless when every row is one event.
    Candidates are categorical dimensions (facility / area / category …), name-ranked."""
    out = []
    for item in meta.get("schema", []):
        col = item["col"]
        if item.get("dtype_inferred") == "datetime" or _JUNK_HINTS.search(col):
            continue
        nu = int(nunique.get(col, 0))
        if nu < 2 or nu > max_card:
            continue
        strong = bool(_DIMENSION_HINTS_STRONG.search(col))
        weak = bool(_DIMENSION_HINTS_WEAK.search(col))
        if not (strong or weak) and _ID_AUDIT_HINTS.search(col):
            continue  # audit/id columns are never dimensions
        tier = 0 if strong else (1 if weak else 2)
        out.append({"cols": [col], "nunique": nu, "uniqueness": None,
                    "dimension_hint": strong or weak, "_tier": tier})
    # place/entity hints first, then attribute hints, then unhinted; within a tier
    # prefer non-degenerate splits (binary flags last) and fewer values otherwise
    out.sort(key=lambda c: (c["_tier"], c["nunique"] < 3, c["nunique"]))
    return [{k: v for k, v in c.items() if k != "_tier"} for c in out[:8]]


# ── Stage 2.5 entry point ────────────────────────────────────────────────────

def detect(run_id: str) -> dict:
    """Rule-based intent suggestions. Reads Stage 1/2 outputs + the RAW parquet
    (cleaning has not run yet). Writes runs/{id}/forecast_intent.json."""
    cfg = _load_config()
    run_dir = RUNS_DIR / run_id
    meta_path = run_dir / "metadata.json"
    parquet = RAW_DIR / f"{run_id}_raw.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"Stage 1 metadata not found: {meta_path}.")
    if not parquet.exists():
        raise FileNotFoundError(f"Raw parquet not found: {parquet}.")

    meta = json.loads(meta_path.read_text())
    n_rows = meta.get("shape", {}).get("rows", 0)

    df = pd.read_parquet(parquet)
    nunique = df.nunique()

    # ── Timestamp ────────────────────────────────────────────────────────────
    ts_cands, ts_suggested, ts_conf = _timestamp_candidates(meta, df)
    ts_cols = {c["col"] for c in ts_cands}

    # Panel evidence needs parsed timestamps of the suggested column
    n_ts = 0
    data_end = None
    if ts_suggested and ts_suggested in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # dayfirst format guesses on raw string dates
            ts_parsed = pd.to_datetime(df[ts_suggested], errors="coerce")
        n_ts = int(ts_parsed.nunique())
        # Last actual timestamp — lets the UI offer a "forecast until <date>" picker.
        # Reflects the SUGGESTED ts column; the integer horizon stays canonical, so a
        # user picking a different column can never corrupt anything downstream.
        _mx = ts_parsed.max()
        data_end = str(_mx) if pd.notna(_mx) else None
    rows_per_ts = n_rows / max(n_ts, 1)
    dup_ts_frac = 1.0 - n_ts / max(n_rows, 1)
    is_panel = dup_ts_frac > cfg.get("panel_dup_ts_threshold", 0.05)

    # ── Target: measures + event counts ─────────────────────────────────────
    measures = _measure_candidates(meta, nunique, n_rows, ts_cols)
    counts = _count_candidates(meta, nunique, n_rows, ts_cols)
    # Event-log mode: no convincing numeric measure but a plausible event-ID column —
    # the forecastable quantity is events per period (the vet-data case). The gate is
    # KEYWORD-based (a measure-named column with a non-degenerate score), deliberately
    # independent of the continuous-signal bonuses/penalties so their tuning can never
    # flip a measure dataset into count mode.
    has_keyword_measure = any(
        _MEASURE_HINTS.search(m["col"]) and (m["score"] or 0) > 0 for m in measures)
    event_log_mode = not has_keyword_measure and len(counts) > 0

    candidates = (counts + measures) if event_log_mode else (measures + counts)
    suggested_target = candidates[0]["col"] if candidates else None
    suggested_agg = candidates[0]["agg_hint"] if candidates else "sum"
    if not candidates:
        target_conf = "low"
    elif len(candidates) == 1 or candidates[0]["score"] - candidates[1]["score"] >= 2:
        target_conf = "high"
    elif candidates[0]["score"] - candidates[1]["score"] >= 1:
        target_conf = "medium"
    else:
        target_conf = "low"  # ties (eight total_* measures) — only the user knows

    # ── Group key ────────────────────────────────────────────────────────────
    if event_log_mode:
        group_cands = _dimension_group_candidates(
            meta, nunique, max_card=cfg.get("dimension_max_cardinality", 10_000))
        group_conf = "low"  # dimensions are a business choice, not a data property
    elif is_panel and ts_suggested:
        group_cands = _panel_group_candidates(df, ts_suggested, nunique, rows_per_ts, cfg)
        best_u = group_cands[0]["uniqueness"] if group_cands else 0.0
        if best_u >= cfg.get("group_uniqueness_high", 0.995):
            group_conf = "high"
        elif best_u >= cfg.get("group_uniqueness_medium", 0.95):
            group_conf = "medium"
        else:
            group_conf = "low"
    else:
        group_cands, group_conf = [], "high"  # single series — confidently no grouping

    # ── Exogenous: numeric measures that aren't IDs / audit / junk. score >= 0 keeps
    # rate columns (fat %) as plausible drivers while dropping negatively-scored ones
    # (phone numbers, tag IDs). Same-unit variants of the target (litre vs kilo) survive
    # here — the LLM refinement and Stage 4's VIF are the leakage guards.
    exog_cands = [
        m["col"] for m in measures
        if m["score"] >= 0
        and not _ID_AUDIT_HINTS.search(m["col"])
        and not _JUNK_HINTS.search(m["col"])
        and (m["distinct_pct"] or 0) <= 95
    ]

    # ── Frequency + horizon ──────────────────────────────────────────────────
    data_freq = meta.get("frequency", {}).get(ts_suggested, "daily") if ts_suggested else "daily"
    freq_options = [data_freq] + _COARSER.get(data_freq, [])
    # Event timestamps carry time-of-day, so Stage 1 reads them "hourly" — but nobody
    # forecasts vet visits per hour by default. Suggest daily; the user can override.
    suggested_freq = "daily" if (event_log_mode and data_freq == "hourly"
                                 and "daily" in freq_options) else data_freq
    freq_conf = "medium" if suggested_freq != data_freq else "high"

    intent = {
        "run_id": run_id,
        "n_rows": n_rows,
        "data_end": data_end,
        "evidence": {
            "is_panel": is_panel,
            "event_log_mode": event_log_mode,
            "duplicate_ts_fraction": _f(dup_ts_frac, 4),
            "rows_per_timestamp": _f(rows_per_ts, 1),
            "n_timestamps": n_ts,
        },
        "timestamp": {
            "candidates": ts_cands,
            "suggested": ts_suggested,
            "confidence": ts_conf,
        },
        "target": {
            # capped generously so real measures can't fall off the UI dropdown
            "candidates": candidates[:int(cfg.get("max_target_candidates", 30))],
            "suggested": suggested_target,
            "suggested_agg": suggested_agg,
            "confidence": target_conf,
        },
        "scope": {
            "suggested": "aggregate",
            "confidence": "high" if not (is_panel or event_log_mode) else "medium",
        },
        "group": {
            "candidates": group_cands,
            "suggested": group_cands[0]["cols"] if group_cands else [],
            "confidence": group_conf,
            # every column's cardinality so the UI can offer ANY column as group key
            "column_nunique": {c: int(nunique[c]) for c in df.columns if c not in ts_cols},
        },
        "exogenous": {
            "candidates": exog_cands,
            "suggested": [c for c in exog_cands if c != suggested_target],
        },
        "frequency": {
            "data": data_freq,
            "options": freq_options,
            "suggested": suggested_freq,
            "confidence": freq_conf,
        },
        "horizon": {"suggested": _DEFAULT_HORIZON.get(suggested_freq, 12)},
        "llm": None,  # filled by agents/forecast_intent_agent.refine()
    }

    (run_dir / "forecast_intent.json").write_text(json.dumps(intent, indent=2, default=str))
    return {"run_id": run_id, "status": "completed", "intent": intent}
