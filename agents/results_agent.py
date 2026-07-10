"""Stage 8: Results forecast report — LLM-written markdown for ONE trained model.

Audience is a NON-TECHNICAL business reader: the report is about the FORECAST (expected
amounts in the target's real-world units, growth vs recent history, best/worst case,
plain caveats) — deliberately NOT about how it was made. No EDA statistics, no model
rankings, no metric jargon; the payload carries only what-was-forecast + the predicted
numbers + recent history + a plain-terms accuracy figure.

Opt-in and LLM-only by design: there is NO rule-based fallback — an LLMError propagates
to the route (HTTP 502) and the UI shows it verbatim. Nothing is sent anywhere until the
user explicitly clicks Generate.

What gets sent (user-approved policy, 2026-07-10): DERIVED data only — column names,
thinned forecast arrays, history/forecast totals, and per-series SUMMARIES (series names
+ totals, never row-level dumps) — and NEVER raw dataset rows. All numbers the narrative
cites are computed here in Python and handed to the model; the prompt forbids inventing
values.

Output is prose markdown (``require_json=False``), not JSON: nothing structured needs to
be parsed back out, and json-mode measurably degrades long-form narrative quality while
reintroducing the invalid-JSON failure mode. Light shape validation (length + headings)
replaces schema validation.

Uses the ``llm_report`` settings profile (its own provider/model/key; falls back to the
main ``llm`` block when the profile is absent).

Reads:  runs/{id}/training_report.json, forecast_user_selections.json,
        data/features/{id}_model_frame.parquet (recent-history figures),
        models/{id}/series_forecasts.parquet (per-series runs)
Writes: runs/{id}/results_report.json
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from agents import llm_client
from agents.llm_client import LLMError

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
MODELS_DIR = ROOT / "models"
FEATURES_DIR = ROOT / "data" / "features"

_THIN_POINTS = 12      # keep first+last N points of any long array in the payload
_TOP_SERIES = 20       # per-series summary caps (top by volume + top movers)

_SYSTEM_PROMPT = """You are writing a FORECAST REPORT for a business reader with no
data-science background. They automated their forecasting and only care about one thing:
what the forecast says. You receive a JSON payload describing what was forecast and the
predicted numbers.

Write a plain-markdown report (no code fences, no JSON) with EXACTLY these ## sections,
in this order:

## Summary
## What was forecast
## The numbers ahead
## Compared with recent history
## Best and worst case
## Things to keep in mind

If the payload contains a `per_series` object, insert one extra section before
"Things to keep in mind":
## Breakdown by series

Rules:
- Read the target column's name and write in its real-world units — e.g. a column named
  `total_litre` is litres of milk, `material_quantity_kg` is kilograms of material,
  `VISIT ID` counted per day is number of visits. Infer the business quantity from the
  name and always phrase amounts in those terms, with the actual dates.
- ABSOLUTELY NO technical jargon: never mention MASE, sMAPE, backtesting, cross-
  validation, models' internal names/parameters, seasonality strength, stationarity,
  entropy, quantiles, conformal intervals, EDA, or pipeline stages. The one allowed
  technical fact: the forecasting method's display name, mentioned at most once.
- "The numbers ahead": the immediate next period, the total over the whole forecast
  window, and the highest/lowest points with their dates (all provided in `forecast`
  and `derived` — cite ONLY numbers present in the payload, never invent or
  extrapolate).
- "Compared with recent history": use `derived.pct_change` and the recent-history
  figures — say plainly whether things are expected to grow, shrink or stay level.
- "Best and worst case": translate the uncertainty band (`lo95`/`hi95`) into plain
  language — "could be as low as X or as high as Y" — and note the range widens the
  further out the forecast goes.
- "Things to keep in mind": 2-4 plain-language caveats — the forecast is based on
  history up to the given end date; `accuracy.typical_error_pct` is roughly how far
  off similar predictions were in testing (phrase it as "past predictions of this kind
  were typically within about N% of what actually happened"); unusual events the data
  can't know about (strikes, weather, policy changes) are not included.
- Format large numbers readably (e.g. 95.8 million kg, not 95802681.4282).
- 300-700 words. Clear, direct, confident. No filler, no marketing tone."""


# ── Payload helpers ───────────────────────────────────────────────────────────

def _thin(arr, n=_THIN_POINTS):
    """First + last n items of a long list (the derived numbers carry exact totals, the
    thinned array is just trajectory shape). Short arrays pass through untouched."""
    if not isinstance(arr, list) or len(arr) <= 2 * n:
        return arr
    return arr[:n] + arr[-n:]


def _derived_numbers(entry, report, run_id) -> dict:
    """The exact Python math the LLM narrates (next period, horizon total, growth vs
    trailing history, widest uncertainty band). Prefers the trainer's target_insights
    (present on new reports); falls back to recomputing history from the model frame."""
    insights = entry.get("target_insights") or _recompute_insights(entry, report, run_id)
    return dict(insights or {})


def _recompute_insights(entry, report, run_id):
    """target_insights fallback for reports written before the field existed."""
    fc = entry.get("forecast") or {}
    ds, yhat = fc.get("ds") or [], [v for v in (fc.get("yhat") or []) if v is not None]
    if not ds or not yhat:
        return None
    horizon = int(report.get("horizon") or len(yhat))
    total = float(np.nansum(yhat))
    trailing = None
    frame_path = FEATURES_DIR / f"{run_id}_model_frame.parquet"
    if frame_path.exists():
        try:
            frame = pd.read_parquet(frame_path, columns=["ds", "y"])
            hist = frame.groupby("ds")["y"].sum().sort_index()
            if len(hist) >= horizon:
                trailing = float(hist.tail(horizon).sum())
        except Exception:
            trailing = None
    pct = (round((total - trailing) / abs(trailing) * 100, 2)
           if trailing not in (None, 0) else None)
    widest = None
    if fc.get("lo95") and fc.get("hi95"):
        widths = [h - l for l, h in zip(fc["lo95"], fc["hi95"])
                  if l is not None and h is not None]
        widest = round(max(widths), 4) if widths else None
    return {"next_ds": str(ds[0]), "next_value": round(float(yhat[0]), 4),
            "horizon_total": round(total, 4),
            "trailing_total": round(trailing, 4) if trailing is not None else None,
            "pct_change": pct, "widest_interval_95": widest}


def _series_summary(run_id: str, model_id: str) -> dict | None:
    """Per-series summaries from series_forecasts.parquet — totals and % changes per
    series NAME only (top by volume + biggest movers), never row-level dumps."""
    pq = MODELS_DIR / run_id / "series_forecasts.parquet"
    if not pq.exists():
        return None
    try:
        sf = pd.read_parquet(pq)
    except Exception:
        return None
    sf = sf[sf["model"] == model_id]
    if sf.empty:
        return None
    fc = sf[sf["kind"] == "forecast"].groupby("unique_id")["predicted"].agg(["sum", "count"])
    bt = sf[sf["kind"] == "backtest"].groupby("unique_id")[["actual", "predicted"]].mean()
    rows = []
    for uid, r in fc.iterrows():
        fc_mean = r["sum"] / max(r["count"], 1)
        bt_actual_mean = float(bt.loc[uid, "actual"]) if uid in bt.index else None
        pct = (round((fc_mean - bt_actual_mean) / abs(bt_actual_mean) * 100, 1)
               if bt_actual_mean not in (None, 0) and np.isfinite(bt_actual_mean) else None)
        rows.append({"series": str(uid), "forecast_total": round(float(r["sum"]), 2),
                     "forecast_mean_per_period": round(float(fc_mean), 2),
                     "backtest_actual_mean_per_period":
                         round(bt_actual_mean, 2) if bt_actual_mean is not None else None,
                     "pct_change_vs_backtest": pct})
    by_volume = sorted(rows, key=lambda x: -(x["forecast_total"] or 0))[:_TOP_SERIES]
    movers = sorted([x for x in rows if x["pct_change_vs_backtest"] is not None],
                    key=lambda x: -abs(x["pct_change_vs_backtest"]))[:_TOP_SERIES]
    return {"n_series": int(len(rows)),
            "note": ("Per-series comparisons are mean-per-period: forecast mean vs the "
                     "backtest window's actual mean."),
            "top_by_volume": by_volume, "biggest_movers": movers}


def _history_summary(run_id: str, horizon: int) -> dict | None:
    """Recent-history figures in the target's own units, from the exact modeling frame —
    what the reader compares the forecast against."""
    frame_path = FEATURES_DIR / f"{run_id}_model_frame.parquet"
    if not frame_path.exists():
        return None
    try:
        frame = pd.read_parquet(frame_path, columns=["ds", "y"])
        hist = frame.groupby("ds")["y"].sum().sort_index()
    except Exception:
        return None
    if hist.empty:
        return None
    recent = hist.tail(horizon)
    return {
        "history_start": str(hist.index[0]), "history_end": str(hist.index[-1]),
        "n_periods": int(len(hist)),
        "last_period_value": round(float(hist.iloc[-1]), 2),
        "recent_mean_per_period": round(float(recent.mean()), 2),
        "recent_total": round(float(recent.sum()), 2),
        "recent_window_periods": int(len(recent)),
    }


def _forecast_block(fc: dict) -> dict:
    """The predicted numbers plus their peak/low, computed here so the LLM never has to
    do arithmetic."""
    ds, yhat = fc.get("ds") or [], fc.get("yhat") or []
    pairs = [(d, v) for d, v in zip(ds, yhat) if v is not None]
    peak = max(pairs, key=lambda p: p[1]) if pairs else None
    low = min(pairs, key=lambda p: p[1]) if pairs else None
    return {
        "ds": _thin(ds),
        "yhat": _thin(yhat),
        "lo95": _thin(fc.get("lo95") or []),
        "hi95": _thin(fc.get("hi95") or []),
        "highest_period": {"ds": str(peak[0]), "value": round(float(peak[1]), 2)} if peak else None,
        "lowest_period": {"ds": str(low[0]), "value": round(float(low[1]), 2)} if low else None,
        "note": (f"Arrays thinned to first/last {_THIN_POINTS} points when longer; "
                 "the `derived` object carries the exact totals."),
    }


def _build_payload(run_id: str, model_id: str, report: dict, entry: dict) -> dict:
    """Business-report payload: what was forecast + the predicted numbers + recent
    history + plain-terms accuracy. Deliberately NO pipeline internals (EDA statistics,
    model rankings, CV plans, hyperparameters) — the reader is non-technical and the
    report is about the FORECAST, not how it was made."""
    run_dir = RUNS_DIR / run_id
    sel_path = run_dir / "forecast_user_selections.json"
    selections = json.loads(sel_path.read_text()) if sel_path.exists() else {}

    horizon = int(report.get("horizon") or 0)
    metrics = entry.get("metrics") or {}
    baseline_mase, model_mase = None, metrics.get("mase")
    for e in report.get("results") or []:
        if e.get("model") == report.get("baseline") and not e.get("error"):
            baseline_mase = (e.get("metrics") or {}).get("mase")

    fc = entry.get("forecast") or {}
    payload = {
        "task": ("Write the forecast report. Infer the real-world quantity and units "
                 "from the target column name and phrase everything in those terms."),
        "what_was_forecast": {
            "target_column": selections.get("target_col"),
            "aggregation_per_period": selections.get("agg"),
            "frequency": report.get("frequency"),
            "horizon_periods": horizon,
            "series_key_columns": selections.get("group_cols") or [],
            "external_driver_columns": selections.get("exog_cols") or [],
            "forecasting_method": entry.get("label", model_id),
        },
        "recent_history": _history_summary(run_id, horizon),
        "forecast": _forecast_block(fc),
        "derived": _derived_numbers(entry, report, run_id),
        "accuracy": {
            "typical_error_pct": metrics.get("smape"),
            "better_than_naive_repeat_of_last_season": (
                bool(model_mase < baseline_mase)
                if model_mase is not None and baseline_mase is not None else None),
        },
    }
    if report.get("has_series_forecasts"):
        payload["per_series"] = _series_summary(run_id, model_id)
    return payload


# ── Output validation ─────────────────────────────────────────────────────────

def _validate_markdown(text: str) -> str:
    """LLM-only feature: junk output is an error, never a silent fallback."""
    md = (text or "").strip()
    # Reasoning models (qwen3, deepseek-r1) prepend a <think>…</think> block — internal
    # monologue, not report content.
    md = re.sub(r"<think>.*?</think>", "", md, flags=re.DOTALL).strip()
    if md.startswith("```"):
        md = md.strip("`").strip()
        if md.lower().startswith("markdown"):
            md = md[len("markdown"):].strip()
    n_headings = sum(1 for ln in md.splitlines() if ln.lstrip().startswith("#"))
    if len(md) < 400 or n_headings < 2:
        raise LLMError(
            f"Report response too short or unstructured "
            f"({len(md)} chars, {n_headings} headings) — not saving it.")
    return md


# ── Entry point ───────────────────────────────────────────────────────────────

def run(run_id: str, model_id: str) -> dict:
    run_dir = RUNS_DIR / run_id
    report_path = run_dir / "training_report.json"
    if not report_path.exists():
        raise FileNotFoundError("training_report.json missing — train a model first.")
    report = json.loads(report_path.read_text())
    entry = next((e for e in report.get("results") or [] if e.get("model") == model_id), None)
    if entry is None or entry.get("error"):
        raise ValueError(f"Model '{model_id}' was not successfully trained in this run.")

    payload = _build_payload(run_id, model_id, report, entry)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, indent=2, default=str)},
    ]
    raw = llm_client.call(messages, require_json=False, profile="report")  # LLMError propagates
    markdown = _validate_markdown(raw.get("text", ""))

    out = {
        "run_id": run_id,
        "model": model_id,
        "model_label": entry.get("label", model_id),
        "markdown": markdown,
        "llm": {"model": llm_client.describe("report"), "profile": "report"},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / "results_report.json").write_text(json.dumps(out, indent=2))
    return out
