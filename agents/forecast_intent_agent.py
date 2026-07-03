"""Stage 2.5 intent agent: LLM refinement of the rule-based forecast-intent suggestions.

The rules in pipeline/forecast_intent.py compute the evidence; this agent makes the
semantic call the rules can't ("VISIT ID counted per day is the vet-visits forecast,
grouped by veterinary center") from column NAMES + statistics only — no raw values ever.

Same architecture as the cleaning agent: pydantic-validated enum picks, a deterministic
field-level sanitizer (an invalid field falls back to the rule suggestion for THAT field,
not the whole response), and llm_client.LLMError → rules-only. The refined suggestions
are written back into runs/{id}/forecast_intent.json; the user still confirms everything
in the Setup & Cleaning tab before anything heavy runs.
"""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError, field_validator

from agents import llm_client
from agents.llm_client import LLMError

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"


# ── Pydantic output schema ────────────────────────────────────────────────────

class IntentSuggestion(BaseModel):
    timestamp_col: str
    target_col: str
    target_agg: Literal["sum", "mean", "nunique"]
    scope: Literal["aggregate", "per_series"]
    group_cols: list[str] = []
    exog_cols: list[str] = []
    forecast_frequency: str
    horizon: int
    rationale: str
    confidence: Literal["high", "medium", "low"]

    # Models write count aliases freely — normalize instead of rejecting.
    @field_validator("target_agg", mode="before")
    @classmethod
    def _count_aliases(cls, v):
        if isinstance(v, str) and v.lower() in ("count", "count_distinct", "distinct_count"):
            return "nunique"
        return v

    @field_validator("group_cols", "exog_cols", mode="before")
    @classmethod
    def _none_is_empty(cls, v):
        return v or []


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Task:
You are a forecasting setup expert. You receive a JSON description of a dataset
(column names, dtypes, statistics — never raw values) plus rule-based suggestions,
and you must decide the forecast intent: what quantity should be forecast, over what
time column, at what granularity.

Return STRICT JSON with exactly these fields:
{
  "timestamp_col":      "<column>",
  "target_col":         "<column>",
  "target_agg":         "sum" | "mean" | "nunique",
  "scope":              "aggregate" | "per_series",
  "group_cols":         ["<column>", ...] or [],
  "exog_cols":          ["<column>", ...] or [],
  "forecast_frequency": "<one of the payload's frequency.options>",
  "horizon":            <integer number of periods>,
  "rationale":          "<1-2 short sentences: what is being forecast and why these choices>",
  "confidence":         "high" | "medium" | "low"
}

Rules:
- timestamp_col: the BUSINESS/EVENT time column, never an ETL/audit one
  (loaded_at, created_on, inserted_at ...). Pick from timestamp_candidates.
- target_col + target_agg define the forecast quantity:
  * numeric measure column (sales, litres, demand) -> "sum" (totals) or "mean" (rates/percentages)
  * event/entity ID column (visit id, order id, ticket no) -> "nunique"
    = number of distinct events per period. Use this for transactional/event-log
    data with no numeric measure — e.g. "number of visits per day".
  * NEVER pick phone numbers, addresses, tags, audit columns or free text as target.
- scope: "aggregate" forecasts one overall series; "per_series" additionally profiles
  each series separately. Prefer "aggregate" unless the data is clearly many parallel
  series AND per-series forecasts are the obvious business need.
- group_cols: the column(s) that identify one series (facility, store, region ...).
  Give them even when scope is "aggregate" (they may be used later). [] if none apply.
- exog_cols: numeric driver columns whose FUTURE values would be known at forecast
  time. Never include unit-variants or components of the target itself (leakage),
  IDs, or contact/free-text columns. [] is a perfectly good answer.
- forecast_frequency: must come from frequency.options. Event timestamps often read
  as "hourly" but businesses forecast per day/week — choose what makes business sense.
- horizon: a sensible default number of periods for that frequency
  (daily ~30, weekly ~13, monthly ~12, hourly ~48).
- confidence: "high" only when the column names make the intent unambiguous.
- Return pure JSON. No markdown, no extra text.
"""


# ── Field-level sanitizer ─────────────────────────────────────────────────────

def _sanitize(s: IntentSuggestion, intent: dict) -> tuple[dict, list[str]]:
    """Validate each LLM-chosen field against the dataset; an invalid field reverts to
    the rule suggestion for that field only. Returns (clean dict, list of overrides)."""
    overrides: list[str] = []
    ts_cands = {c["col"] for c in intent["timestamp"]["candidates"]}
    all_cols = set(intent["group"]["column_nunique"]) | ts_cands
    nunique = intent["group"]["column_nunique"]
    n_rows = max(intent.get("n_rows", 1), 1)
    target_cands = {c["col"]: c for c in intent["target"]["candidates"]}
    freq_options = intent["frequency"]["options"]

    out = s.model_dump()

    if out["timestamp_col"] not in ts_cands:
        out["timestamp_col"] = intent["timestamp"]["suggested"]
        overrides.append("timestamp_col")

    if out["target_col"] not in all_cols:
        out["target_col"] = intent["target"]["suggested"]
        out["target_agg"] = intent["target"]["suggested_agg"]
        overrides.append("target_col")
    else:
        cand = target_cands.get(out["target_col"])
        distinct_pct = (cand or {}).get("distinct_pct") or (
            nunique.get(out["target_col"], 0) / n_rows * 100)
        kind = (cand or {}).get("kind")
        # An ID-like target summed/averaged is meaningless — counting it is the intent.
        if (kind == "count" or distinct_pct >= 80) and out["target_agg"] != "nunique":
            out["target_agg"] = "nunique"
            overrides.append("target_agg")

    valid_groups = [
        c for c in out["group_cols"]
        if c in all_cols and c != out["timestamp_col"]
        and 2 <= nunique.get(c, 0) <= 50_000
    ]
    if valid_groups != out["group_cols"]:
        overrides.append("group_cols")
    out["group_cols"] = valid_groups
    if out["scope"] == "per_series" and not valid_groups:
        out["scope"] = "aggregate"
        overrides.append("scope")

    exog_allowed = set(intent["exogenous"]["candidates"])
    valid_exog = [c for c in out["exog_cols"]
                  if c in exog_allowed and c != out["target_col"]]
    if valid_exog != out["exog_cols"]:
        overrides.append("exog_cols")
    out["exog_cols"] = valid_exog

    if out["forecast_frequency"] not in freq_options:
        out["forecast_frequency"] = intent["frequency"]["suggested"]
        overrides.append("forecast_frequency")

    try:
        out["horizon"] = max(1, min(int(out["horizon"]), 1000))
    except (TypeError, ValueError):
        out["horizon"] = intent["horizon"]["suggested"]
        overrides.append("horizon")

    return out, overrides


def _build_payload(intent: dict, run_dir: Path) -> dict:
    """Statistics + names only (~1.5KB). column_profile comes from Stage 2 when present."""
    profile = {}
    dp_path = run_dir / "cleaning_decision_payload.json"
    if dp_path.exists():
        profile = json.loads(dp_path.read_text()).get("column_profile", {})
    return {
        "n_rows": intent.get("n_rows"),
        "evidence": intent.get("evidence"),
        "frequency": {k: intent["frequency"][k] for k in ("data", "options")},
        "timestamp_candidates": intent["timestamp"]["candidates"],
        "target_candidates": intent["target"]["candidates"],
        "group_candidates": intent["group"]["candidates"],
        "exogenous_candidates": intent["exogenous"]["candidates"],
        "column_profile": profile or {
            c: {"nunique": n} for c, n in intent["group"]["column_nunique"].items()
        },
        "rule_suggestions": {
            "timestamp_col": intent["timestamp"]["suggested"],
            "target_col": intent["target"]["suggested"],
            "target_agg": intent["target"]["suggested_agg"],
            "scope": intent["scope"]["suggested"],
            "group_cols": intent["group"]["suggested"],
            "exog_cols": intent["exogenous"].get("suggested",
                                                 intent["exogenous"]["candidates"]),
            "forecast_frequency": intent["frequency"]["suggested"],
            "horizon": intent["horizon"]["suggested"],
        },
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def refine(run_id: str, use_llm: bool = True) -> dict:
    """Refine the rule suggestions in forecast_intent.json with an LLM pass (in place).

    On LLM failure the rule suggestions stand and the error is recorded — the pipeline
    never blocks on the model. Returns the updated intent dict.
    """
    run_dir = RUNS_DIR / run_id
    intent_path = run_dir / "forecast_intent.json"
    if not intent_path.exists():
        raise FileNotFoundError(
            f"Intent file not found: {intent_path}. Run forecast_intent.detect() first.")
    intent = json.loads(intent_path.read_text())

    llm_cfg = llm_client._load_llm_config()
    llm_model = f"{llm_cfg.get('provider', '?')}/{llm_cfg.get('model', '?')}"
    llm_info: dict = {"suggested_by": "rules", "model": llm_model if use_llm else None,
                      "rationale": None, "error": None, "response": None,
                      "sanitizer_overrides": []}

    if use_llm:
        payload = _build_payload(intent, run_dir)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Here is the dataset description:\n\n" + json.dumps(payload, indent=2)
                + "\n\nReturn the forecast intent JSON."
            )},
        ]
        try:
            raw = llm_client.call(messages, require_json=True)
            llm_info["response"] = json.loads(json.dumps(raw))
            suggestion = IntentSuggestion.model_validate(raw)
            clean, overrides = _sanitize(suggestion, intent)

            intent["timestamp"]["suggested"] = clean["timestamp_col"]
            intent["target"]["suggested"] = clean["target_col"]
            intent["target"]["suggested_agg"] = clean["target_agg"]
            intent["scope"]["suggested"] = clean["scope"]
            intent["group"]["suggested"] = clean["group_cols"]
            intent["exogenous"]["suggested"] = clean["exog_cols"]
            intent["frequency"]["suggested"] = clean["forecast_frequency"]
            intent["horizon"]["suggested"] = clean["horizon"]
            # LLM-confirmed fields inherit its confidence — dampened to at most one
            # level above the rule confidence (small models are overconfident on
            # genuinely ambiguous evidence like an 8-way measure tie). Sanitizer
            # overrides always drop to low.
            _rank = {"low": 0, "medium": 1, "high": 2}
            _name = {0: "low", 1: "medium", 2: "high"}
            conf = clean["confidence"]
            for section, field in [("timestamp", "timestamp_col"), ("target", "target_col"),
                                   ("scope", "scope"), ("group", "group_cols"),
                                   ("frequency", "forecast_frequency")]:
                if field in overrides:
                    intent[section]["confidence"] = "low"
                else:
                    rule_conf = _rank.get(intent[section].get("confidence", "low"), 0)
                    intent[section]["confidence"] = _name[
                        min(_rank.get(conf, 0), rule_conf + 1)]
            llm_info.update({
                "suggested_by": "llm",
                "rationale": clean["rationale"],
                "sanitizer_overrides": overrides,
            })
        except (LLMError, ValidationError, Exception) as exc:  # noqa: BLE001
            llm_info["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[forecast_intent_agent] LLM unavailable or invalid "
                  f"({llm_info['error']}). Rule suggestions stand.")

    intent["llm"] = llm_info
    intent_path.write_text(json.dumps(intent, indent=2, default=str))
    return {"run_id": run_id, "status": "completed", "intent": intent}
