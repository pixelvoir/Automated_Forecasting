"""Stage 5 agent: LLM model selection on top of the deterministic rule engine.

pipeline/rule_engine.py ALWAYS runs first and produces a pick + full reasoning trace.
When use_llm is on, the LLM receives the Stage 4 statistics, the derived decision
features, the eligible model cards and the rule suggestion, and may override the pick —
a field-level sanitizer reverts anything invalid (unknown/ineligible model, mismatched
category) to the rule engine's answer for that field only, and LLM confidence is
dampened to at most one level above the rule confidence. LLM failure → the rule pick
stands and the error is recorded for the UI. Same architecture as the intent agent.

Writes runs/{id}/model_selection.json — the single file Stage 6 (feature engineering)
and Stage 7 (training) read: model + runner_up + training_hints.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError, field_validator

from agents import llm_client
from agents.llm_client import LLMError
from pipeline import rule_engine

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

_RANK = {"low": 0, "medium": 1, "high": 2}
_NAME = {0: "low", 1: "medium", 2: "high"}

# Common aliases small models produce — normalize instead of rejecting.
_MODEL_ALIASES = {
    "arima": "auto_arima", "sarima": "auto_arima", "sarimax": "auto_arima",
    "autoarima": "auto_arima", "ets": "auto_ets", "autoets": "auto_ets",
    "exponential_smoothing": "auto_ets", "theta": "auto_theta", "autotheta": "auto_theta",
    "lgbm": "lightgbm", "light_gbm": "lightgbm", "gbm": "lightgbm",
    "croston_optimized": "croston", "crostonoptimized": "croston",
    "naive": "seasonal_naive", "seasonalnaive": "seasonal_naive",
}
_CATEGORY_ALIASES = {
    "machine_learning": "ml", "machine-learning": "ml", "gbm": "ml", "boosting": "ml",
    "classical": "statistical", "stats": "statistical", "univariate": "statistical",
    "neural": "deep_learning", "deep": "deep_learning", "dl": "deep_learning",
    "croston": "intermittent", "sparse": "intermittent",
    "benchmark": "baseline", "naive": "baseline",
}


# ── Pydantic output schema ────────────────────────────────────────────────────

class ModelChoice(BaseModel):
    # NOTE: the Literal must match the category ids in config/model_categories.yaml.
    category: Literal["baseline", "statistical", "intermittent", "ml", "deep_learning"]
    model: str
    runner_up: str | None = None
    confidence: Literal["high", "medium", "low"]
    reason: str

    @field_validator("category", mode="before")
    @classmethod
    def _category_aliases(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
            return _CATEGORY_ALIASES.get(v, v)
        return v

    @field_validator("model", "runner_up", mode="before")
    @classmethod
    def _model_aliases(cls, v):
        if isinstance(v, str):
            v = v.strip().lower().replace(" ", "_").replace("-", "_")
            return _MODEL_ALIASES.get(v, v)
        return v

    @field_validator("reason", mode="before")
    @classmethod
    def _reason_str(cls, v):
        return str(v)[:500] if v is not None else ""


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Task:
You are a time-series model-selection expert. You receive the dataset context (shape,
column names + per-column statistics, the user-confirmed forecast intent — never raw
values), the computed statistics of the forecast series, the catalog of AVAILABLE
candidate models with guidance on when each fits, and a deterministic rule-engine
suggestion with its reasoning trace. Use the dataset context to understand WHAT is
being forecast (the business meaning of the target, series keys and drivers); use the
series statistics to decide HOW to forecast it. Decide the single best model for
training. ineligible_models are listed with the reason they were ruled out — never
pick one.

Return STRICT JSON with exactly these fields:
{
  "category":   "baseline" | "statistical" | "intermittent" | "ml" | "deep_learning",
  "model":      "<one id from eligible_models>",
  "runner_up":  "<a different id from eligible_models>" | null,
  "confidence": "high" | "medium" | "low",
  "reason":     "<2-3 short sentences citing the statistics that drove the choice>"
}

Rules:
- model MUST be one of the ids in eligible_models. Never invent a model.
- Intermittent/lumpy demand (high ADI, many zero periods) needs croston/tsb; smooth
  series never do.
- Two or more seasonal periods -> mstl (or prophet if eligible).
- Exogenous drivers only help if they LEAD the target (best_lag >= 1) or their future
  values are known; lag-0 correlation alone is NOT usable at forecast time. High VIF
  means the drivers are collinear — at most 1-2 of them add real signal. A best_lag
  within +-2 of the seasonal period may just echo the seasonality (seasonal_echo flag).
- SARIMA-family is infeasible for seasonal periods above ~24; prefer
  auto_ets/mstl/lightgbm there.
- Short history (under ~3 full seasonal cycles) favors auto_ets/auto_theta over
  auto_arima; lightgbm needs long history or a many-series panel to beat classical
  models.
- Growing variance (heteroskedastic with a log recommendation) favors multiplicative
  smoothing (auto_ets) or a log transform, which training applies from the hints.
- A recent structural break favors models robust to level shifts and a shorter
  training window; mention it in reason if it influenced you.
- Prefer the simplest adequate model. Agree with the rule engine unless the evidence
  clearly contradicts it; overriding a "high"-confidence rule suggestion needs strong
  justification.
- confidence: "high" only when the statistics point one way unambiguously.
- Return pure JSON. No markdown, no extra text.
"""


def _dataset_context(run_dir: Path) -> dict:
    """Dataset shape + column inventory + confirmed intent — names and statistics only,
    never raw values (same rule as the cleaning/intent agents). Gives the model the
    business context of WHAT is being forecast, not just the series statistics."""
    ctx: dict = {}
    dp_path = run_dir / "cleaning_decision_payload.json"
    if dp_path.exists():
        dp = json.loads(dp_path.read_text())
        ctx["n_rows_raw"] = dp.get("n_rows")
        profile = dp.get("column_profile") or {}
        ctx["columns"] = {
            col: {k: v for k, v in p.items()
                  if k in ("dtype", "null_pct", "distinct_pct", "zero_pct")}
            for col, p in profile.items()
        }
    cm_path = run_dir / "cleaned_metadata.json"
    if cm_path.exists():
        cm = json.loads(cm_path.read_text())
        ctx["n_rows_cleaned"] = cm.get("rows")
        ctx["n_columns"] = cm.get("cols")
    sel_path = run_dir / "forecast_user_selections.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text())
        ctx["confirmed_intent"] = {
            k: sel.get(k) for k in ("timestamp_col", "target_col", "agg", "scope",
                                    "group_cols", "exog_cols")
        }
    return ctx


def _build_llm_payload(payload: dict, rule: dict, catalog: dict, run_dir: Path) -> dict:
    """~3KB of statistics + dataset context + eligible model cards + rule suggestion."""
    stats = {k: v for k, v in payload.items() if k != "panel"}
    if payload.get("panel"):
        stats["panel"] = payload["panel"]
    cards = [
        {
            "id": mid,
            "category": card["categories"][0],
            "supports_exog": card["supports_exog"],
            "min_length": card["min_length"],
            "when_to_use": card["when_to_use"],
        }
        for mid, card in catalog["models"].items()
        if mid in rule["eligible_models"]
    ]
    return {
        "dataset": _dataset_context(run_dir),
        "statistics": stats,
        "derived": rule["derived"],
        "eligible_models": cards,
        "ineligible_models": rule["excluded"],
        "rule_suggestion": {
            "category": rule["category"],
            "model": rule["model"],
            "runner_up": rule["runner_up"],
            "confidence": rule["confidence"],
            "trace": [t["detail"] for t in rule["trace"] if t["fired"]],
        },
    }


# ── Field-level sanitizer ─────────────────────────────────────────────────────

def _sanitize(choice: ModelChoice, rule: dict, catalog: dict) -> tuple[dict, list[str]]:
    """Validate each LLM field against the eligible set; an invalid field reverts to
    the rule pick for that field only. Returns (clean dict, list of overrides)."""
    overrides: list[str] = []
    eligible = set(rule["eligible_models"])
    out = choice.model_dump()

    if out["model"] not in eligible:
        out["model"] = rule["model"]
        out["category"] = rule["category"]
        overrides.append("model")
    else:
        # A valid model in the wrong category: trust the model, fix the category
        # from the catalog (the model id is the decision that matters downstream).
        card = catalog["models"].get(out["model"])
        if card and out["category"] not in card["categories"]:
            out["category"] = card["categories"][0]
            overrides.append("category")

    if out["runner_up"] is not None and (
            out["runner_up"] not in eligible or out["runner_up"] == out["model"]):
        out["runner_up"] = rule["runner_up"] if rule["runner_up"] != out["model"] else None
        overrides.append("runner_up")

    return out, overrides


# ── Entry point ───────────────────────────────────────────────────────────────

def run(run_id: str, use_llm: bool = True) -> dict:
    """Select the forecasting model for a run. The rule engine always runs; the LLM
    (when enabled) may override it. Writes runs/{id}/model_selection.json."""
    run_dir = RUNS_DIR / run_id
    rule = rule_engine.run(run_id)
    catalog = rule_engine.load_catalog()

    llm_cfg = llm_client._load_llm_config()
    llm_model = f"{llm_cfg.get('provider', '?')}/{llm_cfg.get('model', '?')}"
    llm_info: dict = {"suggested_by": "rules", "model": llm_model if use_llm else None,
                      "rationale": None, "error": None, "response": None,
                      "sanitizer_overrides": []}

    final = {k: rule[k] for k in ("category", "model", "runner_up", "confidence", "reason")}
    source = "rule_engine"

    if use_llm:
        payload = _build_llm_payload(
            json.loads((run_dir / "model_selection_payload.json").read_text()),
            rule, catalog, run_dir)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Here is the forecasting task description:\n\n"
                + json.dumps(payload, indent=2)
                + "\n\nReturn the model selection JSON."
            )},
        ]
        try:
            raw = llm_client.call(messages, require_json=True)
            llm_info["response"] = json.loads(json.dumps(raw))
            choice = ModelChoice.model_validate(raw)
            clean, overrides = _sanitize(choice, rule, catalog)

            # Dampen: LLM confidence capped at one level above the rule confidence;
            # a reverted model/category means the LLM's judgement failed -> low.
            if "model" in overrides or "category" in overrides:
                conf = "low"
            else:
                conf = _NAME[min(_RANK[clean["confidence"]],
                                 _RANK[rule["confidence"]] + 1)]

            final = {"category": clean["category"], "model": clean["model"],
                     "runner_up": clean["runner_up"], "confidence": conf,
                     "reason": clean["reason"]}
            # `source` names who actually picked the final model (drives the UI hero
            # card) — if the sanitizer reverted the model, the rule engine picked it.
            source = "rule_engine" if "model" in overrides else "llm"
            llm_info.update({"suggested_by": "llm", "rationale": clean["reason"],
                             "sanitizer_overrides": overrides})
        except (LLMError, ValidationError, Exception) as exc:  # noqa: BLE001
            llm_info["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[model_selector_agent] LLM unavailable or invalid "
                  f"({llm_info['error']}). Rule-engine pick stands.")

    selection = {
        **final,
        "source": source,
        "rule": {k: rule[k] for k in ("category", "model", "runner_up", "confidence",
                                      "reason", "rule_id", "trace")},
        "llm": llm_info,
        "derived": rule["derived"],
        "training_hints": rule["training_hints"],
        "eligible_models": [
            {
                "id": mid,
                "label": card["label"],
                "category": card["categories"][0],
                "available": card["available"],
                "excluded_reason": rule["excluded"].get(mid),
            }
            for mid, card in catalog["models"].items()
        ],
        "selected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    (run_dir / "model_selection.json").write_text(
        json.dumps(selection, indent=2, default=str))
    return {"run_id": run_id, "status": "completed", "selection": selection}
