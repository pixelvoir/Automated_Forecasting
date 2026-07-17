"""Stage 5 agent: LLM model RANKING on top of the deterministic rule engine.

pipeline/rule_engine.py ALWAYS runs first and produces a pick + reasoning trace + a
suitability ranking of every eligible model. When use_llm is on, the LLM receives the
Stage 4 statistics, the derived decision features, the eligible model cards and the
rule suggestion (pick + ranking), and returns a re-ranked TOP-5 with a one-sentence
reason per entry — a list-level sanitizer drops ineligible/duplicate entries, backfills
from the rule ranking, and puts the rule pick back at rank 1 whenever the LLM's #1
didn't survive. LLM confidence is dampened to at most one level above the rule
confidence. LLM failure → the rule ranking stands and the error is recorded for the UI.

Writes runs/{id}/model_selection.json — the single file Stage 6 (feature engineering),
Stage 7 (training) and the UI read: model + runner_up (= ranks 1/2, backward compat) +
`ranking` (the top-5 the user picks training candidates from) + training_hints.
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
    "cat_boost": "catboost", "catboostregressor": "catboost",
    "patch_tst": "patchtst",
    "croston_optimized": "croston", "crostonoptimized": "croston",
    "naive": "seasonal_naive", "seasonalnaive": "seasonal_naive",
    "fbprophet": "prophet", "meta_prophet": "prophet",
    "chronos_bolt": "chronos", "chronos_bolt_small": "chronos",
    "amazon/chronos_bolt_small": "chronos", "chronos_forecasting": "chronos",
}
# ── Pydantic output schema ────────────────────────────────────────────────────

class RankedEntry(BaseModel):
    model: str
    reason: str = ""

    @field_validator("model", mode="before")
    @classmethod
    def _model_aliases(cls, v):
        if isinstance(v, str):
            v = v.strip().lower().replace(" ", "_").replace("-", "_")
            return _MODEL_ALIASES.get(v, v)
        return v

    @field_validator("reason", mode="before")
    @classmethod
    def _reason_str(cls, v):
        return str(v)[:250] if v is not None else ""


class ModelRanking(BaseModel):
    """The LLM re-ranks the eligible models (top-5, best first). Category is NOT part of
    the contract — it is derived from the catalog for whichever model lands rank 1."""
    ranking: list[RankedEntry]
    confidence: Literal["high", "medium", "low"]
    reason: str

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
suggestion: its top pick with the reasoning trace AND its full suitability ranking.
Use the dataset context to understand WHAT is being forecast (the business meaning of
the target, series keys and drivers); use the series statistics to decide HOW to
forecast it. Produce a RANKING of the best models to train, best first — the user
trains the top of this list, so order = expected accuracy. ineligible_models are
listed with the reason they were ruled out — never include one.

Return STRICT JSON with exactly these fields:
{
  "ranking": [
    {"model": "<id from eligible_models>", "reason": "<one sentence: why this rank>"},
    ...
  ],
  "confidence": "high" | "medium" | "low",
  "reason": "<2-3 short sentences on the #1 pick, citing the statistics that drove it>"
}
The ranking must contain min(5, number of eligible models) DISTINCT entries, best
first. Start from rule_suggestion.ranking and reorder only where the statistics
justify it.

Model guidance (beyond each card's when_to_use):
- Intermittent/lumpy demand (high ADI, many zero periods) needs croston/tsb; smooth
  series never do.
- Two or more seasonal periods -> lightgbm (Fourier/calendar features capture each
  cycle after seasonal-index deseasonalization) or prophet (decomposes them natively).
- chronos is a pretrained zero-shot transformer: a strong generalist on regular,
  pattern-rich series with no training cost, but weak on intermittent/zero-heavy
  demand and it cannot use exogenous drivers.
- prophet fits trend with automatic changepoints plus multiple seasonalities and
  calendar effects — strongest on daily/weekly data and long seasonal periods (e.g.
  yearly-365) where ARIMA is infeasible.
- auto_arima fits LOW-ORDER autoregressive structure: prefer it only when the PACF
  cuts off after a few lags (pacf_significant_lags short), seasonality is weak to
  moderate, and the single seasonal period is <= 24. On strongly seasonal series the
  rule engine's gradient-boosting pick has empirically outperformed it in this
  pipeline — many significant ACF lags alone are NOT evidence for ARIMA (every
  seasonal series has them). Never for weekly-52 / daily-365 seasonality.
- auto_ets suits smooth single-seasonal series with SHORT history where tree models
  overfit; its multiplicative variants handle level-dependent variance
  (heteroskedastic with a log recommendation). Do not promote it over a
  high-confidence ML pick just because the series is seasonal.
- Exogenous drivers only help if they LEAD the target (best_lag >= 1) or their future
  values are known; lag-0 correlation alone is NOT usable at forecast time. High VIF
  means the drivers are collinear — at most 1-2 of them add real signal. A best_lag
  within +-2 of the seasonal period may just echo the seasonality (seasonal_echo flag).
  Only the ML models (lightgbm/xgboost/catboost) can exploit exogenous drivers.
- lightgbm, xgboost, catboost, nhits, patchtst and chronos serve many-series panels with
  ONE global model: the panel's TOTAL observations count toward their history requirement.
- Short history (under ~3 full seasonal cycles) favors auto_theta/auto_ets/chronos;
  lightgbm/nhits need long history or a many-series panel to beat classical models.
- Growing variance (heteroskedastic with a log recommendation) is handled by the log
  transform, which training applies automatically from the hints.
- A recent structural break favors models robust to level shifts; mention it in
  reason if it influenced the order.
- When rule_suggestion.confidence is "high", KEEP its #1 at rank 1 unless a SPECIFIC
  statistic in the payload directly contradicts the fired rule — and name that
  statistic in `reason`. Reordering ranks 2-5 is always fine. (A demotion of a
  high-confidence rule pick without such evidence is reverted deterministically.)
- Extra payload signals when present: pacf_significant_lags (short list = low-order AR
  structure), structural_breaks {n, recent_dates}, slope_pct_per_period vs
  recent_slope_pct_per_period (trend acceleration), approx_entropy / dfa_alpha
  (regularity / long-range persistence).
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
            "ranking": [{"model": e["model"], "score": e["score"],
                         "reasons": e["reasons"]}
                        for e in (rule.get("ranking") or [])[:5]],
        },
    }


# ── List-level sanitizer ──────────────────────────────────────────────────────

def _rank_cap(rule: dict) -> int:
    return min(5, len(rule["eligible_models"]))


def _sanitize_ranking(choice: ModelRanking, rule: dict) -> tuple[list[dict], list[str]]:
    """Validate the LLM ranking against the eligible set: drop invalid/duplicate
    entries, then backfill from the rule ranking up to min(5, n_eligible). Returns
    (entries [{model, reason, source}], overrides). Override "rank1" means the LLM's
    top pick did not survive — the rule engine owns the decision then."""
    overrides: list[str] = []
    eligible = set(rule["eligible_models"])
    llm_first = choice.ranking[0].model if choice.ranking else None

    entries, seen = [], set()
    for e in choice.ranking:
        if e.model in eligible and e.model not in seen:
            entries.append({"model": e.model, "reason": e.reason, "source": "llm"})
            seen.add(e.model)
        else:
            overrides.append("ranking")
    if llm_first is None or not entries or entries[0]["model"] != llm_first:
        overrides.append("rank1")

    for r in rule.get("ranking") or []:
        if len(entries) >= _rank_cap(rule):
            break
        if r["model"] not in seen:
            entries.append({"model": r["model"],
                            "reason": "; ".join(r["reasons"]) or "rule-ranked",
                            "source": "rule"})
            seen.add(r["model"])
            overrides.append("ranking")

    # The LLM's #1 didn't survive → the rule pick owns rank 1 (the hero card must
    # attribute the decision honestly; a demoted-by-accident rule pick would lie).
    if "rank1" in overrides:
        entries = [e for e in entries if e["model"] != rule["model"]]
        entries.insert(0, {"model": rule["model"], "reason": rule["reason"],
                           "source": "rule"})
    return entries[:_rank_cap(rule)], sorted(set(overrides))


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

    def _cat_of(mid):
        card = catalog["models"].get(mid)
        return card["categories"][0] if card else None

    # Default (no LLM / LLM failure): the rule engine's suitability ranking, with the
    # ladder's own reason on rank 1.
    entries = [{"model": e["model"], "reason": "; ".join(e["reasons"]) or rule["reason"],
                "source": "rule"}
               for e in (rule.get("ranking") or [])[:_rank_cap(rule)]]
    if entries and entries[0]["model"] == rule["model"]:
        entries[0]["reason"] = rule["reason"]
    confidence, top_reason = rule["confidence"], rule["reason"]
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
                + "\n\nReturn the model ranking JSON."
            )},
        ]
        try:
            raw = llm_client.call(messages, require_json=True)
            llm_info["response"] = json.loads(json.dumps(raw))
            choice = ModelRanking.model_validate(raw)
            clean_entries, overrides = _sanitize_ranking(choice, rule)
            if clean_entries:
                entries = clean_entries

            # High-confidence guard (2026-07-17): a VALID LLM #1 that demotes a
            # high-confidence rule pick is either reverted (when its rule suitability
            # is far below the rule pick's — llm_demote_margin) or kept with
            # confidence capped at medium. Observed failures this closes: gemini
            # promoted auto_arima/auto_ets over high-confidence R6 lightgbm picks
            # with zero sanitizer intervention and kept "high" confidence.
            if ("rank1" not in overrides and entries and rule["confidence"] == "high"
                    and entries[0]["model"] != rule["model"]):
                scores = {e["model"]: e["score"] for e in rule.get("ranking") or []}
                margin = float(rule_engine._load_cfg().get("llm_demote_margin", 1.5))
                llm_s, rule_s = scores.get(entries[0]["model"]), scores.get(rule["model"])
                if llm_s is not None and rule_s is not None and llm_s < rule_s - margin:
                    entries = [e for e in entries if e["model"] != rule["model"]]
                    entries.insert(0, {"model": rule["model"], "reason": rule["reason"],
                                       "source": "rule"})
                    overrides = sorted({*overrides, "rank1",
                                        "demoted_high_rule_pick_reverted"})
                else:
                    overrides = sorted({*overrides, "demoted_high_rule_pick"})

            # Dampen: LLM confidence capped at one level above the rule confidence;
            # a reverted rank-1 means the LLM's judgement failed -> low.
            if "rank1" in overrides:
                confidence = "low"
                top_reason = rule["reason"]
            else:
                confidence = _NAME[min(_RANK[choice.confidence],
                                       _RANK[rule["confidence"]] + 1)]
                top_reason = choice.reason
                if "demoted_high_rule_pick" in overrides:
                    confidence = _NAME[min(_RANK[confidence], _RANK["medium"])]
            # `source` names who actually picked the final model (drives the UI hero
            # card) — if the sanitizer reverted rank 1, the rule engine picked it.
            source = "rule_engine" if "rank1" in overrides else "llm"
            llm_info.update({"suggested_by": "llm", "rationale": choice.reason,
                             "sanitizer_overrides": overrides})
        except (LLMError, ValidationError, Exception) as exc:  # noqa: BLE001
            llm_info["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[model_selector_agent] LLM unavailable or invalid "
                  f"({llm_info['error']}). Rule-engine ranking stands.")

    rule_score_of = {e["model"]: e["score"] for e in rule.get("ranking") or []}
    label_of = {mid: card["label"] for mid, card in catalog["models"].items()}
    ranking = [{"rank": i + 1, "model": e["model"], "category": _cat_of(e["model"]),
                "label": label_of.get(e["model"], e["model"]), "reason": e["reason"],
                "source": e["source"], "rule_score": rule_score_of.get(e["model"])}
               for i, e in enumerate(entries)]

    top = ranking[0]["model"] if ranking else rule["model"]
    final = {
        "category": _cat_of(top) or rule["category"],
        "model": top,
        "runner_up": ranking[1]["model"] if len(ranking) > 1 else None,
        "confidence": confidence,
        "reason": top_reason,
    }

    selection = {
        **final,
        "source": source,
        "ranking": ranking,
        "rule": {**{k: rule[k] for k in ("category", "model", "runner_up", "confidence",
                                         "reason", "rule_id", "trace")},
                 "ranking": rule.get("ranking") or []},
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
