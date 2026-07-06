"""Stage 5 rule engine: deterministic model selection from the Stage 4 evidence.

Consumes runs/{id}/model_selection_payload.json (statistics only — never raw data) and
the catalog in config/model_categories.yaml, and picks one model + runner-up via an
ordered first-match rule ladder (R1..R9). Every rule evaluated leaves a trace entry, so
the decision is fully auditable in the UI and in the LLM prompt.

The engine ALWAYS runs — agents/model_selector_agent.py sends its pick + trace to the
LLM as the rule suggestion, and reverts to it field-by-field when the LLM answer is
invalid or the LLM is unavailable. `decide()` is pure (payload + catalog + cfg in,
decision out) so it can be tested without a run directory; `run()` is the thin file
wrapper. Nothing is persisted here — the agent writes model_selection.json.
"""
import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
CONFIG_DIR = ROOT / "config"

_DEFAULT_CFG = {
    "short_series_length": 20,
    "low_forecastability": 0.2,
    "seasonality_strong": 0.6,
    "seasonality_moderate": 0.3,
    "trend_strong": 0.6,
    "min_cycles_for_seasonal": 3,
    "sarima_max_period": 24,
    "exog_min_abs_corr": 0.4,
    "exog_route_min_abs_corr": 0.5,
    "exog_route_min_length": 100,
    "vif_collinear": 10,
    "panel_global_min_series": 30,
    "panel_global_min_cycles": 2,
    "intermittent_veto_zero_pct": 20,
}

# Preference order when a rule's pick turns out ineligible (e.g. library missing).
_FALLBACK_ORDER = ["auto_ets", "auto_theta", "seasonal_naive"]

_RANK = {"low": 0, "medium": 1, "high": 2}
_NAME = {0: "low", 1: "medium", 2: "high"}


def _load_cfg() -> dict:
    cfg = dict(_DEFAULT_CFG)
    settings_path = CONFIG_DIR / "settings.yaml"
    if settings_path.exists():
        settings = yaml.safe_load(settings_path.read_text()) or {}
        cfg.update(settings.get("model_selection") or {})
    return cfg


def load_catalog() -> dict:
    """Catalog from model_categories.yaml + runtime availability per model.

    Availability is derived (never stored): installing a library enables its models
    with zero config changes. Runs inside the job subprocess, so it sees the venv.
    """
    raw = yaml.safe_load((CONFIG_DIR / "model_categories.yaml").read_text())
    models = {}
    for card in raw["models"]:
        card = dict(card)
        try:
            card["available"] = importlib.util.find_spec(card["import_check"]) is not None
        except (ImportError, ValueError):
            card["available"] = False
        models[card["id"]] = card
    return {"categories": raw["categories"], "models": models}


# ── Derived decision features ────────────────────────────────────────────────

def _band(value, strong, moderate) -> str:
    if value is None:
        return "unknown"
    if value >= strong:
        return "strong"
    if value >= moderate:
        return "moderate"
    return "weak"


def _derive(payload: dict, cfg: dict) -> dict:
    """Decision features the raw payload doesn't state directly."""
    length = payload.get("series_length") or 0
    period = payload.get("seasonality_period")
    horizon = payload.get("horizon") or 0
    seas = payload.get("seasonality_strength")
    trend = payload.get("trend_strength")
    ljung = payload.get("residual_ljung_box_p")

    n_cycles = round(length / period, 2) if period and period > 1 else None

    exog_usable, exog_lag0_only = [], []
    for e in payload.get("exog_top") or []:
        best_corr = e.get("best_corr") or 0
        best_lag = e.get("best_lag") or 0
        if best_lag >= 1 and abs(best_corr) >= cfg["exog_min_abs_corr"]:
            exog_usable.append({
                "col": e["col"],
                "best_lag": best_lag,
                "best_corr": best_corr,
                # lag >= horizon: known values cover the whole forecast window
                "covers_horizon": best_lag >= horizon > 0,
                # a lag within ±2 of the seasonal period may just echo the seasonality
                "seasonal_echo": bool(period and abs(best_lag - period) <= 2),
            })
        elif best_lag == 0 and abs(best_corr) >= cfg["exog_min_abs_corr"]:
            # correlated but concurrent — unusable unless future values are known
            exog_lag0_only.append(e["col"])

    panel = payload.get("panel") or None
    panel_cycles = None
    panel_intermittent_share = None
    if panel:
        mix = panel.get("intermittency_mix") or {}
        panel_intermittent_share = round(
            (mix.get("intermittent") or 0) + (mix.get("lumpy") or 0), 3)
        if panel.get("length_unit") == "periods" and period and period > 1:
            median_len = (panel.get("series_length") or {}).get("median")
            if median_len:
                panel_cycles = round(median_len / period, 2)

    vif_max = payload.get("vif_max")
    return {
        "n_cycles": n_cycles,
        "seasonality_band": _band(seas, cfg["seasonality_strong"], cfg["seasonality_moderate"]),
        "trend_band": _band(trend, cfg["trend_strong"], cfg["seasonality_moderate"]),
        "horizon_ratio": round(horizon / length, 3) if length else None,
        "count_target": payload.get("agg") in ("nunique", "count"),
        "sarima_feasible": period is None or period <= cfg["sarima_max_period"],
        "residual_structure": ljung is not None and ljung < 0.05,
        "exog_usable": exog_usable,
        "exog_lag0_only": exog_lag0_only,
        "vif_collinear": vif_max is not None and vif_max > cfg["vif_collinear"],
        "panel_cycles": panel_cycles,
        "panel_intermittent_share": panel_intermittent_share,
    }


# ── Eligibility ──────────────────────────────────────────────────────────────

def _eligible(payload: dict, derived: dict, catalog: dict, cfg: dict) -> tuple[list, dict]:
    """Availability + data-size gates + the smooth-demand veto on Croston-family.

    Returns (eligible_ids, {model_id: exclusion_reason}) — exclusions feed the UI table
    and make an invalid LLM pick (e.g. croston on smooth monthly data) structurally
    impossible: the sanitizer only accepts eligible ids.
    """
    length = payload.get("series_length") or 0
    icls = payload.get("intermittency_class")
    zero_pct = payload.get("zero_pct") or 0

    eligible, excluded = [], {}
    for mid, card in catalog["models"].items():
        if not card["available"]:
            excluded[mid] = f"library '{card['import_check']}' not installed"
        elif length < card["min_length"]:
            excluded[mid] = f"series too short ({length} < {card['min_length']} points)"
        elif (mid in ("croston", "tsb") and icls in ("smooth", "erratic")
              and zero_pct < cfg["intermittent_veto_zero_pct"]):
            excluded[mid] = (f"demand is {icls} (zero_pct {zero_pct}%) — "
                             "intermittent models don't apply")
        else:
            eligible.append(mid)
    return eligible, excluded


# ── Rule ladder ──────────────────────────────────────────────────────────────

def decide(payload: dict, catalog: dict, cfg: dict | None = None) -> dict:
    """Pure first-match decision over R1..R9. Every rule evaluated gets a trace entry."""
    cfg = cfg or _load_cfg()
    derived = _derive(payload, cfg)
    eligible, excluded = _eligible(payload, derived, catalog, cfg)

    length = payload.get("series_length") or 0
    seas = payload.get("seasonality_strength") or 0
    trend = payload.get("trend_strength") or 0
    period = payload.get("seasonality_period")
    icls = payload.get("intermittency_class")
    n_cycles = derived["n_cycles"]
    trace: list[dict] = []
    pick: dict | None = None

    def _fire(rule_id, detail, **decision):
        nonlocal pick
        trace.append({"rule": rule_id, "fired": True, "detail": detail})
        pick = {"rule_id": rule_id, "reason": detail, **decision}

    def _skip(rule_id, detail):
        trace.append({"rule": rule_id, "fired": False, "detail": detail})

    # R1 — degenerate-short history
    if pick is None:
        if length < cfg["short_series_length"]:
            _fire("R1", f"only {length} points — too short for anything beyond a baseline",
                  category="baseline", model="seasonal_naive", runner_up="auto_theta",
                  confidence="high")
        elif n_cycles is not None and 1 <= n_cycles < 2:
            _fire("R1", f"only {n_cycles} seasonal cycles observed (period {period}) — "
                        "repeat the one known cycle rather than fit a seasonal model",
                  category="baseline", model="seasonal_naive", runner_up="auto_theta",
                  confidence="medium")
        else:
            _skip("R1", f"{length} points / {n_cycles if n_cycles is not None else 'n/a'} cycles — long enough")

    # R2 — intermittent demand quadrant (deterministic: ADI/CV² classification)
    if pick is None:
        if icls == "intermittent":
            _fire("R2", f"intermittent demand (ADI {payload.get('adi')}, CV² {payload.get('cv2')}) — "
                        "Croston forecasts the demand rate",
                  category="intermittent", model="croston", runner_up="tsb", confidence="high")
        elif icls == "lumpy":
            _fire("R2", f"lumpy demand (ADI {payload.get('adi')}, CV² {payload.get('cv2')}) — "
                        "TSB handles variable sizes and obsolescence",
                  category="intermittent", model="tsb", runner_up="croston", confidence="high")
        else:
            _skip("R2", f"demand class '{icls}' — not intermittent")

    # R3 — many-series panel routes to a global model
    panel = payload.get("panel") or None
    if pick is None:
        if (payload.get("scope") == "per_series" and panel
                and (panel.get("n_series") or 0) >= cfg["panel_global_min_series"]):
            n_series = panel["n_series"]
            share = derived["panel_intermittent_share"] or 0
            if share > 0.5:
                mix = panel.get("intermittency_mix") or {}
                model = "croston" if (mix.get("intermittent") or 0) >= (mix.get("lumpy") or 0) else "tsb"
                _fire("R3", f"panel of {n_series} series, {share:.0%} intermittent/lumpy — "
                            "per-series Croston-family beats a global fit on sparse demand",
                      category="intermittent", model=model, runner_up="lightgbm",
                      confidence="medium", policy="per_series_local")
            elif derived["panel_cycles"] is not None and derived["panel_cycles"] >= cfg["panel_global_min_cycles"]:
                _fire("R3", f"panel of {n_series} series with median {derived['panel_cycles']} "
                            "cycles of history — one global ML model learns shared patterns",
                      category="ml", model="lightgbm", runner_up="auto_ets",
                      confidence="high", policy="global")
            else:
                _fire("R3", f"panel of {n_series} series — global ML routing, but per-series "
                            "history is measured in rows (not periods) so cycle depth is unverified",
                      category="ml", model="lightgbm", runner_up="auto_ets",
                      confidence="medium", policy="global")
        else:
            _skip("R3", "not a many-series panel"
                        + (f" (n_series {panel.get('n_series')})" if panel else ""))

    # R4 — strong leading exogenous drivers route to ML (only with enough history)
    if pick is None:
        strong_exog = [e for e in derived["exog_usable"]
                       if abs(e["best_corr"]) >= cfg["exog_route_min_abs_corr"]]
        if strong_exog and length >= cfg["exog_route_min_length"] and "lightgbm" in eligible:
            best = max(strong_exog, key=lambda e: abs(e["best_corr"]))
            _fire("R4", f"leading exogenous driver {best['col']} (lag {best['best_lag']}, "
                        f"corr {best['best_corr']}) with {length} points — "
                        "gradient boosting exploits driver features best",
                  category="ml", model="lightgbm", runner_up="auto_arima",
                  confidence="high" if abs(best["best_corr"]) >= 0.7 else "medium")
        else:
            detail = "no strong leading exog"
            if strong_exog:
                detail = (f"leading exog exists but only {length} points "
                          f"(< {cfg['exog_route_min_length']}) — kept as a hint for SARIMAX")
            _skip("R4", detail)

    # R5 — multiple seasonal periods
    if pick is None:
        secondary = payload.get("secondary_seasonality_periods") or []
        if secondary:
            _fire("R5", f"multiple seasonal periods ({period} + {secondary}) — "
                        "MSTL decomposes each separately",
                  category="statistical", model="mstl", runner_up="auto_ets",
                  confidence="high" if seas >= cfg["seasonality_strong"] else "medium")
        else:
            _skip("R5", "single seasonal period")

    # R6 — strong single seasonality with enough cycles
    if pick is None:
        if seas >= cfg["seasonality_strong"] and n_cycles is not None \
                and n_cycles >= cfg["min_cycles_for_seasonal"]:
            acf_rich = len(payload.get("significant_acf_lags") or []) >= 3
            arima_evidence = derived["sarima_feasible"] and (acf_rich or derived["residual_structure"])
            conf = "high" if seas >= 0.75 else "medium"
            if payload.get("heteroskedastic") and payload.get("transform_recommendation") == "log":
                _fire("R6", f"seasonality {seas} over {n_cycles} cycles with level-dependent "
                            "variance — multiplicative exponential smoothing",
                      category="statistical", model="auto_ets", runner_up="auto_arima",
                      confidence=conf)
            elif arima_evidence:
                _fire("R6", f"seasonality {seas} over {n_cycles} cycles with rich autocorrelation "
                            f"(period {period} is SARIMA-feasible) — seasonal ARIMA",
                      category="statistical", model="auto_arima", runner_up="auto_ets",
                      confidence=conf)
            else:
                detail = f"seasonality {seas} over {n_cycles} cycles — exponential smoothing"
                if not derived["sarima_feasible"]:
                    detail += f" (period {period} too long for SARIMA)"
                _fire("R6", detail, category="statistical", model="auto_ets",
                      runner_up="seasonal_naive", confidence=conf)
        else:
            _skip("R6", f"seasonality {seas} / {n_cycles if n_cycles is not None else 'n/a'} cycles — "
                        "not a strong-seasonal case")

    # R7 — trend-dominant
    if pick is None:
        if trend >= cfg["trend_strong"] and trend > seas:
            _fire("R7", f"trend {trend} dominates seasonality {seas} — "
                        "Theta deseasonalizes and extrapolates the trend robustly",
                  category="statistical", model="auto_theta", runner_up="auto_ets",
                  confidence="high" if seas < cfg["seasonality_moderate"] else "medium")
        else:
            _skip("R7", f"trend {trend} not dominant")

    # R8 — barely forecastable
    if pick is None:
        fcast = payload.get("forecastability")
        if fcast is not None and fcast < cfg["low_forecastability"] \
                and seas < cfg["seasonality_moderate"]:
            _fire("R8", f"forecastability {fcast} with weak seasonality — the series is "
                        "mostly noise; a simple baseline is the honest pick",
                  category="baseline", model="seasonal_naive", runner_up="auto_theta",
                  confidence="medium")
        else:
            _skip("R8", f"forecastability {fcast} — signal exists")

    # R9 — default
    if pick is None:
        _fire("R9", "no rule matched decisively — exponential smoothing is the safest "
                    "general-purpose default",
              category="statistical", model="auto_ets", runner_up="auto_theta",
              confidence="low")

    # The pick must be eligible; degrade along the fallback order if not.
    if pick["model"] not in eligible:
        replacement = next(
            (m for m in [pick.get("runner_up"), *_FALLBACK_ORDER] if m in eligible), None)
        trace.append({"rule": pick["rule_id"], "fired": True,
                      "detail": f"pick '{pick['model']}' ineligible "
                                f"({excluded.get(pick['model'], 'unknown')}) — "
                                f"degraded to '{replacement}'"})
        pick["model"] = replacement or pick["model"]
        pick["category"] = _category_of(pick["model"], catalog) or pick["category"]
        pick["confidence"] = _NAME[max(_RANK[pick["confidence"]] - 1, 0)]
    if pick.get("runner_up") not in eligible or pick.get("runner_up") == pick["model"]:
        pick["runner_up"] = next(
            (m for m in _FALLBACK_ORDER if m in eligible and m != pick["model"]), None)

    # Training hints — consumed by Stages 6/7 regardless of which rule fired.
    secondary = payload.get("secondary_seasonality_periods") or []
    default_policy = "aggregate" if payload.get("scope") != "per_series" else "per_series_local"
    hints = {
        # Stage 4's transform recommendation is the authority — don't invent one here
        "transform": payload.get("transform_recommendation") or "none",
        "exog_usable": derived["exog_usable"],
        "seasonal_periods": ([period] + secondary) if period and period > 1 else [1],
        "policy": pick.get("policy", default_policy),
        "metric": "MASE",
    }
    pick.pop("policy", None)

    return {
        **pick,
        "trace": trace,
        "derived": derived,
        "training_hints": hints,
        "eligible_models": eligible,
        "excluded": excluded,
    }


def _category_of(model_id: str, catalog: dict) -> str | None:
    card = catalog["models"].get(model_id)
    return card["categories"][0] if card else None


# ── Entry point ──────────────────────────────────────────────────────────────

def run(run_id: str) -> dict:
    """Decide from runs/{run_id}/model_selection_payload.json. Writes nothing —
    agents/model_selector_agent.py persists the final selection."""
    payload_path = RUNS_DIR / run_id / "model_selection_payload.json"
    if not payload_path.exists():
        raise FileNotFoundError(
            f"Payload not found: {payload_path}. Run forecast EDA (Stage 4) first.")
    payload = json.loads(payload_path.read_text())
    return decide(payload, load_catalog(), _load_cfg())
