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
# ML-first since the 2026-07 catalog trim: gradient boosting almost always outperformed
# the classical models in practice, so it is also the safest degradation target.
_FALLBACK_ORDER = ["lightgbm", "auto_theta", "seasonal_naive"]

# Deterministic tie-break for equal suitability scores in the ranking (nothing more —
# scores dominate). Roughly "observed general strength" order.
_PREF_ORDER = ["lightgbm", "xgboost", "nhits", "chronos", "auto_theta", "auto_ets",
               "auto_arima", "prophet", "croston", "tsb", "seasonal_naive"]

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


def _total_obs(payload: dict) -> int:
    """Per-series scope: global models (ML/DL) train across the whole panel, so the TOTAL
    observations count toward min_length — a 1,800-series × 60-month panel is 108k
    training points for NHITS/LightGBM/Chronos, not 60."""
    length = payload.get("series_length") or 0
    panel = payload.get("panel") or None
    if payload.get("scope") == "per_series" and panel:
        n_series = panel.get("n_series") or 1
        median_len = (panel.get("series_length") or {}).get("median") or length
        return max(length, int(n_series * median_len))
    return length


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
    total_obs = _total_obs(payload)

    eligible, excluded = [], {}
    for mid, card in catalog["models"].items():
        # Global-capable categories are gated on the panel total; per-series/local models
        # still need each series (proxied by the aggregate grid) to be long enough.
        is_global_capable = any(c in ("ml", "deep_learning") for c in card["categories"])
        eff_length = total_obs if is_global_capable else length
        if not card["available"]:
            excluded[mid] = f"library '{card['import_check']}' not installed"
        elif eff_length < card["min_length"]:
            excluded[mid] = f"series too short ({eff_length} < {card['min_length']} points)"
        elif (mid in ("croston", "tsb") and icls in ("smooth", "erratic")
              and zero_pct < cfg["intermittent_veto_zero_pct"]):
            excluded[mid] = (f"demand is {icls} (zero_pct {zero_pct}%) — "
                             "intermittent models don't apply")
        else:
            eligible.append(mid)
    return eligible, excluded


# ── Suitability scoring (drives the ranked top-N; the ladder still owns rank 1) ──

def _suitability_scores(payload: dict, derived: dict, eligible: list, cfg: dict) -> dict:
    """Per-model suitability score + human-readable reasons, from the SAME signals and
    cfg thresholds the ladder uses — generic, no dataset-specific constants. The scores
    order ranks 3+ of the ranking (the ladder's pick and runner-up own ranks 1-2)."""
    length = payload.get("series_length") or 0
    seas = payload.get("seasonality_strength") or 0
    trend = payload.get("trend_strength") or 0
    period = payload.get("seasonality_period")
    icls = payload.get("intermittency_class")
    zero_pct = payload.get("zero_pct") or 0
    n_cycles = derived["n_cycles"]
    secondary = payload.get("secondary_seasonality_periods") or []
    fcast = payload.get("forecastability")
    freq = payload.get("forecast_frequency") or payload.get("frequency")
    panel = payload.get("panel") or {}
    n_series = panel.get("n_series") or 0
    is_panel = payload.get("scope") == "per_series" and n_series >= cfg["panel_global_min_series"]
    total_obs = _total_obs(payload)
    sband, tband = derived["seasonality_band"], derived["trend_band"]
    strong_exog = [e for e in derived["exog_usable"]
                   if abs(e["best_corr"]) >= cfg["exog_route_min_abs_corr"]]
    sparse = icls in ("intermittent", "lumpy") or zero_pct >= 30
    too_short = length < cfg["short_series_length"] or (
        n_cycles is not None and 1 <= n_cycles < 2)
    near_noise = (fcast is not None and fcast < cfg["low_forecastability"]
                  and seas < cfg["seasonality_moderate"])
    single_seasonal = (sband in ("strong", "moderate")) and not secondary

    scores = {m: {"score": 0.0, "reasons": []} for m in eligible}

    def add(mid, pts, why):
        if mid in scores and pts:
            scores[mid]["score"] += pts
            scores[mid]["reasons"].append(f"{pts:+g} {why}")

    # baseline
    if too_short:
        add("seasonal_naive", 3.0, "history too short for fitted models")
    elif near_noise:
        add("seasonal_naive", 2.0, "near-noise series — honest baseline")
    else:
        add("seasonal_naive", -1.0, "signal exists — baseline is a reference only")

    # intermittent family
    if icls == "intermittent":
        add("croston", 4.0, "intermittent demand quadrant")
        add("tsb", 2.0, "intermittent demand (TSB is the lumpy specialist)")
    elif icls == "lumpy":
        add("tsb", 4.0, "lumpy demand quadrant")
        add("croston", 2.0, "lumpy demand (Croston prefers stable sizes)")
    share = derived.get("panel_intermittent_share") or 0
    if is_panel and share > 0.5:
        mix = panel.get("intermittency_mix") or {}
        winner = "croston" if (mix.get("intermittent") or 0) >= (mix.get("lumpy") or 0) else "tsb"
        add(winner, 2.0, f"{share:.0%} of panel series are intermittent/lumpy")

    # classical
    if tband == "strong" and trend > seas:
        add("auto_theta", 2.0, "trend dominates seasonality")
    elif tband == "moderate":
        add("auto_theta", 1.0, "moderate trend")
    if length < cfg["exog_route_min_length"]:
        add("auto_theta", 1.0, "short/medium history favors classical")
        add("auto_ets", 0.5, "short/medium history favors classical")
    if secondary:
        add("auto_theta", -1.0, "multiple seasonal periods (single-season model)")
        add("auto_ets", -1.0, "multiple seasonal periods (single-season model)")
    if single_seasonal:
        add("auto_ets", 1.5, "single seasonal cycle — ETS territory")
    if payload.get("heteroskedastic") and payload.get("transform_recommendation") == "log":
        add("auto_ets", 1.0, "level-dependent variance suits multiplicative ETS")
    if sparse:
        add("auto_ets", -0.5, "sparse/zero-heavy demand")

    if not derived["sarima_feasible"]:
        add("auto_arima", -3.0, f"period {period} infeasible for SARIMA (> {cfg['sarima_max_period']})")
    elif single_seasonal:
        add("auto_arima", 2.0, "single seasonal period within SARIMA range")
    if len(payload.get("significant_acf_lags") or []) >= 3:
        add("auto_arima", 0.5, "rich autocorrelation structure")
    if (payload.get("ndiffs") or 0) >= 1:
        add("auto_arima", 0.25, "differencing evidence (ndiffs ≥ 1)")
    if length > 1000:
        add("auto_arima", -1.0, "very long series — pmdarima search is slow")

    if secondary:
        add("prophet", 1.5, "multiple seasonal cycles decomposed natively")
    if tband == "strong":
        add("prophet", 1.0, "strong trend with automatic changepoints")
    if period and period > cfg["sarima_max_period"] and sband in ("strong", "moderate"):
        add("prophet", 1.0, f"long seasonal period ({period}) where ARIMA is infeasible")
    if freq in ("daily", "weekly"):
        add("prophet", 0.5, "daily/weekly grid — calendar effects likely")
    if sparse:
        add("prophet", -1.0, "sparse/zero-heavy demand")
    if n_cycles is not None and n_cycles < 2:
        add("prophet", -0.5, "under two seasonal cycles observed")

    # ML
    add("lightgbm", 0.5, "strongest general-purpose default")
    if strong_exog:
        add("lightgbm", 2.0, f"strong leading exogenous driver ({strong_exog[0]['col']})")
    elif derived["exog_usable"]:
        add("lightgbm", 1.0, "usable leading exogenous driver")
    if is_panel:
        add("lightgbm", 1.5, f"{n_series}-series panel — one global model learns shared patterns")
    if secondary:
        add("lightgbm", 1.0, "multiple seasonal periods via Fourier/calendar features")
    if seas >= cfg["seasonality_strong"] and n_cycles is not None \
            and n_cycles >= cfg["min_cycles_for_seasonal"]:
        add("lightgbm", 1.0, "strong seasonality handled by seasonal-index deseasonalization")
    if length >= cfg["exog_route_min_length"]:
        add("lightgbm", 0.5, "long history for feature learning")
    elif length < 60:
        add("lightgbm", -0.5, "short history for a tree ensemble")
    if "lightgbm" in scores and "xgboost" in scores:
        scores["xgboost"]["score"] = scores["lightgbm"]["score"] - 0.25
        scores["xgboost"]["reasons"] = [*scores["lightgbm"]["reasons"],
                                        "-0.25 same learner family; ladder prefers lightgbm"]

    # deep
    if total_obs >= 2000:
        add("nhits", 1.5, f"{total_obs:,} total observations feed a neural net well")
    if is_panel:
        add("nhits", 1.0, "global training across the panel")
    if strong_exog:
        add("nhits", -1.5, "cannot use exogenous drivers")

    add("chronos", 1.5, "pretrained zero-shot generalist (no training cost)")
    if sband in ("strong", "moderate"):
        add("chronos", 0.5, "regular, pattern-rich series")
    if length < 200:
        add("chronos", 0.5, "data-starved regime where trained models overfit")
    if sparse:
        add("chronos", -2.0, "weak on intermittent/zero-heavy demand")
    if strong_exog:
        add("chronos", -1.0, "cannot use exogenous drivers")

    return scores


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
                      category="ml", model="lightgbm", runner_up="nhits",
                      confidence="high", policy="global")
            else:
                _fire("R3", f"panel of {n_series} series — global ML routing, but per-series "
                            "history is measured in rows (not periods) so cycle depth is unverified",
                      category="ml", model="lightgbm", runner_up="nhits",
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
                  category="ml", model="lightgbm", runner_up="xgboost",
                  confidence="high" if abs(best["best_corr"]) >= 0.7 else "medium")
        else:
            detail = "no strong leading exog"
            if strong_exog:
                detail = (f"leading exog exists but only {length} points "
                          f"(< {cfg['exog_route_min_length']}) — kept as a training hint")
            _skip("R4", detail)

    # R5 — multiple seasonal periods (catalog trim 2026-07: MSTL removed — gradient
    # boosting on deseasonalized targets + calendar/Fourier features covers this case)
    if pick is None:
        secondary = payload.get("secondary_seasonality_periods") or []
        if secondary:
            _fire("R5", f"multiple seasonal periods ({period} + {secondary}) — "
                        "gradient boosting with per-period Fourier/calendar features",
                  category="ml", model="lightgbm", runner_up="auto_theta",
                  confidence="high" if seas >= cfg["seasonality_strong"] else "medium")
        else:
            _skip("R5", "single seasonal period")

    # R6 — strong single seasonality with enough cycles. Gradient boosting stays the
    # ladder pick (it consistently outperformed the classical seasonal models here; the
    # ML backend deseasonalizes via per-series seasonal indices) — auto_arima/auto_ets
    # are back in the catalog and surface through the suitability RANKING instead.
    if pick is None:
        if seas >= cfg["seasonality_strong"] and n_cycles is not None \
                and n_cycles >= cfg["min_cycles_for_seasonal"]:
            conf = "high" if seas >= 0.75 else "medium"
            _fire("R6", f"seasonality {seas} over {n_cycles} cycles — gradient boosting on "
                        "the seasonally-indexed target",
                  category="ml", model="lightgbm", runner_up="auto_theta",
                  confidence=conf)
        else:
            _skip("R6", f"seasonality {seas} / {n_cycles if n_cycles is not None else 'n/a'} cycles — "
                        "not a strong-seasonal case")

    # R7 — trend-dominant
    if pick is None:
        if trend >= cfg["trend_strong"] and trend > seas:
            _fire("R7", f"trend {trend} dominates seasonality {seas} — "
                        "Theta deseasonalizes and extrapolates the trend robustly",
                  category="statistical", model="auto_theta", runner_up="lightgbm",
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

    # R9 — default (was auto_ets; ML-first since the catalog trim)
    if pick is None:
        _fire("R9", "no rule matched decisively — gradient boosting is the strongest "
                    "general-purpose default in this catalog",
              category="ml", model="lightgbm", runner_up="auto_theta",
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

    # Ranked eligible models: suitability scores order the list, but the ladder verdict
    # is authoritative for the top — the pick is hard-promoted to rank 1 and the ladder
    # runner-up to rank 2 (scores alone decide ranks 3+).
    scores = _suitability_scores(payload, derived, eligible, cfg)
    order = sorted(eligible, key=lambda m: (
        -scores[m]["score"],
        _PREF_ORDER.index(m) if m in _PREF_ORDER else len(_PREF_ORDER)))
    for m in (pick.get("runner_up"), pick["model"]):
        if m in order:
            order.remove(m)
            order.insert(0, m)
    ranking = [{"rank": i + 1, "model": m, "category": _category_of(m, catalog),
                "score": round(scores[m]["score"], 2), "reasons": scores[m]["reasons"]}
               for i, m in enumerate(order)]

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
        "ranking": ranking,
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
