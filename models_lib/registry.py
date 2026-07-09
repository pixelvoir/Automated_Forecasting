"""Model registry — the single dispatch point from a catalog model id to a Forecaster.

The trainer never imports the individual backends; it goes through ``make_forecaster`` and
the ``*_IDS`` sets here. Prophet and torch (nhits) are imported lazily and
tolerated-missing, matching the catalog's runtime ``available`` gate.
"""
from __future__ import annotations

from models_lib import intermittent, ml_models, statistical

ML_IDS = {"lightgbm", "xgboost"}
NHITS_IDS = {"nhits"}
CHRONOS_IDS = {"chronos"}
STATISTICAL_IDS = set(statistical.IDS)
INTERMITTENT_IDS = set(intermittent.IDS)
PROPHET_IDS = {"prophet"}

# Every model this training layer knows how to build (must be a superset of the catalog's
# selectable models).
KNOWN_IDS = STATISTICAL_IDS | INTERMITTENT_IDS | ML_IDS | NHITS_IDS | CHRONOS_IDS | PROPHET_IDS


def is_ml(model_id: str) -> bool:
    return model_id in ML_IDS


def is_global_capable(model_id: str) -> bool:
    """Models that serve a whole panel with ONE model (routing target for the ``global``
    policy); the panel's TOTAL observations count toward their min_length."""
    return model_id in ML_IDS or model_id in NHITS_IDS or model_id in CHRONOS_IDS


def always_global(model_id: str) -> bool:
    """Models that on per-series scope ALWAYS route through their global/batched path
    regardless of the Stage 5 policy — a per-series-local network per series (nhits) or
    per-series pipeline loads (chronos) would be absurd."""
    return model_id in NHITS_IDS or model_id in CHRONOS_IDS


def make_forecaster(model_id: str, recipe: dict, exog_cols, sarima_max_period: int = 24,
                    params: dict | None = None, frequency: str = "monthly",
                    train_cfg: dict | None = None):
    """Return a fresh Forecaster for `model_id`. Raises KeyError for unknown/unimplemented
    ids (the trainer records that as a per-model error)."""
    transform = recipe.get("transform", "none")
    seasonal_periods = recipe.get("seasonal_periods")
    if model_id in STATISTICAL_IDS:
        return statistical.make(model_id, transform, seasonal_periods, sarima_max_period)
    if model_id in INTERMITTENT_IDS:
        return intermittent.make(model_id, transform)
    if model_id in ML_IDS:
        return ml_models.make(model_id, recipe, params, exog_cols, transform, train_cfg)
    if model_id in NHITS_IDS:
        from models_lib import nhits_model  # lazy: torch is a heavy import
        return nhits_model.make(recipe, transform, train_cfg)
    if model_id in CHRONOS_IDS:
        from models_lib import chronos_model  # lazy: transformers is a heavy import
        return chronos_model.make(recipe, transform, train_cfg)
    if model_id in PROPHET_IDS:
        from models_lib import prophet_model  # lazy: prophet may be absent
        return prophet_model.make(model_id, transform, seasonal_periods, frequency)
    raise KeyError(f"No trainer backend for model '{model_id}'")
