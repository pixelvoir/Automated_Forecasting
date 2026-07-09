"""Amazon Chronos (Bolt) — pretrained zero-shot forecaster, inference only.

No training happens here at all: `chronos-forecasting` loads a pretrained transformer
(default `amazon/chronos-bolt-small`, ~190MB from HuggingFace on first use, cached
locally afterwards) and forecasts directly from the raw context window. That makes it
the cheapest backend in the catalog — one batched inference pass even for a
1,800-series panel — and a strong generalist on regular, pattern-rich series. It cannot
use exogenous drivers and is weak on intermittent / zero-heavy demand (the rule engine's
suitability scoring encodes both).

Two entry points mirror ``nhits_model``:
  * ``ChronosForecaster`` — single series (aggregate scope), standard ``Forecaster``
    interface so the walk-forward harness backtests it like any other model. "Fit" just
    stores the history and loads the pipeline (so a missing package / blocked download
    fails fast inside fit, where the trainer's per-model isolation records it).
  * ``run_global`` — per-series scope (always routed here, like nhits): batched holdout
    backtest (last h periods per series, predicted from the truncated context) + batched
    final forecast. Returns the same dict contract as ``nhits_model.run_global``.

Quantiles: Chronos-Bolt natively outputs quantiles, but only within its trained range
[0.1, 0.9] — anything outside is edge-clamped by the library. ``predict_quantiles``
therefore returns ONLY the interval levels whose quantile pair fits that range (80 does,
95 does not); the trainer fills the missing levels with conformal intervals.

The pipeline object is cached per process and NEVER pickled — ``_fitted`` descriptors
carry only the variant id (the weights are re-downloadable by name).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_lib.base_model import Forecaster, apply_transform, invert_transform

IDS = {"chronos"}
_DEFAULT_VARIANT = "amazon/chronos-bolt-small"
_NATIVE_Q_MIN = 0.1          # Chronos-Bolt's trained quantile range is [0.1, 0.9]
_BATCH_SERIES = 256          # panel inference chunk (bounds RAM on huge panels)

_PIPELINES: dict[str, object] = {}


def _variant(train_cfg: dict | None) -> str:
    return (train_cfg or {}).get("chronos_variant") or _DEFAULT_VARIANT


def _load_pipeline(variant: str):
    if variant not in _PIPELINES:
        import torch
        from chronos import BaseChronosPipeline  # lazy: transformers is a heavy import
        _PIPELINES[variant] = BaseChronosPipeline.from_pretrained(
            variant, torch_dtype=torch.float32)
    return _PIPELINES[variant]


def _supported_levels(levels) -> list[int]:
    """Interval levels whose lower quantile lies inside Bolt's trained range."""
    return [int(l) for l in levels if (100 - int(l)) / 200.0 >= _NATIVE_Q_MIN - 1e-9]


def _predict_batch(pipe, contexts: list[np.ndarray], h: int, qlevels: list[float]):
    """One batched quantile inference call. Returns (n_series, h, n_q) numpy."""
    import torch
    tensors = [torch.tensor(np.asarray(c, dtype=np.float32)) for c in contexts]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # long-horizon / clamped-quantile advisories
        q, _mean = pipe.predict_quantiles(tensors, prediction_length=int(h),
                                          quantile_levels=list(qlevels))
    return q.numpy()


# ── Single-series forecaster (aggregate scope) ────────────────────────────────

class ChronosForecaster(Forecaster):
    uses_features = False

    def __init__(self, recipe: dict, transform: str = "none", train_cfg: dict | None = None):
        super().__init__(transform)
        self.recipe = recipe
        self.train_cfg = train_cfg or {}

    def _fit(self, hist):
        self._y = hist["y"].to_numpy(dtype=float)  # transformed by the base class
        if len(self._y) < 5:
            raise RuntimeError(f"chronos: series too short ({len(self._y)} points)")
        self._variant_id = _variant(self.train_cfg)
        self._pipe = _load_pipeline(self._variant_id)  # fail fast on package/download
        self.params_ = {"backend": "chronos", "variant": self._variant_id,
                        "zero_shot": True}

    def _predict(self, h, future_ds, future_exog=None):
        q = _predict_batch(self._pipe, [self._y], int(h), [0.5])
        return q[0, :, 0]  # transformed scale; base inverts

    def predict_quantiles(self, h: int, future_ds, levels=(80, 95)) -> dict:
        """Native quantile bands on the ORIGINAL scale — only for levels within Bolt's
        trained quantile range (80 yes, 95 no); the trainer conformal-fills the rest."""
        supported = _supported_levels(levels)
        if not supported:
            return {}
        alphas = sorted({a for lvl in supported
                         for a in ((100 - lvl) / 200.0, 1 - (100 - lvl) / 200.0)})
        q = _predict_batch(self._pipe, [self._y], int(h), alphas)[0]  # (h, n_q)
        q = np.sort(q, axis=1)  # non-crossing
        vals = {a: invert_transform(q[:, i], self.transform) for i, a in enumerate(alphas)}
        if np.nanmin(self._y_raw) >= 0:
            vals = {a: np.clip(v, 0, None) for a, v in vals.items()}
        return {lvl: (vals[(100 - lvl) / 200.0], vals[1 - (100 - lvl) / 200.0])
                for lvl in supported}


# ── Global panel path (batched zero-shot inference, no training) ──────────────

def run_global(frame, recipe, cfg, horizon, freq_alias, transform, levels=(80, 95),
               train_cfg=None):
    from models_lib.base_model import future_index
    h = max(int(horizon), 1)
    variant = _variant(train_cfg)
    pipe = _load_pipeline(variant)

    work = frame.copy()
    work["ds"] = pd.to_datetime(work["ds"])
    work = work.sort_values(["unique_id", "ds"])

    series = {}
    for uid, g in work.groupby("unique_id", observed=True, sort=False):
        y = apply_transform(g["y"].to_numpy(), transform)
        series[uid] = {"ds": g["ds"].to_numpy(), "y": y}
    uids = list(series)

    def batched(contexts):
        out = []
        for s in range(0, len(contexts), _BATCH_SERIES):
            out.append(_predict_batch(pipe, contexts[s:s + _BATCH_SERIES], h, [0.5]))
        return np.concatenate(out) if out else np.empty((0, h, 1))

    # Holdout backtest: predict each series' last h periods from the truncated context.
    bt_uids = [u for u in uids if len(series[u]["y"]) >= h + 5]
    cv_df = None
    if bt_uids:
        preds = batched([series[u]["y"][:-h] for u in bt_uids])
        cv_rows = []
        for i, uid in enumerate(bt_uids):
            info = series[uid]
            cv_rows.append(pd.DataFrame({
                "unique_id": uid, "ds": info["ds"][-h:],
                "step": np.arange(1, h + 1),
                "actual": invert_transform(info["y"][-h:], transform),
                "predicted": invert_transform(preds[i, :, 0], transform)}))
        cv_df = pd.concat(cv_rows, ignore_index=True)

    # Final forecast from every series' full context.
    preds = batched([series[u]["y"] for u in uids])
    fc_parts = []
    for i, uid in enumerate(uids):
        fut_ds = future_index(series[uid]["ds"][-1], h, freq_alias)
        fc_parts.append(pd.DataFrame({
            "unique_id": uid, "ds": fut_ds,
            "yhat": invert_transform(preds[i, :, 0], transform)}))
    fc_df = pd.concat(fc_parts, ignore_index=True)

    return {"cv": cv_df, "forecast": fc_df,
            "params": {"backend": "chronos", "variant": variant, "zero_shot": True,
                       "n_series": len(uids), "backtested_series": len(bt_uids)},
            "tuning": None, "strategy": "holdout", "error": None, "_global": True,
            "_fitted": {"backend": "chronos_zero_shot", "variant": variant,
                        "recipe": recipe, "transform": transform}}


def make(recipe: dict, transform: str, train_cfg: dict | None = None):
    return ChronosForecaster(recipe, transform, train_cfg)
