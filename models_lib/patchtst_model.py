"""Hand-rolled PatchTST on pure PyTorch (CPU).

Like NHITS, the reference implementations (neuralforecast, and the research repo's
pipeline) are unusable here — neuralforecast depends on ``coreforecast``, the compiled
kernel that segfaults (0xC0000005) under this venv's numpy 2.4.x. ``transformers`` does
ship ``PatchTSTForPrediction``, but this module hand-rolls the architecture instead so
the training contract (lazy torch, quantile pinball heads, per-series scaling, two-phase
global run) stays byte-identical to ``models_lib/nhits_model.py`` — whose scaffolding
(``_quantiles`` / ``_windows`` / ``_pad_input`` / ``_train_net``) it imports directly.

Architecture (PatchTST, Nie et al. 2023, sized for a CPU-only small box):

* **instance normalization** (RevIN-lite): each input window is normalized by its own
  mean/std in ``forward`` and the quantile outputs are de-normalized — the paper's key
  trick against distribution shift, composing with the outer per-series scaling;
* the window is split into overlapping **patches** (``patch_len`` adapted to the input
  size, stride = half a patch) which are linearly embedded + learned positional encoding;
* a small **transformer encoder** (2 layers, 4 heads, d_model 64/128) attends across
  patches — channel independence is trivial here (univariate windows);
* a flatten head emits **direct multi-horizon quantile forecasts** ``(B, n_q, h)``
  trained with pinball loss — native prediction intervals on the single-series path.

Entry points mirror nhits: ``PatchTSTForecaster`` (single series; subclasses
``NHITSForecaster`` — only ``_fit`` differs, the forward/quantile methods are inherited)
and ``run_global`` (ONE network over all panel series; truncated-data net for the honest
holdout backtest, fresh full-data net for the final forecast).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_lib.base_model import apply_transform, invert_transform
from models_lib.nhits_model import (
    NHITSForecaster, _pad_input, _quantiles, _train_net, _windows,
)

_DEF_EPOCHS = 100
# Half of nhits' cap — attention is heavier per window than the NHITS MLP stacks.
_MAX_WINDOWS = 250_000
_MIN_INPUT = 8  # patching needs >= 2 patches of >= 4 points


def _build_net(input_size: int, h: int, n_q: int, d_model: int = 64,
               n_layers: int = 2, n_heads: int = 4):
    import torch
    from torch import nn

    patch_len = int(min(16, max(4, input_size // 4)))
    patch_len = min(patch_len, input_size)
    stride = max(patch_len // 2, 1)
    n_patches = (input_size - patch_len) // stride + 1

    class _PatchTSTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(patch_len, d_model)
            self.pos = nn.Parameter(torch.zeros(n_patches, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=2 * d_model, dropout=0.1,
                batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                                 enable_nested_tensor=False)
            self.head = nn.Linear(n_patches * d_model, h * n_q)

        def forward(self, x):                     # x: (B, L), per-series-scaled
            mu = x.mean(dim=1, keepdim=True)
            sd = x.std(dim=1, keepdim=True).clamp_min(1e-4)
            z = (x - mu) / sd                     # RevIN-lite instance normalization
            p = z.unfold(1, patch_len, stride)    # (B, n_patches, patch_len)
            enc = self.encoder(self.embed(p) + self.pos)
            out = self.head(enc.flatten(1)).view(-1, n_q, h)
            return out * sd.unsqueeze(1) + mu.unsqueeze(1)

    return _PatchTSTNet()


# ── Single-series forecaster ──────────────────────────────────────────────────

class PatchTSTForecaster(NHITSForecaster):
    """Same fit/predict contract as NHITSForecaster — only the network, the epochs knob
    and the minimum-input guard differ; _forward_last/_predict/predict_quantiles are
    inherited unchanged (they only read self._net/_qs/_mu/_sd/_L/_ys)."""

    def _fit(self, hist):
        import torch  # noqa: F401 — fail fast if torch is missing
        y = hist["y"].to_numpy(dtype=float)
        h = max(int(self.recipe.get("horizon") or 1), 1)
        period = max(int(self.recipe.get("primary_period") or self.period or 1), 1)
        n = len(y)
        L = int(min(max(2 * period, 2 * h), n - h, 512))
        if L < _MIN_INPUT or n < L + 2 * h:
            raise RuntimeError(f"patchtst: series too short ({n} points) for "
                               f"input_size {L} + horizon {h}")
        mu, sd = float(np.nanmean(y)), float(np.nanstd(y))
        sd = sd if sd > 1e-8 else 1.0
        ys = (y - mu) / sd
        X, Y = _windows(ys, L, h)
        if X is None:
            raise RuntimeError("patchtst: no training windows")
        val_mask = np.zeros(len(X), dtype=bool)
        val_mask[-max(1, len(X) // 10):] = True
        self._qs = _quantiles(self.train_cfg.get("interval_levels") or [80, 95])
        epochs = int(self.train_cfg.get("patchtst_max_epochs", _DEF_EPOCHS))
        net = _build_net(L, h, len(self._qs), d_model=64)
        self._net, val_loss = _train_net(net, X, Y, val_mask, self._qs, epochs)
        self._mu, self._sd, self._L, self._h, self._ys = mu, sd, L, h, ys
        self.params_ = {"backend": "torch_patchtst", "input_size": L,
                        "epochs_max": epochs, "quantiles": self._qs,
                        "val_pinball": round(val_loss, 5) if val_loss is not None else None}


# ── Global panel path (one network, all series) ───────────────────────────────

def run_global(frame, recipe, cfg, horizon, freq_alias, transform, levels=(80, 95),
               train_cfg=None):
    from models_lib.base_model import future_index
    train_cfg = train_cfg or {}
    h = max(int(horizon), 1)
    period = max(int(recipe.get("primary_period") or 1), 1)
    qs = _quantiles(train_cfg.get("interval_levels") or levels)
    epochs = int(train_cfg.get("patchtst_max_epochs", _DEF_EPOCHS))

    work = frame.copy()
    work["ds"] = pd.to_datetime(work["ds"])
    work = work.sort_values(["unique_id", "ds"])

    series = {}
    max_len = 0
    for uid, g in work.groupby("unique_id", observed=True, sort=False):
        y = apply_transform(g["y"].to_numpy(), transform)
        cut = max(len(y) - h, 1)
        mu, sd = float(np.nanmean(y[:cut])), float(np.nanstd(y[:cut]))
        sd = sd if sd > 1e-8 else 1.0
        series[uid] = {"ds": g["ds"].to_numpy(), "ys": (y - mu) / sd, "mu": mu, "sd": sd,
                       "y": y}
        max_len = max(max_len, len(y))
    L = int(min(max(2 * period, 2 * h), max(max_len - 2 * h, 4), 512))
    if L < _MIN_INPUT:
        return {"cv": None, "forecast": None, "params": {}, "tuning": None,
                "strategy": "none", "error": "patchtst global: series too short for "
                f"patching (input_size {L} < {_MIN_INPUT}) + horizon {h}"}

    def collect(last_target_offset):
        Xp, Yp, vm = [], [], []
        for info in series.values():
            ys = info["ys"]
            X, Y = _windows(ys, L, h, last_target=len(ys) - last_target_offset)
            if X is None:
                continue
            Xp.append(X)
            Yp.append(Y)
            m = np.zeros(len(X), dtype=bool)
            m[-1] = True  # each series' chronologically-last window is validation
            vm.append(m)
        if not Xp:
            return None, None, None
        return np.concatenate(Xp), np.concatenate(Yp), np.concatenate(vm)

    X, Y, vm = collect(last_target_offset=h)  # holdout targets excluded
    if X is None:
        return {"cv": None, "forecast": None, "params": {}, "tuning": None,
                "strategy": "none", "error": "patchtst global: no series long enough "
                f"for input_size {L} + horizon {h}"}
    if len(X) > _MAX_WINDOWS:
        keep = np.random.RandomState(0).permutation(len(X))[:_MAX_WINDOWS]
        keep = np.union1d(keep, np.where(vm)[0])
        X, Y, vm = X[keep], Y[keep], vm[keep]

    d_model = 128
    net_cv, _ = _train_net(_build_net(L, h, len(qs), d_model), X, Y, vm, qs, epochs)

    import torch
    q50 = qs.index(0.5)
    cv_rows = []
    with torch.no_grad():
        for uid, info in series.items():
            ys = info["ys"]
            if len(ys) < 2 * h + 1:
                continue
            x = _pad_input(ys[:len(ys) - h], L)
            out = net_cv(torch.tensor(x[None]))[0].numpy()
            pred = out[q50] * info["sd"] + info["mu"]
            cv_rows.append(pd.DataFrame({
                "unique_id": uid, "ds": info["ds"][-h:],
                "step": np.arange(1, h + 1),
                "actual": invert_transform(info["y"][-h:], transform),
                "predicted": invert_transform(pred, transform)}))
    cv_df = pd.concat(cv_rows, ignore_index=True) if cv_rows else None

    # Final model: fresh net on ALL windows (holdout included), then per-series forecast.
    Xf, Yf, vmf = collect(last_target_offset=0)
    if len(Xf) > _MAX_WINDOWS:
        keep = np.random.RandomState(0).permutation(len(Xf))[:_MAX_WINDOWS]
        keep = np.union1d(keep, np.where(vmf)[0])
        Xf, Yf, vmf = Xf[keep], Yf[keep], vmf[keep]
    net, _ = _train_net(_build_net(L, h, len(qs), d_model), Xf, Yf, vmf, qs, epochs)

    fc_parts = []
    with torch.no_grad():
        for uid, info in series.items():
            out = net(torch.tensor(_pad_input(info["ys"], L)[None]))[0].numpy()
            pred = out[q50] * info["sd"] + info["mu"]
            fut_ds = future_index(info["ds"][-1], h, freq_alias)
            fc_parts.append(pd.DataFrame({
                "unique_id": uid, "ds": fut_ds,
                "yhat": invert_transform(pred, transform)}))
    fc_df = pd.concat(fc_parts, ignore_index=True)

    return {"cv": cv_df, "forecast": fc_df,
            "params": {"backend": "torch_patchtst", "input_size": L, "d_model": d_model,
                       "epochs_max": epochs, "quantiles": qs, "n_windows": int(len(Xf))},
            "tuning": None, "strategy": "holdout", "error": None, "_global": True,
            "_fitted": {"backend": "patchtst_global", "model": net, "input_size": L,
                        "recipe": recipe, "transform": transform}}


def make(recipe: dict, transform: str, train_cfg: dict | None = None):
    return PatchTSTForecaster(recipe, transform, train_cfg)
