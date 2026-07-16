"""Hand-rolled NHITS on pure PyTorch (CPU).

The official ``neuralforecast`` package cannot run here: it depends on ``coreforecast``,
the SAME compiled kernel that segfaults (0xC0000005) under this venv's numpy 2.4.x —
verified again on 2026-07-09. So this module implements the NHITS architecture directly
on torch (which imports and runs fine):

* stacked MLP blocks, each with input **max-pooling** at a different rate (coarse → fine)
  so successive stacks specialize on different frequency bands;
* each block emits a **backcast** (subtracted from the residual input, doubly-residual
  stacking) and a low-dimensional **forecast** that is linearly **interpolated** up to the
  horizon — the "hierarchical interpolation" that keeps parameters small;
* **direct multi-horizon** output with **quantile heads** (pinball loss over P50 + the
  interval alphas derived from ``training.interval_levels``) — native prediction
  intervals, no conformal needed;
* per-series standard scaling, Adam, early stopping on the chronologically-last windows.

Two entry points mirror ``ml_models``: ``NHITSForecaster`` (single series, standard
``Forecaster`` interface for the walk-forward harness) and ``run_global`` (ONE network
trained on windows from every series of a panel — the honest scheme: a truncated-data
model produces the holdout backtest, a fresh full-data model produces the final forecast).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from models_lib.base_model import Forecaster, apply_transform, invert_transform

_DEF_EPOCHS = 150
_MAX_WINDOWS = 500_000  # global training-window cap (random subsample, val kept)


def _quantiles(levels) -> list[float]:
    qs = {0.5}
    for lvl in levels:
        a = (100 - int(lvl)) / 200.0
        qs.update((a, 1 - a))
    return sorted(qs)


# ── Network ───────────────────────────────────────────────────────────────────

def _build_net(input_size: int, h: int, n_q: int, hidden: int):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class _Block(nn.Module):
        def __init__(self, pool: int, knots: int):
            super().__init__()
            self.pool = nn.MaxPool1d(pool, stride=pool, ceil_mode=True) if pool > 1 else None
            in_dim = math.ceil(input_size / pool)
            self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
            self.back = nn.Linear(hidden, input_size)
            self.fore = nn.Linear(hidden, knots * n_q)
            self.knots = knots

        def forward(self, x):
            z = x if self.pool is None else self.pool(x.unsqueeze(1)).squeeze(1)
            z = self.mlp(z)
            b = self.back(z)
            f = self.fore(z).view(-1, n_q, self.knots)
            if self.knots == 1:
                f = f.expand(-1, -1, h)
            elif self.knots != h:
                f = F.interpolate(f, size=h, mode="linear", align_corners=True)
            return b, f

    class _NHITSNet(nn.Module):
        def __init__(self):
            super().__init__()
            pools = [max(min(input_size // 4, 8), 1), max(min(input_size // 8, 4), 1), 1]
            knots = [max(h // 4, 1), max(h // 2, 1), h]
            self.blocks = nn.ModuleList([_Block(p, k) for p, k in zip(pools, knots)])

        def forward(self, x):
            residual, out = x, None
            for blk in self.blocks:
                b, f = blk(residual)
                residual = residual - b
                out = f if out is None else out + f
            return out  # (B, n_q, h)

    return _NHITSNet()


def _train_net(net, X, Y, val_mask, quantiles, epochs, lr=1e-3, batch=256, seed=0):
    import torch
    from pipeline import resource_limits
    torch.set_num_threads(resource_limits.thread_cap())  # keep a core free for the OS
    torch.manual_seed(seed)
    q = torch.tensor(quantiles, dtype=torch.float32).view(1, -1, 1)

    def pinball(pred, target):
        diff = target.unsqueeze(1) - pred
        return torch.mean(torch.maximum(q * diff, (q - 1) * diff))

    Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
    Yt = torch.tensor(np.asarray(Y), dtype=torch.float32)
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    patience = max(10, int(epochs) // 10)
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(int(epochs)):
        net.train()
        perm = np.random.RandomState(seed + ep).permutation(tr_idx)
        for s in range(0, len(perm), batch):
            bidx = perm[s:s + batch]
            opt.zero_grad()
            loss = pinball(net(Xt[bidx]), Yt[bidx])
            loss.backward()
            opt.step()
        if len(va_idx):
            net.eval()
            with torch.no_grad():
                vl = float(pinball(net(Xt[va_idx]), Yt[va_idx]))
            if vl < best_val - 1e-6:
                best_val, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, (best_val if len(va_idx) else None)


def _windows(ys: np.ndarray, L: int, h: int, last_target: int | None = None):
    """Sliding (input, target) windows; ``last_target`` caps the final target index
    (exclusive) so holdout targets can be excluded from training."""
    n = last_target if last_target is not None else len(ys)
    n_w = n - L - h + 1
    if n_w < 1:
        return None, None
    X = np.stack([ys[i:i + L] for i in range(n_w)])
    Y = np.stack([ys[i + L:i + L + h] for i in range(n_w)])
    return X, Y


def _pad_input(ys: np.ndarray, L: int) -> np.ndarray:
    """Last L scaled values, left-padded with 0 (= the series mean) when short."""
    x = ys[-L:] if len(ys) >= L else np.concatenate([np.zeros(L - len(ys)), ys])
    return x.astype(np.float32)


# ── Single-series forecaster ──────────────────────────────────────────────────

class NHITSForecaster(Forecaster):
    uses_features = False

    def __init__(self, recipe: dict, transform: str = "none", train_cfg: dict | None = None):
        super().__init__(transform)
        self.recipe = recipe
        self.train_cfg = train_cfg or {}

    def _fit(self, hist):
        import torch  # noqa: F401 — fail fast if torch is missing
        y = hist["y"].to_numpy(dtype=float)
        h = max(int(self.recipe.get("horizon") or 1), 1)
        period = max(int(self.recipe.get("primary_period") or self.period or 1), 1)
        n = len(y)
        L = int(min(max(2 * period, 2 * h), n - h))
        if L < 3 or n < L + 2 * h:
            raise RuntimeError(f"nhits: series too short ({n} points) for "
                               f"input_size {L} + horizon {h}")
        mu, sd = float(np.nanmean(y)), float(np.nanstd(y))
        sd = sd if sd > 1e-8 else 1.0
        ys = (y - mu) / sd
        X, Y = _windows(ys, L, h)
        if X is None:
            raise RuntimeError("nhits: no training windows")
        val_mask = np.zeros(len(X), dtype=bool)
        val_mask[-max(1, len(X) // 10):] = True
        self._qs = _quantiles(self.train_cfg.get("interval_levels") or [80, 95])
        epochs = int(self.train_cfg.get("nhits_max_epochs", _DEF_EPOCHS))
        net = _build_net(L, h, len(self._qs), hidden=min(256, max(64, 8 * L)))
        self._net, val_loss = _train_net(net, X, Y, val_mask, self._qs, epochs)
        self._mu, self._sd, self._L, self._h, self._ys = mu, sd, L, h, ys
        self.params_ = {"backend": "torch", "input_size": L, "epochs_max": epochs,
                        "quantiles": self._qs,
                        "val_pinball": round(val_loss, 5) if val_loss is not None else None}

    def _forward_last(self) -> np.ndarray:
        import torch
        with torch.no_grad():
            out = self._net(torch.tensor(_pad_input(self._ys, self._L)[None]))
        return out[0].numpy()  # (n_q, h)

    def _predict(self, h, future_ds, future_exog=None):
        out = self._forward_last()
        med = out[self._qs.index(0.5)][:int(h)]
        return med * self._sd + self._mu  # transformed scale; base inverts

    def predict_quantiles(self, h: int, future_ds, levels=(80, 95)) -> dict:
        out = np.sort(self._forward_last(), axis=0)  # non-crossing across quantiles
        vals = {q: invert_transform(out[i][:int(h)] * self._sd + self._mu, self.transform)
                for i, q in enumerate(self._qs)}
        if np.nanmin(self._y_raw) >= 0:
            vals = {q: np.clip(v, 0, None) for q, v in vals.items()}
        res = {}
        for lvl in levels:
            a = (100 - int(lvl)) / 200.0
            if a in vals and (1 - a) in vals:
                res[int(lvl)] = (vals[a], vals[1 - a])
        return res


# ── Global panel path (one network, all series) ───────────────────────────────

def run_global(frame, recipe, cfg, horizon, freq_alias, transform, levels=(80, 95),
               train_cfg=None):
    from models_lib.base_model import future_index
    train_cfg = train_cfg or {}
    h = max(int(horizon), 1)
    period = max(int(recipe.get("primary_period") or 1), 1)
    qs = _quantiles(train_cfg.get("interval_levels") or levels)
    epochs = int(train_cfg.get("nhits_max_epochs", _DEF_EPOCHS))

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

    def collect(last_target_offset):
        """Stack windows over all series; targets end at n - offset per series."""
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
                "strategy": "none", "error": "nhits global: no series long enough "
                f"for input_size {L} + horizon {h}"}
    if len(X) > _MAX_WINDOWS:
        keep = np.random.RandomState(0).permutation(len(X))[:_MAX_WINDOWS]
        keep = np.union1d(keep, np.where(vm)[0])
        X, Y, vm = X[keep], Y[keep], vm[keep]

    hidden = min(512, max(128, 8 * L))
    net_cv, _ = _train_net(_build_net(L, h, len(qs), hidden), X, Y, vm, qs, epochs)

    # Holdout backtest: predict each series' last h periods from the pre-holdout window.
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
    net, _ = _train_net(_build_net(L, h, len(qs), hidden), Xf, Yf, vmf, qs, epochs)

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
            "params": {"backend": "torch", "input_size": L, "hidden": hidden,
                       "epochs_max": epochs, "quantiles": qs, "n_windows": int(len(Xf))},
            "tuning": None, "strategy": "holdout", "error": None, "_global": True,
            "_fitted": {"backend": "nhits_global", "model": net, "input_size": L,
                        "recipe": recipe, "transform": transform}}


def make(recipe: dict, transform: str, train_cfg: dict | None = None):
    return NHITSForecaster(recipe, transform, train_cfg)
