"""Stage 4: Forecasting EDA — runs on CLEAN data only, answers one question:
"What forecasting model fits this data?"

Pure Python statistics (no LLM in computation; Stage 5 will only ever see
model_selection_payload.json). The user-facing choices — timestamp, target (+aggregation
including event counts), scope, series key, exogenous, forecast frequency, horizon —
were confirmed back at Stage 2.5 (pipeline/forecast_intent.py + the Setup & Cleaning
tab) and are read from runs/{id}/forecast_user_selections.json.

run(run_id) builds the analysis series per the confirmed intent (aggregate / per-series
panel, sum / mean / distinct-count per period) and computes every model-selection
statistic: trend, seasonality (STL + FFT + ACF), stationarity (ADF/KPSS/PP diffs),
forecastability entropies, intermittency (ADI/CV²), SNR, CV, exogenous cross-correlation,
VIF, heteroskedasticity, structural breaks, Ljung-Box residual whiteness.

Writes:
  runs/{id}/forecasting_eda_full.json      — all stats + downsampled plot arrays
  runs/{id}/model_selection_payload.json   — compact decision JSON for Stage 5
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline import progress

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
CLEANED_DIR = ROOT / "data" / "cleaned"
CONFIG_PATH = ROOT / "config" / "settings.yaml"

# frequency label → pandas offset alias. Weekly is anchored later to the actual weekday
# observed in the data (resampling to W-MON when the data is W-SUN shifts every point).
_FREQ_TO_PANDAS = {
    "hourly": "h", "daily": "D", "weekly": "W",
    "monthly": "MS", "quarterly": "QS", "yearly": "YS",
}
# Candidate seasonal periods per analysis frequency. Long secondary cycles (365 for
# daily, 168 for hourly) are only tested when the series actually spans ≥ 2 cycles.
_FREQ_PERIODS = {
    "hourly": [24, 168], "daily": [7, 365], "weekly": [52],
    "monthly": [12], "quarterly": [4], "yearly": [],
}
_DEFAULT_HORIZON = {"hourly": 48, "daily": 30, "weekly": 13, "monthly": 12, "quarterly": 8, "yearly": 3}


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("forecast_eda", {})


def _f(x, nd: int = 4):
    """Round to a plain JSON float; None for NaN/inf/None."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, nd)


def _paths(run_id: str) -> tuple[Path, Path]:
    run_dir = RUNS_DIR / run_id
    parquet = CLEANED_DIR / f"{run_id}_cleaned.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"Cleaned parquet not found: {parquet}. Run the Cleaning stage first.")
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}.")
    return run_dir, parquet


def _recipe(run_dir: Path) -> dict:
    p = run_dir / "cleaning_recipe.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _resolve_selections(run_dir: Path) -> dict:
    """The confirmed forecast intent, with per-field fallbacks for older runs:
    forecast_user_selections.json → forecast_intent.json suggestions → cleaning recipe
    (which always carries timestamp_col + data frequency)."""
    sel_path = run_dir / "forecast_user_selections.json"
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    intent_path = run_dir / "forecast_intent.json"
    intent = json.loads(intent_path.read_text()) if intent_path.exists() else {}
    recipe = _recipe(run_dir)

    target = sel.get("target_col") or intent.get("target", {}).get("suggested")
    if not target:
        raise ValueError(
            "No forecast target confirmed. Run Stage 2.5 (forecast intent) and confirm "
            "the settings on the Pipeline Setup tab first.")
    ts_col = (sel.get("timestamp_col") or recipe.get("timestamp_col")
              or intent.get("timestamp", {}).get("suggested"))
    if not ts_col:
        raise ValueError("No timestamp column available for this run.")
    data_freq = (recipe.get("frequency")
                 or intent.get("frequency", {}).get("data") or "daily")
    fcst_freq = (sel.get("forecast_frequency")
                 or intent.get("frequency", {}).get("suggested") or data_freq)
    exog = sel.get("exog_cols")
    if exog is None:
        exog = [c for c in intent.get("exogenous", {}).get("candidates", []) if c != target]
    return {
        "target_col": target,
        "agg": sel.get("agg") or intent.get("target", {}).get("suggested_agg") or "sum",
        "timestamp_col": ts_col,
        "scope": sel.get("scope") or intent.get("scope", {}).get("suggested", "aggregate"),
        "group_cols": sel.get("group_cols") or intent.get("group", {}).get("suggested", []),
        "exog_cols": exog,
        "forecast_frequency": fcst_freq,
        "horizon": int(sel.get("horizon") or intent.get("horizon", {}).get("suggested")
                       or _DEFAULT_HORIZON.get(fcst_freq, 12)),
        "data_frequency": data_freq,
        # Training window (optional, default = full data). Set from the Forecast EDA
        # tab's advanced options; consumed by Stage 4 AND Stage 6 via apply_train_filter.
        "train_start": sel.get("train_start") or None,
        "train_end": sel.get("train_end") or None,
        "exclude_ranges": sel.get("exclude_ranges") or [],
    }


def apply_train_filter(df: pd.DataFrame, ts_col: str, sel: dict) -> tuple[pd.DataFrame, dict]:
    """Restrict the cleaned frame to the confirmed training window (Stage 4 + Stage 6).

    ``train_start``/``train_end`` bound the series (inclusive, whole days);
    ``exclude_ranges`` drops rows inside each [start, end] window — the resample grid
    then has empty bins there and the standard empty-bin fill rules apply unchanged
    (count/zero-inflated-sum → 0, else interpolate). Returns (filtered_df, report);
    the report always carries the FULL data span so the UI's range picker keeps its
    original bounds even after a cut is applied."""
    start, end = sel.get("train_start"), sel.get("train_end")
    excludes = [p for p in (sel.get("exclude_ranges") or []) if p and len(p) == 2]
    report = {
        "train_start": start, "train_end": end, "exclude_ranges": excludes,
        "data_start": str(df[ts_col].min()), "data_end": str(df[ts_col].max()),
        "rows_before": int(len(df)), "active": bool(start or end or excludes),
    }
    if not report["active"]:
        report["rows_after"] = report["rows_before"]
        return df, report
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df[ts_col] >= pd.Timestamp(start)
    if end:  # inclusive of the whole end day (timestamps may carry a time of day)
        mask &= df[ts_col] < pd.Timestamp(end) + pd.Timedelta(days=1)
    for pair in excludes:
        try:
            ex0 = pd.Timestamp(pair[0])
            ex1 = pd.Timestamp(pair[1]) + pd.Timedelta(days=1)
        except (ValueError, TypeError):
            continue  # malformed pair — ignore rather than kill the run
        mask &= ~((df[ts_col] >= ex0) & (df[ts_col] < ex1))
    out = df.loc[mask]
    report["rows_after"] = int(len(out))
    if out.empty:
        raise ValueError(
            "The training range/exclusions filtered out every row — widen the range "
            "or remove an exclusion.")
    return out, report


def _pandas_freq(label: str, sample_ts: pd.Series) -> str:
    """Pandas offset alias for a frequency label; weekly anchored to the observed weekday."""
    alias = _FREQ_TO_PANDAS.get(label, "D")
    if label == "weekly" and len(sample_ts) > 0:
        anchor = sample_ts.dt.day_name().mode()
        if len(anchor) > 0:
            alias = "W-" + anchor.iloc[0][:3].upper()
    return alias


def _build_series(frame: pd.DataFrame, ts_col: str, target: str, exog_cols: list[str],
                  agg: str, freq_label: str) -> tuple[pd.Series, pd.DataFrame, dict]:
    """Collapse rows onto a regular time grid at freq_label.

    agg semantics:
      sum / mean — aggregate a numeric measure. Empty grid bins become NaN and are then
        filled: with 0 when the data is zero-inflated sum-aggregated demand (an absent
        period IS zero demand — required for honest ADI/intermittency), otherwise by
        time interpolation (a genuine gap in a continuous measure).
      nunique / count — events per period (distinct IDs / non-null rows). Empty bins are
        TRUE zeros (no events happened), never missing.
    """
    alias = _pandas_freq(freq_label, frame[ts_col])
    indexed = frame.set_index(ts_col).sort_index()

    if agg == "nunique":
        y = indexed[target].resample(alias).nunique().astype("float64")
    elif agg == "count":
        y = indexed[target].resample(alias).count().astype("float64")
    elif agg == "mean":
        y = indexed[target].resample(alias).mean()
    else:
        y = indexed[target].resample(alias).sum(min_count=1)  # min_count: empty bin → NaN, not 0
    exog = indexed[exog_cols].resample(alias).mean() if exog_cols else pd.DataFrame(index=y.index)

    n_grid = len(y)
    n_missing = int(y.isna().sum())
    observed = y.dropna()
    zero_pct = float((observed == 0).mean() * 100) if len(observed) else 0.0

    if agg in ("nunique", "count"):
        fill_method = "count_zero"  # empty bins are already honest zeros
    else:
        fill_method = "zero" if (agg == "sum" and zero_pct > 10) else "interpolate"
        if n_missing > 0:
            if fill_method == "zero":
                y = y.fillna(0.0)
            else:
                y = y.interpolate(method="time", limit_direction="both")
    if not exog.empty:
        exog = exog.interpolate(method="time", limit_direction="both")

    report = {
        "grid_points": n_grid,
        "missing_grid_points": n_missing,
        "missing_pct": _f(n_missing / max(n_grid, 1) * 100, 2),
        "fill_method": fill_method if (n_missing or agg in ("nunique", "count")) else "none",
        "zero_pct": _f(float((y == 0).mean() * 100), 2),
    }
    return y, exog, report


# ── Single-series statistics ─────────────────────────────────────────────────

def _decompose(y: pd.Series, period: int, stl_max_points: int):
    """STL for normal lengths; classical moving-average decomposition for very long
    series (STL's iterative LOESS costs minutes past ~50k points). Returns
    (trend, seasonal, resid) as aligned numpy arrays, or None if not decomposable."""
    if period < 2 or len(y) < 2 * period + 1:
        return None
    try:
        if len(y) > stl_max_points:
            from statsmodels.tsa.seasonal import seasonal_decompose
            res = seasonal_decompose(y, period=period, model="additive", extrapolate_trend="freq")
        else:
            from statsmodels.tsa.seasonal import STL
            res = STL(y, period=period, robust=True).fit()
        trend = np.asarray(res.trend, dtype=float)
        seasonal = np.asarray(res.seasonal, dtype=float)
        resid = np.asarray(res.resid, dtype=float)
        ok = np.isfinite(trend) & np.isfinite(seasonal) & np.isfinite(resid)
        if ok.sum() < period:
            return None
        return trend[ok], seasonal[ok], resid[ok]
    except Exception:
        return None


def _strengths(trend: np.ndarray, seasonal: np.ndarray, resid: np.ndarray) -> dict:
    """Hyndman-style component strengths + SNR."""
    var_r = float(np.var(resid))
    trend_strength = max(0.0, 1.0 - var_r / max(float(np.var(trend + resid)), 1e-12))
    seasonal_strength = max(0.0, 1.0 - var_r / max(float(np.var(seasonal + resid)), 1e-12))
    snr = float(np.var(trend + seasonal)) / max(var_r, 1e-12)
    return {
        "trend_strength": _f(trend_strength),
        "seasonal_strength": _f(seasonal_strength),
        "snr": _f(snr, 2),
    }


def _fft_periods(y: np.ndarray, max_period: int) -> list[dict]:
    """Dominant cycles via periodogram of the detrended series (top 5 by power share)."""
    n = len(y)
    if n < 16:
        return []
    x = np.arange(n, dtype=float)
    detrended = y - np.polyval(np.polyfit(x, y, 1), x)
    from scipy.signal import periodogram
    freqs, power = periodogram(detrended, detrend=False)
    mask = freqs > 0
    freqs, power = freqs[mask], power[mask]
    if power.sum() <= 0:
        return []
    periods = 1.0 / freqs
    valid = (periods >= 2) & (periods <= min(max_period, n / 2))
    freqs, power, periods = freqs[valid], power[valid], periods[valid]
    if len(power) == 0:
        return []
    top = np.argsort(power)[::-1][:5]
    total = float(power.sum())
    return [
        {"period": _f(periods[i], 1), "power_share": _f(power[i] / total, 3)}
        for i in top if power[i] / total > 0.01
    ]


def _acf_pacf(y: np.ndarray, nlags: int) -> dict:
    from statsmodels.tsa.stattools import acf, pacf
    n = len(y)
    nlags = min(nlags, n // 2 - 1)
    if nlags < 2:
        return {"acf": [], "pacf": [], "significant_lags": [], "conf_bound": None}
    acf_vals = acf(y, nlags=nlags, fft=True)
    try:
        pacf_vals = pacf(y, nlags=min(nlags, n // 2 - 2), method="ywm")
    except Exception:
        pacf_vals = np.array([])
    bound = 1.96 / np.sqrt(n)
    significant = [int(k) for k in range(1, len(acf_vals)) if abs(acf_vals[k]) > bound]
    return {
        "acf": [_f(v) for v in acf_vals[1:]],
        "pacf": [_f(v) for v in pacf_vals[1:]] if len(pacf_vals) else [],
        "significant_lags": significant,
        "conf_bound": _f(bound),
    }


def _stationarity(y: np.ndarray, period: int, max_points: int) -> dict:
    """ADF + KPSS p-values and the differencing consensus (ADF/KPSS/PP via pmdarima)."""
    x = y[-max_points:]  # unit-root tests past a few thousand points: slow, no extra power
    out = {"adf_p": None, "kpss_p": None, "ndiffs": None, "nsdiffs": None, "stationary": None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from statsmodels.tsa.stattools import adfuller
            out["adf_p"] = _f(adfuller(x, autolag="AIC")[1])
        except Exception:
            pass
        try:
            from statsmodels.tsa.stattools import kpss
            out["kpss_p"] = _f(kpss(x, regression="c", nlags="auto")[1])
        except Exception:
            pass
        try:
            from pmdarima.arima.utils import ndiffs
            out["ndiffs"] = int(max(
                ndiffs(x, test="adf"), ndiffs(x, test="kpss"), ndiffs(x, test="pp")))
        except Exception:
            pass
        if period >= 2 and len(x) >= 2 * period + 10:
            try:
                from pmdarima.arima.utils import nsdiffs
                out["nsdiffs"] = int(nsdiffs(x, m=period, test="ocsb"))
            except Exception:
                out["nsdiffs"] = None
    # stationary = ADF rejects unit root AND KPSS fails to reject stationarity
    if out["adf_p"] is not None and out["kpss_p"] is not None:
        out["stationary"] = bool(out["adf_p"] < 0.05 and out["kpss_p"] > 0.05)
    elif out["ndiffs"] is not None:
        out["stationary"] = out["ndiffs"] == 0
    return out


def _forecastability(y: np.ndarray, max_points: int) -> dict:
    """Entropy family. Score = 1 − normalized spectral entropy (1 = clean signal)."""
    out = {"spectral_entropy": None, "sample_entropy": None, "approx_entropy": None,
           "dfa_alpha": None, "score": None}
    x = y[-max_points * 3:]
    std = x.std()
    z = (x - x.mean()) / std if std > 0 else x - x.mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import antropy as ant
            out["spectral_entropy"] = _f(ant.spectral_entropy(x, sf=1.0, method="welch",
                                                              normalize=True))
            zc = z[-max_points:]  # O(n²) — hard cap
            out["sample_entropy"] = _f(ant.sample_entropy(zc))
            out["approx_entropy"] = _f(ant.app_entropy(zc))
            out["dfa_alpha"] = _f(ant.detrended_fluctuation(z[-max_points * 3:]))
        except Exception:
            pass
    if out["spectral_entropy"] is not None:
        out["score"] = _f(1.0 - out["spectral_entropy"])
    return out


def _intermittency(y: np.ndarray) -> dict:
    """ADI / CV² and the Syntetos-Boylan quadrant (smooth/intermittent/erratic/lumpy)."""
    nz_mask = y != 0
    n_nz = int(nz_mask.sum())
    if n_nz == 0:
        return {"adi": None, "cv2": None, "class": "no_demand", "zero_pct": 100.0}
    adi = len(y) / n_nz
    nz = y[nz_mask]
    mean_nz = float(nz.mean())
    cv2 = float((nz.std() / mean_nz) ** 2) if mean_nz != 0 else None
    if cv2 is None:
        cls = "smooth"
    elif adi < 1.32 and cv2 < 0.49:
        cls = "smooth"
    elif adi >= 1.32 and cv2 < 0.49:
        cls = "intermittent"
    elif adi < 1.32:
        cls = "erratic"
    else:
        cls = "lumpy"
    return {"adi": _f(adi, 2), "cv2": _f(cv2, 3), "class": cls,
            "zero_pct": _f((1 - n_nz / len(y)) * 100, 2)}


def _trend_stats(y: np.ndarray) -> dict:
    """OLS slope (overall + last-20% window) scaled to %-of-mean per period."""
    n = len(y)
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    scale = abs(y.mean()) if abs(y.mean()) > 1e-12 else 1.0
    rel_slope = slope / scale * 100  # % of mean level, per period

    tail = y[-max(n // 5, 10):]
    xt = np.arange(len(tail), dtype=float)
    recent_slope = np.polyfit(xt, tail, 1)[0] / scale * 100 if len(tail) >= 3 else None

    if abs(rel_slope) < 0.01:
        direction = "flat"
    else:
        direction = "up" if rel_slope > 0 else "down"
    return {
        "slope_pct_per_period": _f(rel_slope),
        "recent_slope_pct_per_period": _f(recent_slope),
        "linear_r2": _f(r2),
        "direction": direction,
    }


def _heteroskedasticity(trend: np.ndarray, resid: np.ndarray, y: pd.Series) -> dict:
    """Does noise scale with level? → multiplicative model / log transform hint."""
    out = {"level_resid_corr": None, "std_ratio": None, "heteroskedastic": None,
           "transform_recommendation": "none"}
    if len(resid) < 20:
        return out
    abs_r = np.abs(resid)
    if trend.std() > 0 and abs_r.std() > 0:
        out["level_resid_corr"] = _f(np.corrcoef(trend, abs_r)[0, 1])
    half = len(resid) // 2
    s1, s2 = abs_r[:half].std(), abs_r[half:].std()
    if min(s1, s2) > 0:
        out["std_ratio"] = _f(max(s1, s2) / min(s1, s2), 2)
    corr = out["level_resid_corr"] or 0.0
    ratio = out["std_ratio"] or 1.0
    out["heteroskedastic"] = bool(abs(corr) > 0.4 or ratio > 2.0)
    skew = float(pd.Series(y).skew())
    if out["heteroskedastic"] and float(y.min()) >= 0 and skew > 1.0:
        out["transform_recommendation"] = "log"
    return out


def _structural_breaks(y: pd.Series, max_points: int) -> dict:
    """PELT breakpoints on the analysis series; a break in the last 20% is a warning."""
    out = {"break_dates": [], "recent_break": False}
    try:
        import ruptures as rpt
    except ImportError:
        return out
    if len(y) < 50:
        return out
    stride = max(1, len(y) // max_points)
    sub = y.iloc[::stride]
    vals = sub.to_numpy(dtype=float)
    try:
        var = np.var(vals) or 1.0
        bps = rpt.Pelt(model="rbf").fit(vals).predict(pen=np.log(len(vals)) * var)
        bps = [b for b in bps if b < len(sub)]
        out["break_dates"] = [str(sub.index[b - 1].date()) for b in bps if b >= 1]
        out["recent_break"] = any(b >= 0.8 * len(sub) for b in bps)
    except Exception:
        pass
    return out


def _ljung_box(resid: np.ndarray, period: int) -> float | None:
    """p-value of leftover autocorrelation in decomposition residuals. Low p ⇒ structure
    the trend+seasonal split didn't capture ⇒ ARMA-family residual modeling has value."""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lags = min(max(2 * period, 10), 40, len(resid) // 4)
        if lags < 1:
            return None
        res = acorr_ljungbox(resid, lags=[lags])
        return _f(res["lb_pvalue"].iloc[0])
    except Exception:
        return None


def _downsample(y: pd.Series, max_points: int) -> dict:
    stride = max(1, int(np.ceil(len(y) / max_points)))
    sub = y.iloc[::stride]
    return {"x": [str(t) for t in sub.index], "y": [_f(v) for v in sub.to_numpy()]}


def _series_stats(y: pd.Series, freq_label: str, cfg: dict) -> dict:
    """The full statistics battery for one regular series (DatetimeIndex, NaN-free)."""
    vals = y.to_numpy(dtype=float)
    n = len(vals)
    acf_lags = cfg.get("acf_lags", 40)
    stl_max = cfg.get("stl_max_points", 50_000)
    stat_max = cfg.get("stat_test_max_points", 5_000)
    ent_max = cfg.get("entropy_max_points", 2_000)
    plot_max = cfg.get("plot_max_points", 1_500)

    # Candidate seasonal periods with ≥ 2 complete cycles in the data
    candidates = [p for p in _FREQ_PERIODS.get(freq_label, []) if n >= 2 * p + 1]

    # Decompose at each candidate; primary period = strongest seasonal signal
    seasonality, best = [], None
    for p in candidates:
        comp = _decompose(y, p, stl_max)
        if comp is None:
            continue
        strengths = _strengths(*comp)
        seasonality.append({"period": p, **strengths})
        if best is None or strengths["seasonal_strength"] > best["strengths"]["seasonal_strength"]:
            best = {"period": p, "comp": comp, "strengths": strengths}

    fft_peaks = _fft_periods(vals, max_period=max(candidates, default=n // 2))

    if best is not None:
        primary_period = best["period"]
        trend_c, seasonal_c, resid_c = best["comp"]
        strengths = best["strengths"]
        het = _heteroskedasticity(trend_c, resid_c, y)
        resid_lb_p = _ljung_box(resid_c, primary_period)
    else:
        primary_period = candidates[0] if candidates else 1
        trend_c = seasonal_c = resid_c = None
        strengths = {"trend_strength": None, "seasonal_strength": None, "snr": None}
        het = _heteroskedasticity(np.array([]), np.array([]), y)
        resid_lb_p = None

    mean = float(vals.mean())
    cv = _f(float(vals.std()) / abs(mean), 4) if abs(mean) > 1e-12 else None

    stats = {
        "length": n,
        "start": str(y.index.min()),
        "end": str(y.index.max()),
        "mean": _f(mean, 3),
        "std": _f(float(vals.std()), 3),
        "cv": cv,
        "trend": _trend_stats(vals),
        "seasonality": {
            "candidates": seasonality,
            "primary_period": primary_period if seasonality else None,
            "fft_peaks": fft_peaks,
        },
        **strengths,
        "stationarity": _stationarity(vals, primary_period, stat_max),
        "forecastability": _forecastability(vals, ent_max),
        "intermittency": _intermittency(vals),
        "autocorrelation": _acf_pacf(vals[-stat_max * 4:], acf_lags),
        "heteroskedasticity": het,
        "residual_ljung_box_p": resid_lb_p,
        "structural_breaks": _structural_breaks(y, stat_max),
    }

    plot_data = {"series": _downsample(y, plot_max)}
    if trend_c is not None and len(trend_c) == n:
        idx = y.index
        plot_data["decomposition"] = {
            "trend": _downsample(pd.Series(trend_c, index=idx), plot_max),
            "seasonal": _downsample(pd.Series(seasonal_c, index=idx), plot_max),
            "resid": _downsample(pd.Series(resid_c, index=idx), plot_max),
            "period": primary_period,
        }
    stats["plot_data"] = plot_data
    return stats


# ── Exogenous analysis ───────────────────────────────────────────────────────

def _exog_analysis(y: pd.Series, exog: pd.DataFrame, period: int, cfg: dict) -> dict:
    """Cross-correlation (which lags of each driver move the target) + VIF."""
    if exog.empty:
        return {"has_exogenous": False, "cross_correlation": [], "vif": [], "note": None}

    max_lag = int(min(max(2 * period, 14), 40, len(y) // 3))
    ys = (y - y.mean()) / (y.std() or 1.0)
    cross = []
    for col in exog.columns:
        e = exog[col]
        if e.std() == 0 or e.isna().all():
            continue
        es = (e - e.mean()) / e.std()
        lag0 = float(np.corrcoef(ys, es)[0, 1])
        best_lag, best_corr = 0, lag0
        for lag in range(1, max_lag + 1):
            # positive lag = exog LEADS target (the only direction usable for forecasting)
            c = float(np.corrcoef(ys[lag:], es[:-lag])[0, 1])
            if abs(c) > abs(best_corr):
                best_lag, best_corr = lag, c
        cross.append({"col": col, "corr_lag0": _f(lag0, 3),
                      "best_lag": best_lag, "best_corr": _f(best_corr, 3)})
    cross.sort(key=lambda r: -abs(r["best_corr"] or 0))

    vif_rows = []
    cols = [c for c in exog.columns if exog[c].std() > 0]
    if len(cols) >= 2:
        sample = exog[cols].dropna()
        if len(sample) > 10_000:
            sample = sample.sample(10_000, random_state=0)
        if len(sample) > len(cols) + 2:
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                X = np.column_stack([np.ones(len(sample)), sample.to_numpy(dtype=float)])
                for i, col in enumerate(cols):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        vif_rows.append({"col": col,
                                         "vif": _f(variance_inflation_factor(X, i + 1), 2)})
            except Exception:
                pass

    return {
        "has_exogenous": len(cross) > 0,
        "cross_correlation": cross,
        "vif": vif_rows,
        "note": ("Exogenous usefulness assumes future values will be available at "
                 "forecast time (known drivers like promotions/prices)."),
    }


# ── Panel summary (per-series scope) ─────────────────────────────────────────

def _panel_summary(df: pd.DataFrame, ts_col: str, target: str, group_cols: list[str],
                   agg: str, alias: str, grid_len: int) -> dict:
    """Vectorized per-series profile of the whole panel — this is what Stage 5 needs to
    route global-ML vs per-series-classical (per-series classical cost = fit × n_series).

    For measure targets (sum/mean) the profile is computed on raw row values; for count
    targets it is computed on per-series events-per-period, which is the actual series
    being forecast."""
    if agg in ("nunique", "count"):
        aggfunc = "nunique" if agg == "nunique" else "count"
        counts = (df.groupby([*group_cols, pd.Grouper(key=ts_col, freq=alias)],
                             observed=True)[target].agg(aggfunc))
        levels = list(range(len(group_cols)))
        g = counts.groupby(level=levels, observed=True)
        per = g.agg(n="size", total="sum", mean="mean", std="std")
        # active periods vs the full grid → honest ADI for event series
        per["adi"] = grid_len / per["n"].replace(0, np.nan)
        per["cv2"] = (per["std"] / per["mean"]) ** 2
    else:
        g = df.groupby(group_cols, observed=True)[target]
        per = g.agg(n="size", total="sum", mean="mean", std="std",
                    nz_count=lambda s: int((s != 0).sum()))
        nz = df.loc[df[target] != 0].groupby(group_cols, observed=True)[target]
        per["nz_mean"] = nz.mean()
        per["nz_std"] = nz.std()
        per["adi"] = per["n"] / per["nz_count"].replace(0, np.nan)
        per["cv2"] = (per["nz_std"] / per["nz_mean"]) ** 2

    def _cls(row):
        if not np.isfinite(row["adi"]):
            return "no_demand"
        cv2 = row["cv2"] if np.isfinite(row["cv2"]) else 0.0
        if row["adi"] < 1.32:
            return "smooth" if cv2 < 0.49 else "erratic"
        return "intermittent" if cv2 < 0.49 else "lumpy"

    classes = per.apply(_cls, axis=1)
    mix = {k: float(v) for k, v in classes.value_counts(normalize=True).round(3).items()}

    n_series = len(per)
    volumes = per["total"].sort_values(ascending=False)
    top_k = min(10, n_series)
    total_vol = float(volumes.sum()) or 1.0
    return {
        "n_series": n_series,
        "series_length": {
            "min": int(per["n"].min()), "median": int(per["n"].median()),
            "max": int(per["n"].max()),
        },
        "length_unit": "periods" if agg in ("nunique", "count") else "rows",
        "intermittency_mix": mix,
        "top_series_volume_share": _f(float(volumes.head(top_k).sum()) / total_vol, 3),
        "top_series": [
            {"key": " / ".join(str(v) for v in (idx if isinstance(idx, tuple) else (idx,))),
             "total": _f(vol, 1)}
            for idx, vol in volumes.head(top_k).items()
        ],
    }


# ── Decision payload ─────────────────────────────────────────────────────────

def _pacf_significant_lags(autocorr: dict, cap: int = 5) -> list[int]:
    """1-indexed lags whose |PACF| exceeds the confidence bound (first `cap` kept)."""
    pacf, bound = autocorr.get("pacf") or [], autocorr.get("conf_bound")
    if not pacf or bound is None:
        return []
    return [i + 1 for i, v in enumerate(pacf)
            if v is not None and abs(v) > bound][:cap]


def _build_payload(stats: dict, selections: dict, exog: dict,
                   fill_report: dict, panel: dict | None) -> dict:
    """The compact JSON handed to Stage 5 model selection. Statistics only."""
    seas = stats["seasonality"]
    secondary = [c["period"] for c in seas["candidates"]
                 if c["period"] != seas["primary_period"] and (c["seasonal_strength"] or 0) > 0.3]
    st = stats["stationarity"]
    payload = {
        "scope": selections["scope"],
        "target": selections["target_col"],
        "agg": selections["agg"],
        "frequency": selections["data_frequency"],
        "forecast_frequency": selections["forecast_frequency"],
        "horizon": selections["horizon"],
        "series_length": stats["length"],
        "missing_pct": fill_report["missing_pct"],
        "zero_pct": fill_report["zero_pct"],
        "seasonality_strength": stats["seasonal_strength"],
        "seasonality_period": seas["primary_period"],
        "secondary_seasonality_periods": secondary,
        "trend_strength": stats["trend_strength"],
        "trend_direction": stats["trend"]["direction"],
        "stationarity": st["stationary"],
        "adf_p": st["adf_p"],
        "kpss_p": st["kpss_p"],
        "ndiffs": st["ndiffs"],
        "nsdiffs": st["nsdiffs"],
        "intermittency_class": stats["intermittency"]["class"],
        "adi": stats["intermittency"]["adi"],
        "cv2": stats["intermittency"]["cv2"],
        "forecastability": stats["forecastability"]["score"],
        "spectral_entropy": stats["forecastability"]["spectral_entropy"],
        "sample_entropy": stats["forecastability"]["sample_entropy"],
        "snr": stats["snr"],
        "cv": stats["cv"],
        "significant_acf_lags": stats["autocorrelation"]["significant_lags"][:15],
        # AR-order hint (2026-07-17): a SHORT significant-PACF list = low-order AR
        # structure — the actual evidence for ARIMA-family models; a long ACF list
        # alone is just seasonality. Rules and LLM both read it.
        "pacf_significant_lags": _pacf_significant_lags(stats["autocorrelation"]),
        "heteroskedastic": stats["heteroskedasticity"]["heteroskedastic"],
        "transform_recommendation": stats["heteroskedasticity"]["transform_recommendation"],
        "residual_ljung_box_p": stats["residual_ljung_box_p"],
        "structural_break_recent": stats["structural_breaks"]["recent_break"],
        "structural_breaks": {
            "n": len(stats["structural_breaks"]["break_dates"]),
            "recent_dates": stats["structural_breaks"]["break_dates"][-2:],
        },
        "slope_pct_per_period": stats["trend"]["slope_pct_per_period"],
        "recent_slope_pct_per_period": stats["trend"]["recent_slope_pct_per_period"],
        "approx_entropy": stats["forecastability"]["approx_entropy"],
        "dfa_alpha": stats["forecastability"]["dfa_alpha"],
        "has_exogenous": exog["has_exogenous"],
        "exog_top": exog["cross_correlation"][:5],
        "vif_max": max((r["vif"] for r in exog["vif"] if r["vif"] is not None), default=None),
        "panel": panel,
    }

    warnings_list = []
    period = seas["primary_period"] or 1
    if stats["length"] < 2 * period + 1:
        warnings_list.append("short_series: fewer than 2 seasonal cycles of history")
    if (fill_report["missing_pct"] or 0) > 20:
        warnings_list.append("high_missing: >20% of the time grid had no data")
    if payload["horizon"] > stats["length"] / 3:
        warnings_list.append("long_horizon: horizon exceeds a third of the history")
    if payload["structural_break_recent"]:
        warnings_list.append("recent_structural_break: level/behavior shifted near the end")
    if (payload["forecastability"] or 1) < 0.3:
        warnings_list.append("low_forecastability: series is close to noise")
    payload["warnings"] = warnings_list
    return payload


# ── Stage 4 entry point ──────────────────────────────────────────────────────

def run(run_id: str) -> dict:
    """Stage 4 entry point. Reads cleaned parquet + confirmed intent; zero DB, zero LLM."""
    cfg = _load_config()
    run_dir, parquet = _paths(run_id)
    selections = _resolve_selections(run_dir)

    ts_col = selections["timestamp_col"]
    target = selections["target_col"]
    group_cols = selections["group_cols"] if selections["scope"] == "per_series" else []
    exog_cols = [c for c in selections["exog_cols"] if c != target]

    use_cols = list(dict.fromkeys([ts_col, target, *group_cols, *exog_cols]))
    df = pd.read_parquet(parquet, columns=[c for c in use_cols if c])
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.loc[df[ts_col].notna() & df[target].notna()]
    if df.empty:
        raise ValueError(f"No usable rows for target '{target}'.")
    # User-confirmed training window (default = full data, report always written).
    df, train_filter = apply_train_filter(df, ts_col, selections)

    # sum/mean of a non-numeric column is undefined — the intent for such targets is
    # counting events. Forced here as a backstop; the intent agent normally sets it.
    if selections["agg"] in ("sum", "mean") and not pd.api.types.is_numeric_dtype(df[target]):
        selections["agg"] = "nunique"
        selections["agg_forced"] = True

    freq_label = selections["forecast_frequency"]

    # The aggregate/overall series is always analyzed — in per-series scope it still
    # anchors the payload (global models train on it conceptually) and the panel summary
    # rides alongside.
    y, exog, fill_report = _build_series(df, ts_col, target, exog_cols,
                                         selections["agg"], freq_label)
    if len(y) < 10:
        raise ValueError(
            f"Only {len(y)} points at {freq_label} granularity — too short to analyze.")
    progress.stage_update(run_id, "forecast_eda",
                          f"series built ({len(y)} points) — running statistics", 1, 5)
    stats = _series_stats(y, freq_label, cfg)
    period = stats["seasonality"]["primary_period"] or 1
    progress.stage_update(run_id, "forecast_eda",
                          "seasonality/stationarity done — exogenous analysis", 2, 5)
    exog_stats = _exog_analysis(y, exog, period, cfg)

    panel = None
    per_series_deep = []
    if selections["scope"] == "per_series" and group_cols:
        progress.stage_update(run_id, "forecast_eda", "per-series panel summary", 3, 5)
        alias = _pandas_freq(freq_label, df[ts_col])
        panel = _panel_summary(df, ts_col, target, group_cols,
                               selections["agg"], alias, grid_len=len(y))
        # Deep-dive the top-K series by volume so the human (and Stage 5) can see whether
        # individual series behave like the aggregate or need intermittent-class models.
        for entry in panel["top_series"][:cfg.get("top_series_deep_dive", 3)]:
            key_vals = entry["key"].split(" / ")
            mask = pd.Series(True, index=df.index)
            for col, val in zip(group_cols, key_vals):
                mask &= df[col].astype(str) == val
            sub = df.loc[mask]
            if len(sub) < 10:
                continue
            ys, _, fr = _build_series(sub, ts_col, target, [], selections["agg"], freq_label)
            if len(ys) < 10:
                continue
            s = _series_stats(ys, freq_label, cfg)
            per_series_deep.append({"key": entry["key"], "fill_report": fr, "stats": s})

    progress.stage_update(run_id, "forecast_eda", "building decision payload", 4, 5)
    payload = _build_payload(stats, selections, exog_stats, fill_report, panel)
    if train_filter["active"]:
        payload["train_filter"] = {k: train_filter[k] for k in
                                   ("train_start", "train_end", "exclude_ranges",
                                    "rows_before", "rows_after")}
        if len(y) < 2 * max(period, 1):
            payload["warnings"].append(
                "train_filter_short: the training window leaves fewer than 2 seasonal "
                "cycles — consider widening it")

    full = {
        "run_id": run_id,
        "selections": selections,
        "train_filter": train_filter,
        "fill_report": fill_report,
        "aggregate": stats,
        "exogenous": exog_stats,
        "panel": panel,
        "per_series_deep_dive": per_series_deep,
    }
    (run_dir / "forecasting_eda_full.json").write_text(json.dumps(full, indent=2, default=str))
    (run_dir / "model_selection_payload.json").write_text(json.dumps(payload, indent=2, default=str))
    # Fresh Stage 4 evidence invalidates every downstream decision on disk — Stage 5 model
    # selection AND Stage 6/7 features+training (covers the frequency/horizon-only re-run
    # path, where /clean never fires).
    for name in ("model_selection.json", "feature_report.json", "training_report.json"):
        (run_dir / name).unlink(missing_ok=True)
    features_dir = ROOT / "data" / "features"
    for p in (features_dir / f"{run_id}_model_frame.parquet",
              features_dir / f"{run_id}_features.parquet"):
        p.unlink(missing_ok=True)
    import shutil
    shutil.rmtree(ROOT / "models" / run_id, ignore_errors=True)

    return {
        "run_id": run_id,
        "status": "completed",
        "selections": selections,
        "payload": payload,
        "full": full,
    }
