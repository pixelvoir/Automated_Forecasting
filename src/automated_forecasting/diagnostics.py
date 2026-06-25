from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf, kpss, pacf
from statsmodels.tools.sm_exceptions import InterpolationWarning


@dataclass
class SeriesQuality:
    missing_rate: float
    outlier_rate: float
    outlier_low: float | None
    outlier_high: float | None
    imputation_strategy: str
    imputed_points: int = 0


@dataclass
class SeriesDiagnostics:
    series_value: Any
    row_count: int
    missing_rate: float
    outlier_rate: float
    trend_strength: float
    seasonal_strength: float
    adf_pvalue: float | None
    kpss_pvalue: float | None
    acf_peak_lag: int | None
    pacf_peak_lag: int | None
    order_bounds: dict[str, int]
    quality: SeriesQuality
    kpss_note: str | None = None


@dataclass
class RoutingDiagnostics:
    time_column: str
    target_column: str
    series_column: str | None
    frequency: str | None
    seasonal_period: int
    row_count: int
    series_count: int
    has_exogenous: bool
    is_regular: bool
    parse_success_rate: float | None = None
    column_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_series: list[SeriesDiagnostics] = field(default_factory=list)
    diagnostics_summary: dict[str, Any] = field(default_factory=dict)


def build_quality_summary(values: pd.Series, seasonal_period: int) -> SeriesQuality:
    numeric = pd.to_numeric(values, errors="coerce")
    missing_rate = float(numeric.isna().mean())
    clean = numeric.dropna()
    if clean.empty:
        return SeriesQuality(missing_rate, 0.0, None, None, "ffill_bfill")

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0.0:
        low = high = None
        outlier_rate = 0.0
    else:
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        outlier_mask = (clean < low) | (clean > high)
        outlier_rate = float(outlier_mask.mean())

    strategy = "seasonal_interpolate" if seasonal_period > 1 else "time_interpolate"
    if missing_rate > 0.25:
        strategy = "ffill_bfill"

    return SeriesQuality(missing_rate, outlier_rate, low, high, strategy)


def apply_imputation(frame: pd.DataFrame, time_column: str, target_column: str, seasonal_period: int, strategy: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = frame.copy()
    result[time_column] = pd.to_datetime(result[time_column], errors="coerce")
    result = result.dropna(subset=[time_column]).sort_values(time_column).reset_index(drop=True)
    quality = build_quality_summary(result[target_column], seasonal_period)
    chosen_strategy = strategy or quality.imputation_strategy

    numeric = pd.to_numeric(result[target_column], errors="coerce")
    original_missing = int(numeric.isna().sum())
    clipped = numeric.copy()
    if quality.outlier_low is not None and quality.outlier_high is not None:
        clipped = clipped.clip(quality.outlier_low, quality.outlier_high)

    if chosen_strategy == "seasonal_interpolate" and seasonal_period > 1 and len(clipped) > seasonal_period:
        filled = clipped.interpolate(method="linear", limit_direction="both")
        filled = filled.interpolate(method="nearest", limit_direction="both")
        filled = filled.ffill().bfill()
    elif chosen_strategy == "time_interpolate":
        filled = clipped.interpolate(method="linear", limit_direction="both").ffill().bfill()
    else:
        filled = clipped.ffill().bfill()

    result[target_column] = filled.astype(float)
    imputed_points = int(pd.isna(numeric).sum())
    summary = {
        "strategy": chosen_strategy,
        "missing_before": original_missing,
        "imputed_points": imputed_points,
        "outlier_low": quality.outlier_low,
        "outlier_high": quality.outlier_high,
    }
    return result, summary


def compute_routing_diagnostics(
    frame: pd.DataFrame,
    time_column: str,
    target_column: str,
    series_column: str | None,
    frequency: str | None,
    max_series_to_inspect: int = 50,
) -> RoutingDiagnostics:
    ordered = frame.copy()
    ordered[time_column] = pd.to_datetime(ordered[time_column], errors="coerce")
    ordered = ordered.dropna(subset=[time_column, target_column])
    ordered = ordered.sort_values([series_column, time_column] if series_column else [time_column])

    seasonal_period = _seasonal_period_from_frequency(frequency)
    groups = list(ordered.groupby(series_column, dropna=False)) if series_column else [(None, ordered)]
    inspected = groups[:max_series_to_inspect]

    per_series: list[SeriesDiagnostics] = []
    for series_value, group in inspected:
        per_series.append(_diagnose_single_series(group, target_column, time_column, series_value, seasonal_period))

    has_exogenous = _has_exogenous(ordered, time_column, target_column, series_column)
    return RoutingDiagnostics(
        time_column=time_column,
        target_column=target_column,
        series_column=series_column,
        frequency=frequency,
        seasonal_period=seasonal_period,
        row_count=len(ordered),
        series_count=len(groups),
        has_exogenous=has_exogenous,
        is_regular=_is_regular_series(ordered[time_column]),
        parse_success_rate=float(ordered[time_column].notna().mean()),
        column_stats={},
        per_series=per_series,
        diagnostics_summary=_summarize(per_series),
    )


def _diagnose_single_series(group: pd.DataFrame, target_column: str, time_column: str, series_value: Any, seasonal_period: int) -> SeriesDiagnostics:
    target = pd.to_numeric(group[target_column], errors="coerce").dropna().astype(float)
    quality = build_quality_summary(group[target_column], seasonal_period)
    if len(target) < 8:
        return SeriesDiagnostics(
            series_value=series_value,
            row_count=len(group),
            missing_rate=quality.missing_rate,
            outlier_rate=quality.outlier_rate,
            trend_strength=0.0,
            seasonal_strength=0.0,
            adf_pvalue=None,
            kpss_pvalue=None,
            acf_peak_lag=None,
            pacf_peak_lag=None,
            order_bounds={"p": 0, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0},
            quality=quality,
            kpss_note=None,
        )

    trend_strength, seasonal_strength = _stl_strength(target, seasonal_period)
    adf_pvalue = _safe_adf_pvalue(target)
    kpss_pvalue, kpss_note = _safe_kpss_pvalue(target)
    acf_peak_lag = _peak_lag(acf(target, nlags=min(24, len(target) - 1), fft=True))
    with warnings.catch_warnings():
        # PACF can hit a divide-by-zero log warning on perfectly linear small samples; peak-lag routing still remains usable.
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log")
        # PACF requires nlags below half the sample; cap it so diagnostics do not fail short sanity runs.
        pacf_peak_lag = _peak_lag(pacf(target, nlags=min(24, max(1, len(target) // 2 - 1)), method="yw"))

    d = 1 if (adf_pvalue is not None and adf_pvalue > 0.05) else 0
    D = 1 if seasonal_period > 1 and seasonal_strength >= 0.4 else 0
    p = min(2, acf_peak_lag or 0)
    q = min(2, (acf_peak_lag or 0) // 2)
    P = min(2, pacf_peak_lag or 0)
    Q = min(2, (pacf_peak_lag or 0) // 2)

    return SeriesDiagnostics(
        series_value=series_value,
        row_count=len(group),
        missing_rate=quality.missing_rate,
        outlier_rate=quality.outlier_rate,
        trend_strength=trend_strength,
        seasonal_strength=seasonal_strength,
        adf_pvalue=adf_pvalue,
        kpss_pvalue=kpss_pvalue,
        acf_peak_lag=acf_peak_lag,
        pacf_peak_lag=pacf_peak_lag,
        order_bounds={"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q},
        quality=quality,
        kpss_note=kpss_note,
    )


def _stl_strength(series: pd.Series, seasonal_period: int) -> tuple[float, float]:
    if seasonal_period < 2 or len(series) < seasonal_period * 2:
        return 0.0, 0.0
    try:
        fit = STL(series, period=seasonal_period, robust=True).fit()
        resid_scale = np.var(fit.resid) + 1e-9
        trend_strength = max(0.0, 1.0 - np.var(fit.resid + fit.seasonal) / (np.var(fit.trend + fit.resid) + 1e-9))
        seasonal_strength = max(0.0, 1.0 - np.var(fit.resid + fit.trend) / (np.var(fit.seasonal + fit.resid) + 1e-9))
        return float(np.clip(trend_strength, 0.0, 1.0)), float(np.clip(seasonal_strength, 0.0, 1.0))
    except Exception:
        return 0.0, 0.0


def _safe_adf_pvalue(series: pd.Series) -> float | None:
    try:
        with warnings.catch_warnings():
            # ADF's internal OLS can warn on perfectly collinear tiny samples; a failed statistic is handled by the caller.
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log")
            return float(adfuller(series, autolag="AIC")[1])
    except Exception:
        return None


def _safe_kpss_pvalue(series: pd.Series) -> tuple[float | None, str | None]:
    try:
        with warnings.catch_warnings(record=True) as caught:
            # Statsmodels emits InterpolationWarning for out-of-table KPSS p-values; clamp and record a note instead of printing noise.
            warnings.simplefilter("always", InterpolationWarning)
            pvalue = float(kpss(series, regression="c", nlags="auto")[1])
        if any(issubclass(item.category, InterpolationWarning) for item in caught):
            return float(np.clip(pvalue, 0.01, 0.1)), "kpss_pvalue_clamped_out_of_table"
        return pvalue, None
    except Exception:
        return None, None


def _peak_lag(values: np.ndarray) -> int | None:
    if len(values) < 3:
        return None
    threshold = 2.0 / np.sqrt(len(values))
    peaks = [index for index, value in enumerate(values[1:], start=1) if abs(value) >= threshold]
    return int(peaks[0]) if peaks else None


def _has_exogenous(frame: pd.DataFrame, time_column: str, target_column: str, series_column: str | None) -> bool:
    excluded = {time_column, target_column}
    if series_column:
        excluded.add(series_column)
    for column in frame.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            return True
    return False


def _seasonal_period_from_frequency(frequency: str | None) -> int:
    if not frequency:
        return 1
    freq = str(frequency).upper()
    if freq.startswith("H"):
        return 24
    if freq.startswith("D"):
        return 7
    if freq.startswith("W"):
        return 52
    if freq.startswith("M"):
        return 12
    if freq.startswith("Q"):
        return 4
    return 1


def _is_regular_series(time_index: pd.Series) -> bool:
    index = pd.to_datetime(time_index).dropna().sort_values()
    if len(index) < 3:
        return True
    deltas = index.diff().dropna()
    return deltas.nunique() <= 2


def _summarize(per_series: list[SeriesDiagnostics]) -> dict[str, Any]:
    if not per_series:
        return {}
    notes = sorted({item.kpss_note for item in per_series if item.kpss_note})
    return {
        "series_inspected": len(per_series),
        "avg_missing_rate": float(np.mean([item.missing_rate for item in per_series])),
        "avg_outlier_rate": float(np.mean([item.outlier_rate for item in per_series])),
        "avg_seasonal_strength": float(np.mean([item.seasonal_strength for item in per_series])),
        "stationary_share": float(np.mean([(item.adf_pvalue or 1.0) <= 0.05 for item in per_series])),
        "diagnostic_notes": notes,
    }
