from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

try:
    import holidays  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    holidays = None


@dataclass
class CalendarFeatureConfig:
    feature_set: str = "basic"
    region: str | None = None
    subdiv: str | None = None
    festival_windows: dict[str, list[tuple[int, int, int, int]]] = field(default_factory=dict)


def build_calendar_features(time_index: pd.Series, config: CalendarFeatureConfig | None = None) -> pd.DataFrame:
    config = config or CalendarFeatureConfig()
    index = pd.to_datetime(time_index)
    features = pd.DataFrame(index=index.index)
    features["month"] = index.dt.month.astype(float)
    features["dayofweek"] = index.dt.dayofweek.astype(float)
    features["dayofmonth"] = index.dt.day.astype(float)
    features["quarter"] = index.dt.quarter.astype(float)
    features["weekofyear"] = index.dt.isocalendar().week.astype(float)
    features["is_weekend"] = index.dt.dayofweek.isin([5, 6]).astype(float)

    if config.feature_set in {"fiscal", "all"}:
        features["is_fiscal_year_start"] = ((index.dt.month == 4) & (index.dt.day <= 7)).astype(float)
        features["is_fiscal_year_end"] = ((index.dt.month == 3) & (index.dt.day >= 25)).astype(float)

    if config.feature_set in {"holiday", "all"}:
        features = pd.concat([features, _holiday_flags(index, config)], axis=1)

    if config.feature_set in {"seasonal", "all"}:
        features = pd.concat([features, _festival_flags(index, config)], axis=1)

    return features.fillna(0.0)


def build_future_calendar_features(last_timestamp: Any, frequency: str | None, step: int, config: CalendarFeatureConfig | None = None) -> dict[str, float]:
    config = config or CalendarFeatureConfig()
    timestamp = pd.to_datetime(last_timestamp)
    future = timestamp + _offset_for_frequency(frequency, step)
    features = build_calendar_features(pd.Series([future]), config).iloc[0].to_dict()
    return {key: float(value) for key, value in features.items()}


def _holiday_flags(index: pd.DatetimeIndex, config: CalendarFeatureConfig) -> pd.DataFrame:
    if holidays is None or not config.region:
        return pd.DataFrame(index=index.index)
    try:
        holiday_calendar = holidays.country_holidays(config.region, subdiv=config.subdiv)
        flags = pd.Series([1.0 if timestamp.date() in holiday_calendar else 0.0 for timestamp in index], index=index.index)
        return pd.DataFrame({"is_holiday": flags})
    except Exception:
        return pd.DataFrame(index=index.index)


def _festival_flags(index: pd.DatetimeIndex, config: CalendarFeatureConfig) -> pd.DataFrame:
    if not config.festival_windows:
        return pd.DataFrame(index=index.index)
    frame = pd.DataFrame(index=index.index)
    for name, windows in config.festival_windows.items():
        values = []
        for timestamp in index:
            values.append(1.0 if _within_any_window(timestamp, windows) else 0.0)
        frame[f"is_{name}"] = values
    return frame


def _within_any_window(timestamp: pd.Timestamp, windows: list[tuple[int, int, int, int]]) -> bool:
    for start_month, start_day, end_month, end_day in windows:
        start = pd.Timestamp(timestamp.year, start_month, start_day)
        end = pd.Timestamp(timestamp.year, end_month, end_day)
        if start <= timestamp <= end:
            return True
    return False


def _offset_for_frequency(frequency: str | None, step: int) -> pd.DateOffset | pd.Timedelta:
    if not frequency:
        return pd.Timedelta(days=step)
    freq = str(frequency).upper()
    if freq.startswith("H"):
        return pd.Timedelta(hours=step)
    if freq.startswith("D"):
        return pd.Timedelta(days=step)
    if freq.startswith("W"):
        return pd.Timedelta(weeks=step)
    if freq.startswith("M"):
        return pd.DateOffset(months=step)
    if freq.startswith("Q"):
        return pd.DateOffset(months=3 * step)
    if freq.startswith("Y") or freq.startswith("A"):
        return pd.DateOffset(years=step)
    return pd.Timedelta(days=step)