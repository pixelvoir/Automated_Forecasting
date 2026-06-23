from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None


PREFERRED_TARGET_NAMES = ("target", "y", "value", "sales", "demand", "load", "price", "volume")
PREFERRED_TIME_NAMES = ("date", "datetime", "timestamp", "time", "ds")
PREFERRED_SERIES_NAMES = ("series", "series_id", "id", "store", "item", "sku", "entity", "group")


@dataclass
class ForecastRequest:
    csv_path: Path
    horizon: int
    output_path: Path
    time_column: str | None = None
    target_column: str | None = None
    series_column: str | None = None
    frequency: str | None = None
    interval_level: float = 0.9


@dataclass
class ForecastResult:
    summary: str
    selection_explanation: str
    output_path: Path
    selected_model: str
    rows: int


@dataclass
class DatasetProfile:
    time_column: str
    target_column: str
    series_column: str | None
    frequency: str | None
    seasonal_period: int
    row_count: int
    series_count: int
    has_exogenous: bool
    is_regular: bool


@dataclass
class CandidateScore:
    name: str
    score: float
    metric: float
    model: Any


class ForecastingAgent:
    """Local-only forecasting agent driven by deterministic heuristics."""

    def run(self, request: ForecastRequest) -> ForecastResult:
        frame = pd.read_csv(request.csv_path)
        profile = self._profile_dataset(frame, request)

        if profile.series_column:
            forecast_frame, selected_model, selection_explanation = self._forecast_panel(frame, request, profile)
        else:
            forecast_frame, selected_model, selection_explanation = self._forecast_single_series(frame, request, profile)

        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        forecast_frame.to_csv(request.output_path, index=False)

        return ForecastResult(
            summary=(
                f"Forecast completed with {selected_model}; "
                f"rows={len(forecast_frame)}; output={request.output_path}"
            ),
            selection_explanation=selection_explanation,
            output_path=request.output_path,
            selected_model=selected_model,
            rows=len(forecast_frame),
        )

    def _profile_dataset(self, frame: pd.DataFrame, request: ForecastRequest) -> DatasetProfile:
        time_column = request.time_column or self._infer_time_column(frame)
        target_column = request.target_column or self._infer_target_column(frame, time_column)
        series_column = request.series_column or self._infer_series_column(frame, time_column, target_column)

        frame = frame.copy()
        frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
        frame = frame.dropna(subset=[time_column, target_column])
        frame = frame.sort_values([series_column, time_column] if series_column else [time_column])

        frequencies = []
        regular = True
        if series_column:
            groups = frame.groupby(series_column, dropna=False)
        else:
            groups = [(None, frame)]

        for _, group in groups:
            inferred = self._infer_frequency_from_series(group[time_column])
            if inferred is not None:
                frequencies.append(inferred)
            if not self._is_regular_series(group[time_column]):
                regular = False

        frequency = request.frequency or (self._dominant_frequency(frequencies) if frequencies else None)
        seasonal_period = self._seasonal_period_from_frequency(frequency, frame[time_column])
        exogenous_cols = self._exogenous_columns(frame, time_column, target_column, series_column)

        return DatasetProfile(
            time_column=time_column,
            target_column=target_column,
            series_column=series_column,
            frequency=frequency,
            seasonal_period=seasonal_period,
            row_count=len(frame),
            series_count=frame[series_column].nunique(dropna=False) if series_column else 1,
            has_exogenous=bool(exogenous_cols),
            is_regular=regular,
        )

    def _forecast_single_series(
        self,
        frame: pd.DataFrame,
        request: ForecastRequest,
        profile: DatasetProfile,
    ) -> tuple[pd.DataFrame, str, str]:
        frame = self._prepare_series_frame(frame, profile.time_column, profile.target_column)
        candidates = self._select_candidates(frame, profile)
        scored = self._score_candidates(frame, profile, candidates, request.horizon)
        finite_scores = [item for item in scored if np.isfinite(item.metric)]
        best = min(finite_scores, key=lambda item: item.metric) if finite_scores else CandidateScore("naive", 0.0, 0.0, None)
        forecast = self._fit_and_forecast(frame, profile, best.name, request.horizon, request.interval_level)
        explanation = self._build_selection_explanation(profile, scored, best)
        return forecast, best.name, explanation

    def _forecast_panel(
        self,
        frame: pd.DataFrame,
        request: ForecastRequest,
        profile: DatasetProfile,
    ) -> tuple[pd.DataFrame, str, str]:
        outputs = []
        selected_models = []
        explanations = []
        for series_value, group in frame.groupby(profile.series_column, dropna=False):
            group_profile = DatasetProfile(
                time_column=profile.time_column,
                target_column=profile.target_column,
                series_column=None,
                frequency=profile.frequency,
                seasonal_period=profile.seasonal_period,
                row_count=len(group),
                series_count=1,
                has_exogenous=False,
                is_regular=profile.is_regular,
            )
            group = self._prepare_series_frame(group, profile.time_column, profile.target_column)
            candidates = self._select_candidates(group, group_profile)
            scored = self._score_candidates(group, group_profile, candidates, request.horizon)
            finite_scores = [item for item in scored if np.isfinite(item.metric)]
            best = min(finite_scores, key=lambda item: item.metric) if finite_scores else CandidateScore("naive", 0.0, 0.0, None)
            selected_models.append(best.name)
            explanations.append(f"Series {series_value}: {self._build_selection_explanation(group_profile, scored, best)}")
            forecast = self._fit_and_forecast(
                group,
                group_profile,
                best.name,
                request.horizon,
                request.interval_level,
                series_value=series_value,
                series_column=profile.series_column,
            )
            outputs.append(forecast)

        combined = pd.concat(outputs, ignore_index=True)
        return combined, "+".join(sorted(set(selected_models))), "\n".join(explanations)

    def _select_candidates(self, frame: pd.DataFrame, profile: DatasetProfile) -> list[str]:
        candidates = ["naive"]
        if profile.seasonal_period > 1 and len(frame) >= max(2 * profile.seasonal_period, 12):
            candidates.append("seasonal_naive")
        if len(frame) >= 12:
            candidates.append("ridge")
        if profile.seasonal_period > 1 and len(frame) >= max(3 * profile.seasonal_period, 20):
            candidates.append("ets")
        if len(frame) >= 24:
            candidates.append("sarimax")
        if len(frame) >= 30:
            candidates.append("boosting")
        if self._should_consider_tree_boosters(profile, len(frame)):
            candidates.extend(self._available_booster_candidates())
        return list(dict.fromkeys(candidates))

    def _score_candidates(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        candidates: Iterable[str],
        horizon: int,
    ) -> list[CandidateScore]:
        scores: list[CandidateScore] = []
        for candidate in candidates:
            try:
                metric = self._backtest_metric(frame, profile, candidate, horizon)
                score = metric + self._complexity_penalty(candidate) - self._preference_bonus(profile, candidate)
            except Exception:
                metric = float("inf")
                score = float("inf")
            scores.append(CandidateScore(candidate, score, metric, model=None))
        return scores

    def _build_selection_explanation(self, profile: DatasetProfile, scores: list[CandidateScore], best: CandidateScore) -> str:
        candidate_lines = []
        ordered_scores = sorted(scores, key=lambda entry: entry.score)
        for item in ordered_scores:
            metric_text = "inf" if not np.isfinite(item.metric) else f"{item.metric:.4f}"
            total_text = "inf" if not np.isfinite(item.score) else f"{item.score:.4f}"
            candidate_lines.append(f"- {item.name}: validation_mae={metric_text}, adjusted_score={total_text}")

        data_signals = [
            f"rows={profile.row_count}",
            f"series_count={profile.series_count}",
            f"frequency={profile.frequency or 'unknown'}",
            f"seasonal_period={profile.seasonal_period}",
            f"regular_index={'yes' if profile.is_regular else 'no'}",
            f"exogenous_features={'yes' if profile.has_exogenous else 'no'}",
        ]

        if profile.seasonal_period > 1:
            routing_reason = "Seasonality is present, so seasonal baseline and state-space candidates were considered."
        elif profile.has_exogenous:
            routing_reason = "Exogenous features are available, so lightweight feature-based models were prioritized."
        elif profile.row_count < 24:
            routing_reason = "The history is short, so simple baselines were kept in the shortlist to avoid overfitting."
        else:
            routing_reason = "The series is long enough for a small shortlist of baseline, statistical, and lightweight ML models."

        booster_reason = "Boosted-tree models were considered because the dataset has enough rows and useful exogenous or related-series structure."
        if best.name in {"lightgbm", "xgboost", "catboost", "boosting"}:
            booster_reason = "A boosted-tree model won because it achieved the best adjusted score on time-aware validation after a small complexity penalty."

        runner_up = ordered_scores[1] if len(ordered_scores) > 1 else None
        if runner_up is not None and np.isfinite(best.score) and np.isfinite(runner_up.score):
            margin_text = f"The winner beat the next best candidate by {runner_up.score - best.score:.4f} adjusted-score points."
        else:
            margin_text = "The winning model had the lowest adjusted score among the finite candidates."

        return (
            f"Data signals: {', '.join(data_signals)}. "
            f"Routing rule: {routing_reason} "
            f"Selection note: {booster_reason} {margin_text} "
            f"Chosen model: {best.name} because it had the lowest adjusted score after time-aware validation. "
            f"Candidates:\n" + "\n".join(candidate_lines)
        )

    def _backtest_metric(self, frame: pd.DataFrame, profile: DatasetProfile, candidate: str, horizon: int) -> float:
        series = frame[profile.target_column].astype(float).to_numpy()
        splits = self._rolling_splits(len(series), horizon)
        metrics: list[float] = []
        for train_end, test_end in splits:
            train = frame.iloc[:train_end].copy()
            test = frame.iloc[train_end:test_end].copy()
            if len(test) == 0:
                continue
            preds = self._predict_on_holdout(train, test, profile, candidate, horizon)
            if len(preds) != len(test):
                continue
            metrics.append(mean_absolute_error(test[profile.target_column].astype(float), preds))
        if metrics:
            return float(np.mean(metrics))
        baseline = np.abs(np.diff(series)).mean() if len(series) > 1 else float(np.abs(series).mean())
        return float(baseline)

    def _predict_on_holdout(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        profile: DatasetProfile,
        candidate: str,
        horizon: int,
    ) -> np.ndarray:
        if candidate == "naive":
            return np.repeat(train[profile.target_column].iloc[-1], len(test))
        if candidate == "seasonal_naive" and profile.seasonal_period > 1 and len(train) >= profile.seasonal_period:
            pattern = train[profile.target_column].iloc[-profile.seasonal_period :].to_numpy()
            return np.resize(pattern, len(test))
        if candidate == "ridge":
            return self._recursive_forecast_with_regressor(train, profile, len(test), Ridge(alpha=1.0))
        if candidate == "boosting":
            return self._recursive_forecast_with_regressor(
                train,
                profile,
                len(test),
                HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=200, random_state=42),
            )
        if candidate == "lightgbm" and LGBMRegressor is not None:
            return self._recursive_forecast_with_regressor(
                train,
                profile,
                len(test),
                LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=-1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    verbose=-1,
                    random_state=42,
                ),
            )
        if candidate == "xgboost" and XGBRegressor is not None:
            return self._recursive_forecast_with_regressor(
                train,
                profile,
                len(test),
                XGBRegressor(
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    verbosity=0,
                    random_state=42,
                ),
            )
        if candidate == "catboost" and CatBoostRegressor is not None:
            return self._recursive_forecast_with_regressor(
                train,
                profile,
                len(test),
                CatBoostRegressor(
                    iterations=250,
                    learning_rate=0.05,
                    depth=6,
                    loss_function="RMSE",
                    verbose=False,
                    logging_level="Silent",
                    random_seed=42,
                ),
            )
        if candidate == "ets":
            model = ExponentialSmoothing(
                train[profile.target_column].astype(float),
                trend="add" if len(train) >= 8 else None,
                seasonal="add" if profile.seasonal_period > 1 else None,
                seasonal_periods=profile.seasonal_period if profile.seasonal_period > 1 else None,
                initialization_method="estimated",
            ).fit(optimized=True)
            return np.asarray(model.forecast(len(test)))
        if candidate == "sarimax":
            return self._fit_sarimax(train, profile, len(test))
        return np.repeat(train[profile.target_column].iloc[-1], len(test))

    def _fit_and_forecast(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        candidate: str,
        horizon: int,
        interval_level: float,
        series_value: Any | None = None,
        series_column: str | None = None,
    ) -> pd.DataFrame:
        try:
            forecast_values, lower, upper = self._fit_full_model(frame, profile, candidate, horizon, interval_level)
        except Exception:
            forecast_values = np.repeat(float(frame[profile.target_column].iloc[-1]), horizon)
            lower = None
            upper = None
        future_index = self._future_index(frame[profile.time_column], horizon, profile.frequency)
        output = pd.DataFrame(
            {
                "timestamp": future_index,
                "forecast": forecast_values,
                "model_name": candidate,
            }
        )
        if series_column is not None:
            output.insert(0, series_column, series_value)
        if lower is not None and upper is not None:
            output["lower_bound"] = lower
            output["upper_bound"] = upper
        return output

    def _fit_full_model(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        candidate: str,
        horizon: int,
        interval_level: float,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        target = frame[profile.target_column].astype(float).reset_index(drop=True)
        if candidate == "naive":
            value = float(target.iloc[-1])
            forecast = np.repeat(value, horizon)
            return forecast, None, None
        if candidate == "seasonal_naive" and profile.seasonal_period > 1 and len(target) >= profile.seasonal_period:
            pattern = target.iloc[-profile.seasonal_period :].to_numpy()
            forecast = np.resize(pattern, horizon)
            return forecast, None, None
        if candidate == "ridge":
            return self._recursive_forecast_with_regressor(frame, profile, horizon, Ridge(alpha=1.0), return_interval=True)
        if candidate == "boosting":
            return self._recursive_forecast_with_regressor(
                frame,
                profile,
                horizon,
                HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=250, random_state=42),
                return_interval=True,
            )
        if candidate == "lightgbm" and LGBMRegressor is not None:
            return self._recursive_forecast_with_regressor(
                frame,
                profile,
                horizon,
                LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=-1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    verbose=-1,
                    random_state=42,
                ),
                return_interval=True,
            )
        if candidate == "xgboost" and XGBRegressor is not None:
            return self._recursive_forecast_with_regressor(
                frame,
                profile,
                horizon,
                XGBRegressor(
                    n_estimators=350,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    verbosity=0,
                    random_state=42,
                ),
                return_interval=True,
            )
        if candidate == "catboost" and CatBoostRegressor is not None:
            return self._recursive_forecast_with_regressor(
                frame,
                profile,
                horizon,
                CatBoostRegressor(
                    iterations=350,
                    learning_rate=0.05,
                    depth=6,
                    loss_function="RMSE",
                    verbose=False,
                    logging_level="Silent",
                    random_seed=42,
                ),
                return_interval=True,
            )
        if candidate == "ets":
            model = ExponentialSmoothing(
                target,
                trend="add" if len(target) >= 8 else None,
                seasonal="add" if profile.seasonal_period > 1 else None,
                seasonal_periods=profile.seasonal_period if profile.seasonal_period > 1 else None,
                initialization_method="estimated",
            ).fit(optimized=True)
            forecast = np.asarray(model.forecast(horizon))
            residual_std = float(np.std(model.resid)) if len(model.resid) else 0.0
            delta = self._z_value(interval_level) * residual_std
            return forecast, forecast - delta, forecast + delta
        if candidate == "sarimax":
            forecast = self._fit_sarimax(frame, profile, horizon)
            residual_std = float(np.std(target.diff().dropna())) if len(target) > 1 else 0.0
            delta = self._z_value(interval_level) * residual_std
            return forecast, forecast - delta, forecast + delta
        forecast = np.repeat(float(target.iloc[-1]), horizon)
        return forecast, None, None

    def _fit_sarimax(self, frame: pd.DataFrame, profile: DatasetProfile, horizon: int) -> np.ndarray:
        target = frame[profile.target_column].astype(float)
        seasonal = profile.seasonal_period if profile.seasonal_period > 1 else 0
        seasonal_order = (1, 0, 0, seasonal) if seasonal else (0, 0, 0, 0)
        model = SARIMAX(
            target,
            order=(1, 1, 1),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return np.asarray(model.forecast(horizon))

    def _recursive_forecast_with_regressor(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        horizon: int,
        regressor: Any,
        return_interval: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None] | np.ndarray:
        features, target = self._build_supervised_features(frame, profile)
        if len(features) == 0:
            fallback = np.repeat(float(frame[profile.target_column].iloc[-1]), horizon)
            return (fallback, None, None) if return_interval else fallback
        regressor.fit(features, target)
        history = frame[profile.target_column].astype(float).tolist()
        future_rows = []
        for step in range(1, horizon + 1):
            row = self._feature_row_from_history(history, frame[profile.time_column].iloc[-1], profile, step)
            pred = float(regressor.predict(pd.DataFrame([row]))[0])
            history.append(pred)
            future_rows.append(pred)
        forecast = np.asarray(future_rows)
        if return_interval:
            residuals = target - regressor.predict(features)
            spread = self._z_value(0.9) * float(np.std(residuals)) if len(residuals) else 0.0
            return forecast, forecast - spread, forecast + spread
        return forecast

    def _build_supervised_features(self, frame: pd.DataFrame, profile: DatasetProfile, max_lag: int = 8) -> tuple[pd.DataFrame, pd.Series]:
        target = frame[profile.target_column].astype(float).reset_index(drop=True)
        data = pd.DataFrame({"y": target})
        lag_limit = min(max_lag, max(2, len(data) // 4))
        for lag in range(1, lag_limit + 1):
            data[f"lag_{lag}"] = data["y"].shift(lag)
        data["rolling_mean_3"] = data["y"].shift(1).rolling(3).mean()
        data["rolling_std_3"] = data["y"].shift(1).rolling(3).std()
        data["trend_index"] = np.arange(len(data))
        calendar = self._calendar_features(frame[profile.time_column])
        data = pd.concat([data, calendar], axis=1)
        data = data.dropna()
        if data.empty:
            return pd.DataFrame(), pd.Series(dtype=float)
        y = data.pop("y")
        return data, y

    def _feature_row_from_history(self, history: list[float], last_timestamp: Any, profile: DatasetProfile, step: int) -> dict[str, float]:
        row: dict[str, float] = {"trend_index": float(len(history) + step - 1)}
        for lag in range(1, min(8, len(history)) + 1):
            row[f"lag_{lag}"] = float(history[-lag])
        recent = np.asarray(history[-3:]) if len(history) >= 3 else np.asarray(history)
        row["rolling_mean_3"] = float(np.mean(recent))
        row["rolling_std_3"] = float(np.std(recent)) if len(recent) > 1 else 0.0
        row.update(self._future_calendar_features(last_timestamp, profile.frequency, step))
        return row

    def _calendar_features(self, time_index: pd.Series) -> pd.DataFrame:
        index = pd.to_datetime(time_index)
        return pd.DataFrame(
            {
                "month": index.dt.month.astype(float),
                "dayofweek": index.dt.dayofweek.astype(float),
                "dayofmonth": index.dt.day.astype(float),
                "quarter": index.dt.quarter.astype(float),
                "is_weekend": index.dt.dayofweek.isin([5, 6]).astype(float),
            }
        )

    def _future_calendar_features(self, last_timestamp: Any, frequency: str | None, step: int) -> dict[str, float]:
        timestamp = pd.to_datetime(last_timestamp)
        future = timestamp + self._offset_for_frequency(frequency, step)
        return {
            "month": float(future.month),
            "dayofweek": float(future.dayofweek),
            "dayofmonth": float(future.day),
            "quarter": float(future.quarter),
            "is_weekend": float(future.dayofweek in [5, 6]),
        }

    def _future_index(self, time_index: pd.Series, horizon: int, frequency: str | None) -> pd.Index:
        observed = pd.to_datetime(time_index).dropna().sort_values()
        if frequency:
            try:
                start = observed.iloc[-1]
                return pd.date_range(start=start, periods=horizon + 1, freq=frequency)[1:]
            except Exception:
                pass
        if len(observed) >= 2:
            delta = observed.diff().dropna().median()
            return pd.Index([observed.iloc[-1] + delta * (i + 1) for i in range(horizon)])
        return pd.Index([observed.iloc[-1] + pd.Timedelta(days=i + 1) for i in range(horizon)])

    def _prepare_series_frame(self, frame: pd.DataFrame, time_column: str, target_column: str) -> pd.DataFrame:
        frame = frame.copy()
        frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
        frame = frame.dropna(subset=[time_column, target_column]).sort_values(time_column)
        frame = frame.drop_duplicates(subset=[time_column], keep="last")
        return frame.reset_index(drop=True)

    def _infer_time_column(self, frame: pd.DataFrame) -> str:
        preferred = self._match_preferred_name(frame.columns, PREFERRED_TIME_NAMES)
        if preferred:
            return preferred
        scores = []
        for column in frame.columns:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            ratio = parsed.notna().mean()
            unique_ratio = parsed.nunique(dropna=True) / max(len(parsed), 1)
            scores.append((ratio * 2 + unique_ratio, column))
        best = max(scores, key=lambda item: item[0])
        if best[0] < 1.0:
            raise ValueError("Unable to infer a time column. Please pass --time-column.")
        return best[1]

    def _infer_target_column(self, frame: pd.DataFrame, time_column: str) -> str:
        preferred = self._match_preferred_name(frame.columns, PREFERRED_TARGET_NAMES)
        if preferred and preferred != time_column:
            return preferred
        numeric = [column for column in frame.select_dtypes(include=[np.number]).columns if column != time_column]
        if not numeric:
            raise ValueError("Unable to infer a numeric target column. Please pass --target-column.")
        scored = sorted(
            numeric,
            key=lambda column: (frame[column].notna().mean(), frame[column].nunique(dropna=True), frame[column].var(skipna=True)),
            reverse=True,
        )
        return scored[0]

    def _infer_series_column(self, frame: pd.DataFrame, time_column: str, target_column: str) -> str | None:
        preferred = self._match_preferred_name(frame.columns, PREFERRED_SERIES_NAMES)
        if preferred and preferred not in {time_column, target_column}:
            return preferred
        candidates = []
        for column in frame.columns:
            if column in {time_column, target_column}:
                continue
            uniques = frame[column].nunique(dropna=True)
            if frame[column].dtype == object and 1 < uniques <= min(100, max(len(frame) // 10, 2)):
                candidates.append((uniques, column))
        if candidates:
            return sorted(candidates, key=lambda item: item[0])[0][1]
        return None

    def _exogenous_columns(self, frame: pd.DataFrame, time_column: str, target_column: str, series_column: str | None) -> list[str]:
        excluded = {time_column, target_column}
        if series_column:
            excluded.add(series_column)
        return [column for column in frame.columns if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])]

    def _rolling_splits(self, n_samples: int, horizon: int) -> list[tuple[int, int]]:
        if n_samples <= horizon * 2:
            return []
        split_points = []
        last_train = max(horizon * 2, n_samples - 3 * horizon)
        for train_end in range(last_train, n_samples - horizon, horizon):
            split_points.append((train_end, min(train_end + horizon, n_samples)))
        return split_points[:3]

    def _complexity_penalty(self, candidate: str) -> float:
        return {
            "naive": 0.0,
            "seasonal_naive": 0.02,
            "ridge": 0.03,
            "ets": 0.05,
            "sarimax": 0.08,
            "boosting": 0.06,
            "lightgbm": 0.07,
            "xgboost": 0.07,
            "catboost": 0.07,
        }.get(candidate, 0.0)

    def _preference_bonus(self, profile: DatasetProfile, candidate: str) -> float:
        if candidate in {"lightgbm", "xgboost", "catboost"}:
            if profile.has_exogenous and profile.row_count >= 40:
                return 0.05
            if profile.series_count > 1 and profile.row_count >= 80:
                return 0.03
        if candidate == "boosting" and profile.has_exogenous and profile.row_count >= 40:
            return 0.03
        return 0.0

    def _should_consider_tree_boosters(self, profile: DatasetProfile, row_count: int) -> bool:
        if profile.has_exogenous and row_count >= 40:
            return True
        if profile.series_count > 1 and row_count >= 80:
            return True
        return row_count >= 60 and not profile.is_regular

    def _available_booster_candidates(self) -> list[str]:
        candidates = []
        if LGBMRegressor is not None:
            candidates.append("lightgbm")
        if XGBRegressor is not None:
            candidates.append("xgboost")
        if CatBoostRegressor is not None:
            candidates.append("catboost")
        return candidates

    def _z_value(self, interval_level: float) -> float:
        from statistics import NormalDist

        level = min(max(interval_level, 0.5), 0.99)
        return float(NormalDist().inv_cdf(0.5 + level / 2))

    def _offset_for_frequency(self, frequency: str | None, step: int) -> pd.DateOffset | pd.Timedelta:
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

    def _match_preferred_name(self, columns: Iterable[str], preferred: tuple[str, ...]) -> str | None:
        normalized = {column.lower(): column for column in columns}
        for token in preferred:
            for lower, original in normalized.items():
                if token in lower:
                    return original
        return None

    def _infer_frequency_from_series(self, time_index: pd.Series) -> str | None:
        index = pd.to_datetime(time_index).dropna().sort_values()
        if len(index) < 3:
            return None
        inferred = pd.infer_freq(index)
        if inferred:
            return inferred
        delta = index.diff().dropna().median()
        if delta <= pd.Timedelta(hours=2):
            return "H"
        if delta <= pd.Timedelta(days=2):
            return "D"
        if delta <= pd.Timedelta(weeks=2):
            return "W"
        if delta <= pd.Timedelta(days=40):
            return "MS"
        if delta <= pd.Timedelta(days=100):
            return "QS"
        return "YS"

    def _dominant_frequency(self, frequencies: list[str]) -> str | None:
        if not frequencies:
            return None
        return max(set(frequencies), key=frequencies.count)

    def _seasonal_period_from_frequency(self, frequency: str | None, time_index: pd.Series) -> int:
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

    def _is_regular_series(self, time_index: pd.Series) -> bool:
        index = pd.to_datetime(time_index).dropna().sort_values()
        if len(index) < 3:
            return True
        deltas = index.diff().dropna()
        return deltas.nunique() <= 2
