from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Iterable
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from .calendar_features import CalendarFeatureConfig, build_calendar_features, build_future_calendar_features
from .diagnostics import RoutingDiagnostics, apply_imputation, compute_routing_diagnostics
from .drift_monitor import RunState, compute_drift_signal, load_state, residual_summary, save_state

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


PREFERRED_TARGET_NAMES = ("target", "y", "value", "sales", "demand", "load", "price", "volume", "forecast_target")
PREFERRED_TIME_NAMES = ("date", "datetime", "timestamp", "time", "ds")
PREFERRED_SERIES_NAMES = ("series", "series_id", "id", "store", "item", "sku", "entity", "group")


@dataclass
class ForecastRequest:
    horizon: int
    output_path: Path
    csv_path: Path | None = None
    input_frame: pd.DataFrame | None = None
    time_column: str | None = None
    target_column: str | None = None
    series_column: str | None = None
    frequency: str | None = None
    interval_level: float = 0.9
    source_label: str = "local"
    imputation_strategy: str | None = None
    calendar_feature_config: CalendarFeatureConfig | None = None
    drift_threshold: float = 0.2
    state_path: Path | None = None
    backtest_folds: int = 5
    backtest_fold_size: int | None = None
    tree_booster: str = "lightgbm"
    compare_boosters: bool = False
    progress_callback: Callable[[str], None] | None = None


@dataclass
class ForecastResult:
    summary: str
    selection_explanation: str
    output_path: Path
    selected_model: str
    rows: int


@dataclass
class CandidateScore:
    name: str
    score: float
    metric: float
    residuals: list[float]
    rolling_metric: float | None = None
    holdout_metric: float | None = None
    validation_style: str = "rolling"
    usable_rows: int | None = None
    total_rows: int | None = None
    reliability_note: str | None = None
    backend: str | None = None


DatasetProfile = RoutingDiagnostics


class ForecastingAgent:
    """Deterministic local forecasting agent with time-aware validation."""

    def run(self, request: ForecastRequest) -> ForecastResult:
        self._current_request = request
        self._calendar_feature_config = request.calendar_feature_config or CalendarFeatureConfig()
        self._tree_booster_impl = request.tree_booster
        # Avoid joblib's environment-specific physical-core warning while keeping booster execution single-threaded.
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
        frame = request.input_frame.copy() if request.input_frame is not None else pd.read_csv(self._required_csv_path(request))
        profile = self._profile_dataset(frame, request)
        self._progress(request, f"profiled rows={profile.row_count}, series={profile.series_count}, frequency={profile.frequency or 'unknown'}")

        if profile.series_column:
            forecast_frame, selected_model, explanation = self._forecast_panel(frame, request, profile)
        else:
            forecast_frame, selected_model, explanation = self._forecast_single_series(frame, request, profile)

        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        forecast_frame.to_csv(request.output_path, index=False)
        return ForecastResult(
            summary=f"Forecast completed with {selected_model}; rows={len(forecast_frame)}; output={request.output_path}",
            selection_explanation=explanation,
            output_path=request.output_path,
            selected_model=selected_model,
            rows=len(forecast_frame),
        )

    def _profile_dataset(self, frame: pd.DataFrame, request: ForecastRequest) -> DatasetProfile:
        time_column = request.time_column or self._infer_time_column(frame)
        target_column = request.target_column or self._infer_target_column(frame, time_column)
        series_column = request.series_column or self._infer_series_column(frame, time_column, target_column)
        working = frame.copy()
        working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
        frequency = request.frequency or self._infer_frequency(working, time_column, series_column)
        return compute_routing_diagnostics(working, time_column, target_column, series_column, frequency)

    def _forecast_single_series(
        self,
        frame: pd.DataFrame,
        request: ForecastRequest,
        profile: DatasetProfile,
    ) -> tuple[pd.DataFrame, str, str]:
        prepared = self._prepare_series_frame(frame, profile, request)
        self._progress(request, "selecting candidate models")
        scores = self._score_candidates(prepared, profile, self._select_candidates(prepared, profile), request.horizon, request)
        best = self._choose_best(scores)
        drift_note = self._drift_note(request, None, best, scores)
        self._progress(request, f"fitting final model: {best.name}")
        forecast = self._fit_and_forecast(prepared, profile, best.name, request.horizon, request.interval_level, request)
        explanation = self._build_selection_explanation(profile, scores, best, drift_note)
        return forecast, best.name, explanation

    def _forecast_panel(
        self,
        frame: pd.DataFrame,
        request: ForecastRequest,
        profile: DatasetProfile,
    ) -> tuple[pd.DataFrame, str, str]:
        outputs: list[pd.DataFrame] = []
        selected_models: list[str] = []
        explanations: list[str] = []
        grouped = frame.groupby(profile.series_column, dropna=False)
        series_total = grouped.ngroups
        for series_index, (series_value, group) in enumerate(grouped, start=1):
            self._progress(request, f"processing series {series_index}/{series_total}")
            group_profile = compute_routing_diagnostics(
                group,
                profile.time_column,
                profile.target_column,
                None,
                profile.frequency,
            )
            prepared = self._prepare_series_frame(group, group_profile, request)
            self._progress(request, f"selecting models for series {series_index}/{series_total}")
            scores = self._score_candidates(prepared, group_profile, self._select_candidates(prepared, group_profile), request.horizon, request)
            best = self._choose_best(scores)
            drift_note = self._drift_note(request, series_value, best, scores)
            self._progress(request, f"fitting final model for series {series_index}/{series_total}: {best.name}")
            selected_models.append(best.name)
            explanations.append(f"Series {series_value}: {self._build_selection_explanation(group_profile, scores, best, drift_note)}")
            forecast = self._fit_and_forecast(
                prepared,
                group_profile,
                best.name,
                request.horizon,
                request.interval_level,
                request,
                series_value=series_value,
                series_column=profile.series_column,
            )
            outputs.append(forecast)

        combined = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
        return combined, "+".join(sorted(set(selected_models))), "\n".join(explanations)

    def _prepare_series_frame(self, frame: pd.DataFrame, profile: DatasetProfile, request: ForecastRequest) -> pd.DataFrame:
        prepared, imputation = apply_imputation(
            frame,
            profile.time_column,
            profile.target_column,
            profile.seasonal_period,
            request.imputation_strategy,
        )
        self._progress(
            request,
            f"imputation={imputation['strategy']}, missing_before={imputation['missing_before']}, imputed={imputation['imputed_points']}",
        )
        prepared = prepared.sort_values(profile.time_column)
        return prepared.drop_duplicates(subset=[profile.time_column], keep="last").reset_index(drop=True)

    def _select_candidates(self, frame: pd.DataFrame, profile: DatasetProfile) -> list[str]:
        candidates = ["naive"]
        has_seasonality = self._has_measured_seasonality(profile)
        # The frequency-derived period can be wrong; shortlist explicitly seasonal models only when diagnostics confirm seasonality.
        if has_seasonality and len(frame) >= max(2 * profile.seasonal_period, 12):
            candidates.append("seasonal_naive")
        if len(frame) >= 12:
            candidates.append("ridge")
        if len(frame) >= (max(3 * profile.seasonal_period, 20) if has_seasonality else 20):
            candidates.append("ets")
        if len(frame) >= 24:
            candidates.append("sarimax")
        if len(frame) >= 30:
            candidates.append("boosting")
        if self._should_consider_tree_boosters(profile, len(frame)):
            candidates.extend(self._available_booster_candidates())
        return list(dict.fromkeys(candidates))

    def _score_candidates(self, frame: pd.DataFrame, profile: DatasetProfile, candidates: Iterable[str], horizon: int, request: ForecastRequest) -> list[CandidateScore]:
        scores: list[CandidateScore] = []
        candidate_list = list(candidates)
        for index, candidate in enumerate(candidate_list, start=1):
            self._active_progress(f"scoring candidate {index}/{len(candidate_list)}: {candidate}")
            try:
                metric, residuals, rolling_metric, holdout_metric, validation_style = self._backtest_metric(frame, profile, candidate, horizon, request)
                penalty = self._complexity_penalty(candidate)
                bonus = self._preference_bonus(profile, candidate)
                # Scale tie-breakers by the MAE so preferences still matter when targets are thousands or millions.
                score = metric * max(0.0, 1.0 + penalty - bonus)
            except Exception:
                metric = float("inf")
                score = float("inf")
                residuals = []
                rolling_metric = None
                holdout_metric = None
                validation_style = "failed"
            usable_rows, total_rows, reliability_note = self._candidate_training_info(frame, profile, candidate)
            backend = self._candidate_backend(candidate)
            if usability := self._format_training_info(candidate, usable_rows, total_rows, reliability_note):
                self._active_progress(usability)
            scores.append(
                CandidateScore(
                    candidate,
                    float(score),
                    float(metric),
                    residuals,
                    rolling_metric=rolling_metric,
                    holdout_metric=holdout_metric,
                    validation_style=validation_style,
                    usable_rows=usable_rows,
                    total_rows=total_rows,
                    reliability_note=reliability_note,
                    backend=backend,
                )
            )
        return scores

    def _backtest_metric(self, frame: pd.DataFrame, profile: DatasetProfile, candidate: str, horizon: int, request: ForecastRequest) -> tuple[float, list[float], float | None, float | None, str]:
        fold_size = self._candidate_fold_size(frame, profile, candidate, horizon, request.backtest_fold_size)
        splits = self._rolling_splits(len(frame), horizon, request.backtest_folds, fold_size)
        if len(splits) < 3:
            # Thin validation makes model ranking noisy, so surface the lower confidence instead of silently trusting it.
            self._progress(request, f"warning: only {len(splits)} valid backtest folds available; model selection confidence is low")
        weighted_metrics: list[float] = []
        weights: list[float] = []
        residuals: list[float] = []
        for index, (train_end, test_end) in enumerate(splits, start=1):
            train = frame.iloc[:train_end].copy()
            test = frame.iloc[train_end:test_end].copy()
            if test.empty:
                continue
            preds = self._predict_on_holdout(train, test, profile, candidate)
            actual = test[profile.target_column].astype(float).to_numpy()
            if len(preds) != len(actual):
                continue
            fold_weight = float(index)
            weighted_metrics.append(mean_absolute_error(actual, preds) * fold_weight)
            weights.append(fold_weight)
            residuals.extend((actual - preds).astype(float).tolist())
        rolling_metric = float(np.sum(weighted_metrics) / np.sum(weights)) if weighted_metrics else None
        holdout_metric, holdout_residuals = self._single_holdout_metric(frame, profile, candidate)
        if self._is_regressor_candidate(candidate) and holdout_metric is not None:
            usable_rows, _, _ = self._candidate_training_info(frame, profile, candidate)
            # Regressors lose rows to lag construction; on small n a single holdout better matches final deployment.
            if usable_rows is not None and usable_rows < 80:
                return float(holdout_metric), holdout_residuals, rolling_metric, holdout_metric, "single_holdout"
        if rolling_metric is not None:
            return rolling_metric, residuals, rolling_metric, holdout_metric, "rolling"
        if holdout_metric is not None:
            return float(holdout_metric), holdout_residuals, rolling_metric, holdout_metric, "single_holdout"
        series = frame[profile.target_column].astype(float).to_numpy()
        fallback = np.abs(np.diff(series)).mean() if len(series) > 1 else float(np.abs(series).mean())
        return float(fallback), residuals, None, None, "fallback"

    def _single_holdout_metric(self, frame: pd.DataFrame, profile: DatasetProfile, candidate: str) -> tuple[float | None, list[float]]:
        if len(frame) < 12:
            return None, []
        test_size = max(1, min(len(frame) // 4, max(3, int(round(len(frame) * 0.25)))))
        train = frame.iloc[:-test_size].copy()
        test = frame.iloc[-test_size:].copy()
        if train.empty or test.empty:
            return None, []
        try:
            preds = self._predict_on_holdout(train, test, profile, candidate)
        except Exception:
            return None, []
        actual = test[profile.target_column].astype(float).to_numpy()
        if len(preds) != len(actual):
            return None, []
        residuals = (actual - preds).astype(float).tolist()
        return float(mean_absolute_error(actual, preds)), residuals

    def _predict_on_holdout(self, train: pd.DataFrame, test: pd.DataFrame, profile: DatasetProfile, candidate: str) -> np.ndarray:
        if candidate == "naive":
            return np.repeat(float(train[profile.target_column].iloc[-1]), len(test))
        if candidate == "seasonal_naive" and profile.seasonal_period > 1 and len(train) >= profile.seasonal_period:
            return np.resize(train[profile.target_column].iloc[-profile.seasonal_period:].to_numpy(dtype=float), len(test))
        if candidate == "ridge":
            return self._recursive_forecast_with_regressor(train, profile, len(test), self._ridge_model(train, profile))
        if candidate == "boosting":
            return self._recursive_forecast_with_regressor(train, profile, len(test), self._sklearn_boosting_model(train, profile))
        if candidate == "tree_booster":
            return self._recursive_forecast_with_regressor(train, profile, len(test), self._tree_booster_model(self._tree_booster_impl, train, profile))
        if candidate == "lightgbm" and LGBMRegressor is not None:
            return self._recursive_forecast_with_regressor(train, profile, len(test), self._tree_booster_model("lightgbm", train, profile))
        if candidate == "xgboost" and XGBRegressor is not None:
            return self._recursive_forecast_with_regressor(train, profile, len(test), self._tree_booster_model("xgboost", train, profile))
        if candidate == "catboost" and CatBoostRegressor is not None:
            return self._recursive_forecast_with_regressor(train, profile, len(test), self._tree_booster_model("catboost", train, profile))
        if candidate == "ets":
            use_seasonal = self._has_measured_seasonality(profile)
            model = ExponentialSmoothing(
                train[profile.target_column].astype(float),
                trend="add" if len(train) >= 8 else None,
                seasonal="add" if use_seasonal else None,
                seasonal_periods=profile.seasonal_period if use_seasonal else None,
                initialization_method="estimated",
            ).fit(optimized=True)
            return np.asarray(model.forecast(len(test)))
        if candidate == "sarimax":
            return self._fit_sarimax(train, profile, len(test))
        return np.repeat(float(train[profile.target_column].iloc[-1]), len(test))

    def _fit_and_forecast(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        candidate: str,
        horizon: int,
        interval_level: float,
        request: ForecastRequest,
        series_value: Any | None = None,
        series_column: str | None = None,
    ) -> pd.DataFrame:
        try:
            forecast_values, lower, upper = self._fit_full_model(frame, profile, candidate, horizon, interval_level)
        except Exception:
            value = float(frame[profile.target_column].iloc[-1])
            forecast_values = np.repeat(value, horizon)
            lower = upper = None
        future_index = self._future_index(frame[profile.time_column], horizon, profile.frequency)
        output = pd.DataFrame({"timestamp": future_index, "forecast": forecast_values, "model_name": candidate})
        if series_column is not None:
            output.insert(0, series_column, series_value)
        if lower is not None and upper is not None:
            output["lower_bound"] = lower
            output["upper_bound"] = upper
        return output

    def _fit_full_model(self, frame: pd.DataFrame, profile: DatasetProfile, candidate: str, horizon: int, interval_level: float) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        target = frame[profile.target_column].astype(float).reset_index(drop=True)
        if candidate == "naive":
            forecast = np.repeat(float(target.iloc[-1]), horizon)
            return forecast, *self._interval_from_residuals(forecast, target.diff().dropna(), interval_level)
        if candidate == "seasonal_naive" and profile.seasonal_period > 1 and len(target) >= profile.seasonal_period:
            forecast = np.resize(target.iloc[-profile.seasonal_period:].to_numpy(), horizon)
            return forecast, *self._interval_from_residuals(forecast, target.diff(profile.seasonal_period).dropna(), interval_level)
        if candidate == "ridge":
            return self._recursive_forecast_with_regressor(frame, profile, horizon, self._ridge_model(frame, profile), True, interval_level)
        if candidate == "boosting":
            return self._recursive_forecast_with_regressor(frame, profile, horizon, self._sklearn_boosting_model(frame, profile, final=True), True, interval_level)
        if candidate == "tree_booster":
            return self._recursive_forecast_with_regressor(frame, profile, horizon, self._tree_booster_model(self._tree_booster_impl, frame, profile, final=True), True, interval_level)
        if candidate == "lightgbm" and LGBMRegressor is not None:
            return self._recursive_forecast_with_regressor(frame, profile, horizon, self._tree_booster_model("lightgbm", frame, profile, final=True), True, interval_level)
        if candidate == "xgboost" and XGBRegressor is not None:
            return self._recursive_forecast_with_regressor(frame, profile, horizon, self._tree_booster_model("xgboost", frame, profile, final=True), True, interval_level)
        if candidate == "catboost" and CatBoostRegressor is not None:
            return self._recursive_forecast_with_regressor(frame, profile, horizon, self._tree_booster_model("catboost", frame, profile, final=True), True, interval_level)
        if candidate == "ets":
            use_seasonal = self._has_measured_seasonality(profile)
            model = ExponentialSmoothing(
                target,
                trend="add" if len(target) >= 8 else None,
                seasonal="add" if use_seasonal else None,
                seasonal_periods=profile.seasonal_period if use_seasonal else None,
                initialization_method="estimated",
            ).fit(optimized=True)
            forecast = np.asarray(model.forecast(horizon))
            return forecast, *self._interval_from_residuals(forecast, model.resid, interval_level)
        if candidate == "sarimax":
            forecast = self._fit_sarimax(frame, profile, horizon)
            return forecast, *self._interval_from_residuals(forecast, target.diff().dropna(), interval_level)
        forecast = np.repeat(float(target.iloc[-1]), horizon)
        return forecast, None, None

    def _fit_sarimax(self, frame: pd.DataFrame, profile: DatasetProfile, horizon: int) -> np.ndarray:
        target = frame[profile.target_column].astype(float)
        seasonal_period = profile.seasonal_period if self._has_measured_seasonality(profile) else 0
        seasonal_order = (1, 0, 0, seasonal_period) if seasonal_period else (0, 0, 0, 0)
        # Diagnostics-derived stationarity avoids unnecessary differencing when most inspected series are already stationary.
        differencing = 0 if self._stationary_share(profile) >= 0.7 else 1
        with warnings.catch_warnings():
            # Small backtest folds can make SARIMAX likelihood fits noisy; validation handles failures, so keep stdout clean.
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log")
            model = SARIMAX(target, order=(1, differencing, 1), seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        return np.asarray(model.forecast(horizon))

    def _recursive_forecast_with_regressor(
        self,
        frame: pd.DataFrame,
        profile: DatasetProfile,
        horizon: int,
        regressor: Any,
        return_interval: bool = False,
        interval_level: float = 0.9,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None] | np.ndarray:
        features, target = self._build_supervised_features(frame, profile)
        if features.empty:
            fallback = np.repeat(float(frame[profile.target_column].iloc[-1]), horizon)
            return (fallback, None, None) if return_interval else fallback
        # Ridge is scaled and regularized; catch only the known numerical warning at this fit site.
        with warnings.catch_warnings():
            try:
                from scipy.linalg import LinAlgWarning

                warnings.filterwarnings("ignore", category=LinAlgWarning)
            except Exception:
                pass
            # Some booster libraries ask joblib for physical cores even with n_jobs=1; this environment can report none.
            warnings.filterwarnings("ignore", category=UserWarning, message="Could not find the number of physical cores.*")
            regressor.fit(features, target)
        history = frame[profile.target_column].astype(float).tolist()
        rows: list[float] = []
        for step in range(1, horizon + 1):
            row = self._feature_row_from_history(history, frame[profile.time_column].iloc[-1], profile, step, features.columns)
            pred = float(regressor.predict(pd.DataFrame([row], columns=features.columns))[0])
            history.append(pred)
            rows.append(pred)
        forecast = np.asarray(rows)
        if return_interval:
            residuals = target.to_numpy(dtype=float) - np.asarray(regressor.predict(features), dtype=float)
            lower, upper = self._interval_from_residuals(forecast, residuals, interval_level)
            return forecast, lower, upper
        return forecast

    def _build_supervised_features(self, frame: pd.DataFrame, profile: DatasetProfile, max_lag: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
        target = frame[profile.target_column].astype(float).reset_index(drop=True)
        data = pd.DataFrame({"y": target})
        max_lag = self._adaptive_max_lag(len(data)) if max_lag is None else max_lag
        # Tiny series used to lose too many rows to fixed lag depth; adapt lags to improve fair comparison.
        lag_limit = min(max_lag, max(1, len(data) // 4))
        for lag in range(1, lag_limit + 1):
            data[f"lag_{lag}"] = data["y"].shift(lag)
        data["rolling_mean_3"] = data["y"].shift(1).rolling(3).mean()
        data["rolling_std_3"] = data["y"].shift(1).rolling(3).std()
        data["trend_index"] = np.arange(len(data), dtype=float)
        calendar = build_calendar_features(frame[profile.time_column].reset_index(drop=True), self._calendar_config())
        data = pd.concat([data, calendar.reset_index(drop=True)], axis=1).dropna()
        if data.empty:
            return pd.DataFrame(), pd.Series(dtype=float)
        y = data.pop("y")
        return data, y

    def _feature_row_from_history(self, history: list[float], last_timestamp: Any, profile: DatasetProfile, step: int, columns: Iterable[str]) -> dict[str, float]:
        row: dict[str, float] = {column: 0.0 for column in columns}
        row["trend_index"] = float(len(history) + step - 1)
        for lag in range(1, min(8, len(history)) + 1):
            key = f"lag_{lag}"
            if key in row:
                row[key] = float(history[-lag])
        recent = np.asarray(history[-3:] if len(history) >= 3 else history, dtype=float)
        row["rolling_mean_3"] = float(np.mean(recent))
        row["rolling_std_3"] = float(np.std(recent)) if len(recent) > 1 else 0.0
        for key, value in build_future_calendar_features(last_timestamp, profile.frequency, step, self._calendar_config()).items():
            if key in row:
                row[key] = float(value)
        return row

    def _build_selection_explanation(self, profile: DatasetProfile, scores: list[CandidateScore], best: CandidateScore, drift_note: str) -> str:
        ordered = sorted(scores, key=lambda item: item.score)
        candidate_lines = []
        for item in ordered:
            metric = "inf" if not np.isfinite(item.metric) else f"{item.metric:.4f}"
            score = "inf" if not np.isfinite(item.score) else f"{item.score:.4f}"
            parts = [f"- {item.name}: validation_mae={metric}", f"adjusted_score={score}", f"validation={item.validation_style}"]
            if item.rolling_metric is not None:
                parts.append(f"rolling_mae={item.rolling_metric:.4f}")
            if item.holdout_metric is not None:
                parts.append(f"holdout_mae={item.holdout_metric:.4f}")
            if item.backend:
                parts.append(f"backend={item.backend}")
            if item.usable_rows is not None and item.total_rows is not None:
                parts.append(f"trained_on={item.usable_rows}/{item.total_rows}_rows_after_lag_construction")
            if item.reliability_note:
                parts.append(item.reliability_note)
            candidate_lines.append(", ".join(parts))
        signals = (
            f"rows={profile.row_count}, series_count={profile.series_count}, frequency={profile.frequency or 'unknown'}, "
            f"seasonal_period={profile.seasonal_period}, regular_index={'yes' if profile.is_regular else 'no'}, "
            f"exogenous_features={'yes' if profile.has_exogenous else 'no'}"
        )
        warning = self._diagnostic_warning(profile)
        return (
            f"Data signals: {signals}. Diagnostics: {profile.diagnostics_summary}. "
            f"{warning}"
            f"Chosen model: {best.name} because it had the lowest adjusted score on rolling time-aware validation. "
            f"{drift_note} Candidates:\n" + "\n".join(candidate_lines)
        )

    def _drift_note(self, request: ForecastRequest, series_value: Any | None, best: CandidateScore, scores: list[CandidateScore]) -> str:
        state_path = self._state_path(request)
        series_key = self._safe_state_name(str(series_value)) if series_value is not None else None
        current_sample = best.residuals[-2000:]
        previous = load_state(state_path, series_key=series_key)
        note = "No previous residual state was available for drift comparison."
        if previous and previous.residual_sample:
            signal = compute_drift_signal(previous.residual_sample, current_sample, request.drift_threshold)
            note = f"Residual drift score={signal.combined_score:.4f}; retrain_trigger={'yes' if signal.trigger else 'no'}."
        save_state(
            state_path,
            RunState(
                selected_model=best.name,
                residual_summary=residual_summary(current_sample),
                drift_score=0.0,
                residual_sample=current_sample,
            ),
            series_key=series_key,
        )
        return note

    def _state_path(self, request: ForecastRequest) -> Path:
        if request.state_path:
            return request.state_path
        # Store drift state in a dedicated folder and consolidate panel series into one JSON file.
        return request.output_path.parent / ".forecast_state" / "drift_state.json"

    def _choose_best(self, scores: list[CandidateScore]) -> CandidateScore:
        finite = [score for score in scores if np.isfinite(score.score)]
        return min(finite, key=lambda item: item.score) if finite else CandidateScore("naive", 0.0, 0.0, [])

    def _interval_from_residuals(self, forecast: np.ndarray, residuals: Iterable[float], interval_level: float) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(list(residuals), dtype=float)
        values = values[np.isfinite(values)]
        spread = self._z_value(interval_level) * float(np.std(values)) if len(values) else 0.0
        return forecast - spread, forecast + spread

    def _rolling_splits(self, n_samples: int, horizon: int, max_folds: int = 5, fold_size: int | None = None) -> list[tuple[int, int]]:
        fold_size = max(1, int(fold_size or horizon))
        max_folds = max(1, int(max_folds))
        if n_samples <= horizon + fold_size:
            return []
        split_points = []
        # Fold count and size are configurable so large horizons do not force very thin validation.
        first_train = max(fold_size * 2, n_samples - max_folds * fold_size)
        for train_end in range(first_train, n_samples - fold_size + 1, fold_size):
            split_points.append((train_end, min(train_end + fold_size, n_samples)))
        return split_points[-max_folds:]

    def _future_index(self, time_index: pd.Series, horizon: int, frequency: str | None) -> pd.Index:
        observed = pd.to_datetime(time_index).dropna().sort_values()
        if observed.empty:
            return pd.RangeIndex(1, horizon + 1)
        if frequency:
            try:
                return pd.date_range(start=observed.iloc[-1], periods=horizon + 1, freq=frequency)[1:]
            except Exception:
                pass
        if len(observed) >= 2:
            delta = observed.diff().dropna().median()
            return pd.Index([observed.iloc[-1] + delta * (i + 1) for i in range(horizon)])
        return pd.Index([observed.iloc[-1] + pd.Timedelta(days=i + 1) for i in range(horizon)])

    def _infer_time_column(self, frame: pd.DataFrame) -> str:
        preferred = self._match_preferred_name(frame.columns, PREFERRED_TIME_NAMES)
        if preferred:
            return preferred
        scores = []
        for column in frame.columns:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            scores.append((parsed.notna().mean() * 2 + parsed.nunique(dropna=True) / max(len(parsed), 1), column))
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
        return sorted(numeric, key=lambda column: (frame[column].notna().mean(), frame[column].nunique(dropna=True), frame[column].var(skipna=True)), reverse=True)[0]

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
        return sorted(candidates, key=lambda item: item[0])[0][1] if candidates else None

    def _infer_frequency(self, frame: pd.DataFrame, time_column: str, series_column: str | None) -> str | None:
        groups = frame.groupby(series_column, dropna=False) if series_column else [(None, frame)]
        frequencies = [self._infer_frequency_from_series(group[time_column]) for _, group in groups]
        frequencies = [frequency for frequency in frequencies if frequency]
        return max(set(frequencies), key=frequencies.count) if frequencies else None

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

    def _should_consider_tree_boosters(self, profile: DatasetProfile, row_count: int) -> bool:
        return (profile.has_exogenous and row_count >= 40) or (profile.series_count > 1 and row_count >= 80) or row_count >= 60

    def _available_booster_candidates(self) -> list[str]:
        if getattr(self, "_current_request", None) and self._current_request.compare_boosters:
            candidates = []
            if LGBMRegressor is not None:
                candidates.append("lightgbm")
            if XGBRegressor is not None:
                candidates.append("xgboost")
            if CatBoostRegressor is not None:
                candidates.append("catboost")
            return candidates
        # Default to one configurable booster candidate to avoid redundant compute and side effects.
        return ["tree_booster"] if self._booster_available(getattr(self, "_tree_booster_impl", "lightgbm")) else []

    def _complexity_penalty(self, candidate: str) -> float:
        return {"naive": 0.0, "seasonal_naive": 0.005, "ridge": 0.01, "ets": 0.015, "sarimax": 0.025, "boosting": 0.02, "tree_booster": 0.025, "lightgbm": 0.025, "xgboost": 0.025, "catboost": 0.025}.get(candidate, 0.0)

    def _preference_bonus(self, profile: DatasetProfile, candidate: str) -> float:
        if candidate in {"tree_booster", "lightgbm", "xgboost", "catboost"} and profile.has_exogenous and profile.row_count >= 40:
            return 0.015
        if candidate == "boosting" and profile.has_exogenous and profile.row_count >= 40:
            return 0.01
        return 0.0

    def _is_regressor_candidate(self, candidate: str) -> bool:
        return candidate in {"ridge", "boosting", "tree_booster", "lightgbm", "xgboost", "catboost"}

    def _adaptive_max_lag(self, row_count: int) -> int:
        return max(1, min(4, row_count // 6)) if row_count < 80 else 8

    def _candidate_training_info(self, frame: pd.DataFrame, profile: DatasetProfile, candidate: str) -> tuple[int | None, int | None, str | None]:
        if not self._is_regressor_candidate(candidate):
            return None, None, None
        features, _ = self._build_supervised_features(frame, profile)
        usable_rows = int(len(features))
        total_rows = int(len(frame))
        note = "low-data-reliability" if usable_rows < 40 else None
        return usable_rows, total_rows, note

    def _format_training_info(self, candidate: str, usable_rows: int | None, total_rows: int | None, note: str | None) -> str | None:
        if usable_rows is None or total_rows is None:
            return None
        suffix = f" ({note})" if note else ""
        return f"{candidate} trained on {usable_rows}/{total_rows} rows after lag construction{suffix}"

    def _candidate_fold_size(self, frame: pd.DataFrame, profile: DatasetProfile, candidate: str, horizon: int, requested: int | None) -> int:
        if requested:
            return requested
        if self._is_regressor_candidate(candidate):
            usable_rows, _, _ = self._candidate_training_info(frame, profile, candidate)
            if usable_rows is not None and usable_rows < 80:
                # Use wider folds for small regressor histories so lag dropna is not compounded by tiny fragments.
                return max(horizon, min(max(horizon, len(frame) // 4), max(1, len(frame) - 1)))
        return horizon

    def _ridge_model(self, frame: pd.DataFrame, profile: DatasetProfile) -> Any:
        usable_rows, _, _ = self._candidate_training_info(frame, profile, "ridge")
        alpha = 10.0 if (usable_rows or 0) < 40 else 3.0 if (usable_rows or 0) < 100 else 1.0
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def _sklearn_boosting_model(self, frame: pd.DataFrame, profile: DatasetProfile, final: bool = False) -> HistGradientBoostingRegressor:
        usable_rows, _, _ = self._candidate_training_info(frame, profile, "boosting")
        rows = usable_rows or len(frame)
        max_iter = 80 if rows < 80 else 150 if rows < 200 else (250 if final else 200)
        l2 = 1.0 if rows < 80 else 0.1 if rows < 200 else 0.0
        return HistGradientBoostingRegressor(max_depth=3 if rows < 100 else 4, learning_rate=0.04 if rows < 100 else 0.05, max_iter=max_iter, l2_regularization=l2, random_state=42)

    def _tree_booster_model(self, backend: str, frame: pd.DataFrame, profile: DatasetProfile, final: bool = False) -> Any:
        usable_rows, _, _ = self._candidate_training_info(frame, profile, "tree_booster")
        rows = usable_rows or len(frame)
        estimators = 80 if rows < 80 else 150 if rows < 200 else (300 if final else 220)
        depth = 3 if rows < 100 else 5
        learning_rate = 0.04 if rows < 100 else 0.05
        if backend == "lightgbm" and LGBMRegressor is not None:
            return LGBMRegressor(n_estimators=estimators, learning_rate=learning_rate, max_depth=depth, min_child_samples=max(5, min(20, rows // 4)), reg_lambda=1.0 if rows < 100 else 0.1, n_jobs=1, verbose=-1, random_state=42)
        if backend == "xgboost" and XGBRegressor is not None:
            return XGBRegressor(n_estimators=estimators, learning_rate=learning_rate, max_depth=depth, reg_lambda=2.0 if rows < 100 else 1.0, objective="reg:squarederror", n_jobs=1, verbosity=0, random_state=42)
        if backend == "catboost" and CatBoostRegressor is not None:
            return CatBoostRegressor(iterations=estimators, learning_rate=learning_rate, depth=depth, l2_leaf_reg=6.0 if rows < 100 else 3.0, loss_function="RMSE", verbose=False, random_seed=42, thread_count=1, train_dir=str(self._booster_log_dir()))
        raise ValueError(f"Tree booster '{backend}' is not available.")

    def _booster_available(self, backend: str) -> bool:
        return (backend == "lightgbm" and LGBMRegressor is not None) or (backend == "xgboost" and XGBRegressor is not None) or (backend == "catboost" and CatBoostRegressor is not None)

    def _candidate_backend(self, candidate: str) -> str | None:
        return getattr(self, "_tree_booster_impl", "lightgbm") if candidate == "tree_booster" else (candidate if candidate in {"lightgbm", "xgboost", "catboost"} else None)

    def _booster_log_dir(self) -> Path:
        request = getattr(self, "_current_request", None)
        base = request.output_path.parent if request else Path.cwd()
        path = base / ".forecast_state" / "booster_logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _has_measured_seasonality(self, profile: DatasetProfile, threshold: float = 0.3) -> bool:
        return profile.seasonal_period > 1 and float(profile.diagnostics_summary.get("avg_seasonal_strength", 0.0)) >= threshold

    def _stationary_share(self, profile: DatasetProfile) -> float:
        return float(profile.diagnostics_summary.get("stationary_share", 0.0))

    def _diagnostic_warning(self, profile: DatasetProfile) -> str:
        avg_strength = float(profile.diagnostics_summary.get("avg_seasonal_strength", 0.0))
        if profile.seasonal_period > 1 and avg_strength < 0.1:
            return (
                f"Warning: frequency implies seasonal_period={profile.seasonal_period}, "
                f"but measured avg_seasonal_strength={avg_strength:.4f}; seasonal candidates/components were limited. "
            )
        return ""

    def _z_value(self, interval_level: float) -> float:
        from statistics import NormalDist

        level = min(max(interval_level, 0.5), 0.99)
        return float(NormalDist().inv_cdf(0.5 + level / 2))

    def _match_preferred_name(self, columns: Iterable[str], preferred: tuple[str, ...]) -> str | None:
        normalized = {str(column).lower(): str(column) for column in columns}
        for token in preferred:
            for lower, original in normalized.items():
                if token in lower:
                    return original
        return None

    def _safe_state_name(self, value: str) -> str:
        safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
        return safe[:80] or "local"

    def _required_csv_path(self, request: ForecastRequest) -> Path:
        if request.csv_path is None:
            raise ValueError("ForecastRequest requires either input_frame or csv_path.")
        return request.csv_path

    def _progress(self, request: ForecastRequest, message: str) -> None:
        if request.progress_callback:
            request.progress_callback(message)

    def _active_progress(self, message: str) -> None:
        if hasattr(self, "_current_request"):
            self._progress(self._current_request, message)

    def _calendar_config(self) -> CalendarFeatureConfig:
        return getattr(self, "_calendar_feature_config", CalendarFeatureConfig())
