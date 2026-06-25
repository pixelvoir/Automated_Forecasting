# Automated Forecasting Agent Workflow

This document explains, in plain language, how the local automated forecasting agent works: how it reads data, infers columns, diagnoses the time series, chooses model candidates, scores them, and writes forecasts.

The agent is deterministic and local-only. It does not use an LLM for database access, column inference, diagnostics, model selection, scoring, or forecasting.

## Big Picture

The agent follows this flow:

1. Load data from CSV, Parquet, or PostgreSQL.
2. Infer the time column, target column, optional series column, and frequency if the user did not provide them.
3. Let the user confirm or override those choices.
4. Clean and profile each time series locally.
5. Compute diagnostics such as seasonality strength, stationarity, missing values, and index regularity.
6. Build a small candidate model shortlist from those diagnostics.
7. Backtest each candidate using time-aware validation.
8. Adjust scores with small percentage-based complexity and preference factors.
9. Pick the lowest adjusted score.
10. Fit the winner on the full available history.
11. Forecast the requested horizon.
12. Save the forecast CSV and drift state locally.

## CLI Inputs

File inputs:

- `--csv`: load a CSV file.
- `--parquet`: load a Parquet file.

Database inputs:

- `--db-host`: PostgreSQL host.
- `--db-port`: PostgreSQL port.
- `--db-name`: database name.
- `--db-user`: database user.
- `--db-password`: database password.
- `--db-schema`: schema name.
- `--db-table`: table name.
- `--db-sslmode`: optional SSL mode.
- `--pro-max`: opt in to exact row counts and full SQL column profiling.
- `--start-date`: optional inclusive lower bound for database aggregation.
- `--end-date`: optional exclusive upper bound for database aggregation.

Forecast inputs:

- `--horizon`: number of future time steps to forecast.
- `--output`: output CSV path.
- `--time-column`: explicit time column override.
- `--target-column`: explicit target column override.
- `--series-column`: optional group/entity column for panel forecasting.
- `--frequency`: explicit frequency override such as `D`, `W`, `MS`, `QS`.
- `--interval-level`: prediction interval level, default `0.9`.
- `--imputation-strategy`: optional missing-value strategy override.
- `--calendar-feature-set`: calendar feature set, default `basic`.
- `--calendar-region`: optional region code for holiday features.
- `--drift-threshold`: residual drift trigger threshold, default `0.2`.
- `--tree-booster`: implementation behind `tree_booster`; choices are `lightgbm`, `xgboost`, `catboost`; default is `lightgbm`.
- `--compare-boosters`: opt in to scoring LightGBM, XGBoost, and CatBoost separately.

The CLI opens a PostgreSQL connection for that run, uses it for metadata and data loading, then closes it. It does not keep a background connection alive.

## Data Loading

For file inputs, the agent loads the full file into a local pandas dataframe. It then profiles that dataframe and asks the user to confirm inferred columns.

For database inputs, the agent prompts for missing connection values, lists base tables in the chosen schema, lets the user choose a table, reads table metadata, optionally computes full SQL-side stats in `--pro-max` mode, and loads an aggregated modeling dataframe with `date_trunc(...)`.

Database column profiling separates date-like text from numeric-like text. This avoids showing numeric text columns as failed date parses.

## Column Inference

Column inference is intentionally simple and reviewable. It first looks for common names. If names are not enough, it scores columns using basic data properties.

### Time Column

The agent prefers column names containing:

- `date`
- `datetime`
- `timestamp`
- `time`
- `ds`

If no preferred name is found, it tries to parse each column as datetimes and scores it using datetime parse success rate and number of unique parsed values relative to row count.

Why: a good time column should parse cleanly and usually have many distinct ordered values.

In database `--pro-max` mode, text columns report `date_parse_success` separately from `numeric_parse_success`, so numeric IDs or numeric target columns are not mistaken for failed date columns.

### Target Column

The agent prefers column names containing:

- `target`
- `y`
- `value`
- `sales`
- `demand`
- `load`
- `price`
- `volume`
- `forecast_target`

If no preferred name is found, it chooses among numeric columns, excluding the time column, using non-null rate, unique value count, and variance.

Why: the target should be numeric, sufficiently populated, and variable enough to forecast.

### Series Column

The series column is optional. It is used when one dataset contains many independent entities, such as societies, stores, SKUs, customers, or locations.

The agent prefers names containing:

- `series`
- `series_id`
- `id`
- `store`
- `item`
- `sku`
- `entity`
- `group`

If no preferred name is found, it looks for object/text columns with more than one value but not too many distinct values.

Why: a useful series column should split the data into meaningful groups without making every row its own series.

## Frequency and Time Index Checks

The agent uses `--frequency` if present. Otherwise it tries `pandas.infer_freq`, then falls back to median timestamp delta.

Median delta maps approximately to:

- `H` for hourly
- `D` for daily
- `W` for weekly
- `MS` for monthly
- `QS` for quarterly
- `YS` for yearly

The agent also checks whether the time index is regular. A regular index has mostly consistent time gaps.

## Seasonal Period

Frequency is mapped to a default seasonal period:

- hourly -> `24`
- daily -> `7`
- weekly -> `52`
- monthly -> `12`
- quarterly -> `4`
- otherwise -> `1`

This is only a frequency-derived assumption. The agent also measures actual seasonality from the data. If frequency implies a seasonal period but measured seasonality is near zero, the explanation warns the user and seasonal candidates/components are limited.

## Local Diagnostics

For each inspected series, the agent computes:

- row count
- missing rate
- outlier rate
- STL trend strength
- STL seasonal strength
- ADF p-value
- KPSS p-value
- ACF peak lag
- PACF peak lag
- simple order bounds for SARIMAX-style routing
- imputation recommendation

For panel data, it inspects a bounded number of series and summarizes them.

The most important routing diagnostics are:

- `avg_seasonal_strength`: whether seasonal model components are justified.
- `stationary_share`: whether differencing is likely needed.
- `avg_missing_rate`: how much imputation risk exists.
- `avg_outlier_rate`: how noisy or spiky the series may be.
- `has_exogenous`: whether extra numeric columns exist beyond time, target, and series.
- `is_regular`: whether timestamps are evenly spaced.

Warnings from known benign statistical edge cases are scoped and converted into notes where possible, so CLI output stays readable.

## Imputation

Before forecasting a series, the agent parses timestamps, sorts by time, drops invalid timestamps, clips outliers using IQR bounds when meaningful, and fills missing target values.

Default strategy:

- `seasonal_interpolate` when a seasonal period exists.
- `time_interpolate` otherwise.
- `ffill_bfill` when missingness is high.

Why: model comparison is only meaningful if each candidate sees a clean, ordered target series.

## Candidate Routing

The agent does not brute-force every possible model. It builds a compact shortlist using row count and diagnostics.

### Always Included

- `naive`

Why: the last-value baseline is a sanity check. Any serious model should beat it on validation.

### Seasonal Naive

Included when measured seasonality is strong enough, currently average seasonal strength >= `0.3`, and enough history exists: at least `max(2 * seasonal_period, 12)` rows.

Why: seasonal naive is only fair when there is real measured seasonality, not merely a frequency label.

### Ridge

Included when at least `12` rows exist.

Why: ridge is a fast linear supervised baseline using lag and calendar features. It is useful when patterns are simple or data is small.

### ETS

Included when at least `20` rows exist for non-seasonal ETS, or enough history exists for seasonal ETS when measured seasonality is present.

Why: ETS is data-efficient and strong for level/trend/seasonal business series. If measured seasonality is weak, the agent falls back to non-seasonal ETS.

### SARIMAX

Included when at least `24` rows exist.

The differencing order is guided by `stationary_share`:

- high stationary share -> avoid unnecessary differencing
- otherwise -> use differencing

Seasonal SARIMAX terms are only used when measured seasonality supports them.

Why: SARIMAX is strong for autocorrelated series, but seasonal and differencing terms should be justified by diagnostics.

### HistGradientBoosting

Included when at least `30` rows exist.

Why: this is a local scikit-learn tree boosting baseline. It can learn nonlinear lag/calendar patterns but needs enough usable lag-feature rows.

### Tree Booster

The default external booster candidate is `tree_booster`. By default it uses LightGBM because it is fast and has a low side-effect footprint.

It is considered when boosters are justified:

- exogenous signal exists and rows >= `40`, or
- panel-style structure exists and rows >= `80`, or
- rows >= `60`

The user can choose the implementation with `--tree-booster lightgbm`, `--tree-booster xgboost`, or `--tree-booster catboost`.

The old behavior of comparing all three external boosters is available only with `--compare-boosters`.

Why: scoring LightGBM, XGBoost, and CatBoost separately triples compute and clutters explanations. A single configurable booster is a better default.

CatBoost log output is written under `.forecast_state/booster_logs`, not the project root.

## Lag and Calendar Features

Regressor-based models use supervised features:

- lag values
- rolling mean
- rolling standard deviation
- trend index
- calendar features from the configured calendar feature set

Lag depth is adaptive:

- tiny series use fewer lags
- larger series may use up to 8 lags

Why: fixed deep lags can consume too many rows on small series. Adaptive lag depth makes comparisons fairer between statistical models and data-hungry regressors.

## Small-Sample Reliability

For each regressor candidate, the agent reports how many rows remain after lag-feature `dropna`.

Example:

`tree_booster trained on 55/63 rows after lag construction`

If usable rows fall below `40`, the candidate is marked `low-data-reliability`.

Why: a regressor can look worse than ETS or SARIMAX simply because lag construction removed a large share of the training data. The explanation makes that tradeoff visible.

## Validation and Model Selection

The validation metric is MAE.

Most candidates use rolling-origin backtesting:

- no random shuffle
- train on the past
- test on the next block
- repeat over recent folds

Defaults:

- up to `5` folds
- fold size defaults to forecast horizon
- fewer than `3` valid folds triggers a low-confidence warning

Why: forecasting must be validated in time order. Random train/test splits leak future information.

Regressor candidates also report a single holdout MAE:

- train on the earlier portion
- test on the last roughly 20-25% of rows

When usable lag-feature rows are small, the scorer may use this single holdout instead of fragmented rolling folds.

Why: rolling folds can unfairly punish regressors on small series because each fold loses rows to lag construction. The final deployed model only loses those rows once, so the single holdout can be a better comparison.

Regressor candidates may also use wider fold blocks on small histories. This reduces the double penalty of tiny folds plus lag-feature row loss.

## Adjusted Score

Candidates are ranked by:

`adjusted_score = validation_mae * (1 + complexity_penalty_fraction - preference_bonus_fraction)`

The penalties and bonuses are percentages, not flat constants.

Why: a flat `0.05` bonus is meaningless when MAE is 500,000. A percentage adjustment stays meaningful across different target scales while still acting mainly as a tie-breaker.

Current complexity penalties:

- `naive`: `0.0%`
- `seasonal_naive`: `0.5%`
- `ridge`: `1.0%`
- `ets`: `1.5%`
- `boosting`: `2.0%`
- `sarimax`: `2.5%`
- `tree_booster`: `2.5%`
- separate `lightgbm` / `xgboost` / `catboost`: `2.5%`

Preference bonuses:

- external tree booster with exogenous signal and enough rows: `1.5%`
- scikit-learn boosting with exogenous signal and enough rows: `1.0%`

Why: simpler models get a slight preference unless a more complex model clearly validates better.

## Hyperparameter Scaling

Tree and boosting hyperparameters are adjusted by usable training rows.

Small samples use fewer estimators/iterations, shallower trees, and stronger regularization.

Larger samples use more estimators, larger tree depth, and lighter regularization.

Why: fixed booster settings can overfit small histories and underfit larger datasets.

## Final Fit and Forecast

After scoring, the winner is retrained on the full prepared history.

Model behavior:

- `naive`: repeats the last observed target.
- `seasonal_naive`: repeats the last seasonal pattern.
- `ridge`: fits scaled ridge regression on lag/calendar features and forecasts recursively.
- `boosting`: fits scikit-learn histogram gradient boosting and forecasts recursively.
- `tree_booster`: fits the selected LightGBM/XGBoost/CatBoost backend and forecasts recursively.
- `ets`: fits exponential smoothing with trend and optional measured seasonality.
- `sarimax`: fits SARIMAX with diagnostics-guided differencing and optional measured seasonality.

If the selected model fails during final fitting, the agent falls back to a safe last-value forecast.

## Prediction Intervals

When available, intervals are estimated from residual spread and the requested `interval_level`.

- naive: residuals from first differences
- seasonal naive: seasonal differences
- regressors: in-sample residuals from supervised fit
- ETS: model residuals
- SARIMAX: differenced residual spread

The interval method is lightweight. It is meant as a practical uncertainty band, not a full probabilistic forecast.

## Drift State

The agent stores residual drift state locally under:

`.forecast_state/drift_state.json`

For panel forecasting, all series states are consolidated into that one file and keyed by series identifier.

Why: this avoids creating thousands of per-series JSON files in the project folder.

The drift state stores the selected model, residual summary, residual sample, and drift score. On later runs, the agent compares previous residuals with current residuals and reports whether drift exceeds the configured threshold.

## Panel Forecasting

If a series column is selected:

1. The dataframe is split by series.
2. Each series is diagnosed independently.
3. Each series gets its own candidate shortlist and validation scores.
4. The best model is fit per series.
5. Forecast rows are concatenated.
6. Drift state is saved per series key in the consolidated state file.

Why: different entities can have different trends, seasonality, noise, and best model families.

## Output CSV

The output contains:

- `timestamp`
- `forecast`
- `model_name`
- optional series column
- optional `lower_bound`
- optional `upper_bound`

## Explanation Output

The final explanation includes:

- dataset signals
- diagnostics summary
- seasonality mismatch warnings when applicable
- chosen model
- drift note
- each candidate validation MAE
- adjusted score
- rolling MAE
- holdout MAE
- validation style used for ranking
- tree booster backend when relevant
- usable rows after lag construction for regressors
- low-data reliability notes when relevant

This is designed to make the model choice auditable rather than magical.

## Implementation Map

- CLI and prompts: `src/automated_forecasting/cli.py`
- Database helpers: `src/automated_forecasting/db_source.py`
- Diagnostics and imputation: `src/automated_forecasting/diagnostics.py`
- Routing, validation, fitting, forecasting: `src/automated_forecasting/pipeline.py`
- Drift state: `src/automated_forecasting/drift_monitor.py