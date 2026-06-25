# Automated Forecasting Agent Workflow (Database-First, Local-Only)

## What this document is

This is a simplified, code-accurate description of how the current agent connects to a local PostgreSQL database, inspects metadata, lets the user confirm the inferred columns, and then runs the forecasting workflow entirely on the local machine.

## End-to-end flow

1. Ask for PostgreSQL connection details if they were not passed on the command line.
2. Connect locally to the database.
3. List available tables in the chosen schema.
4. Let the user choose the table to forecast from.
5. Read table metadata, infer columns from a small preview sample, then load only the columns needed for forecasting into a local dataframe.
6. Infer the time column, target column, optional series column, and frequency from the preview sample.
7. Show the inferred metadata and let the user edit it before continuing.
8. Ask for the forecast horizon and output CSV path.
9. Build a dataset profile from local diagnostics.
10. Route to a small model shortlist using deterministic rules.
11. Score candidates with time-aware backtesting.
12. Pick the model with the best adjusted score.
13. Retrain that model on full history.
14. Forecast the requested horizon.
15. Write the forecast output to CSV.

No LLM is used for database access, routing, or model choice.

## Inputs supported by the CLI

Database mode:

- `--db-host`: PostgreSQL server host
- `--db-port`: PostgreSQL server port
- `--db-name`: PostgreSQL database name
- `--db-user`: PostgreSQL username
- `--db-password`: PostgreSQL password
- `--db-schema`: PostgreSQL schema name
- `--db-table`: Optional table name override
- `--db-sslmode`: Optional PostgreSQL sslmode

Forecast settings:

- `--horizon`: forecast horizon in time steps
- `--output`: output forecast CSV path
- `--time-column`: optional explicit time column
- `--target-column`: optional explicit target column
- `--series-column`: optional panel/group identifier
- `--frequency`: optional frequency override
- `--interval-level`: optional prediction interval level (default 0.9)

CSV mode still works for compatibility, but the database flow is the primary path.

## Step 1: Connection and table discovery

The CLI prompts for host, port, database, username, password, and schema when those values are missing.

After connecting, it queries `information_schema.tables` and shows the available base tables in the selected schema.

## Step 2: Table metadata

The agent reads table metadata locally:

- column names
- column data types
- nullability
- approximate dimensional structure through row count and column list

It then loads the selected table into an in-memory dataframe on the local machine.

## Step 3: Schema and profile detection

The agent infers structure if fields are not explicitly provided.

### Time column inference

- Prefer names containing: `date`, `datetime`, `timestamp`, `time`, `ds`.
- Otherwise score each column by datetime parse success and uniqueness.

### Target inference

- Prefer names containing: `target`, `y`, `value`, `sales`, `demand`, `load`, `price`, `volume`.
- Otherwise choose the strongest numeric candidate by non-nullness, uniqueness, and variance.

### Series inference (optional)

- Prefer names containing: `series`, `series_id`, `id`, `store`, `item`, `sku`, `entity`, `group`.
- Otherwise detect suitable low/medium-cardinality object columns.

### Frequency and regularity

For single series or each panel group:

- Try `pandas.infer_freq`.
- If unavailable, infer from median time delta (`H`, `D`, `W`, `MS`, `QS`, `YS`).
- Mark whether the time index is regular.

### Seasonal period mapping

- Hourly -> 24
- Daily -> 7
- Weekly -> 52
- Monthly -> 12
- Quarterly -> 4
- Otherwise -> 1

### Exogenous detection signal

`has_exogenous` is true when numeric columns exist beyond time/target/(optional) series columns.

## Step 4: User confirmation of inferred columns

The CLI prints the inferred time column, target column, optional series column, and frequency.

The user can press Enter to keep each guess or type a different column name before the model runs.

This is still local-only: the dataframe never leaves the machine.

## Step 5: Candidate routing (shortlist)

The router builds a compact shortlist from profile signals. It does not brute-force every model.

### Always include

- `naive`

### Add conditionally

- `seasonal_naive` if seasonal period > 1 and enough history (`>= max(2*seasonal_period, 12)`).
- `ridge` if at least 12 rows.
- `ets` if seasonal period > 1 and enough history (`>= max(3*seasonal_period, 20)`).
- `sarimax` if at least 24 rows.
- `boosting` (HistGradientBoosting) if at least 30 rows.

### Optional tree boosters (if installed and justified)

The agent may add `lightgbm`, `xgboost`, and/or `catboost` if package is available and one of these is true:

- Exogenous signal exists and rows >= 40, or
- Panel-style structure exists and rows >= 80, or
- Rows >= 60 with irregular time index.

This is how the agent narrows to the most plausible model families before scoring.

## Step 6: Time-aware scoring

Each candidate is evaluated using rolling-origin style backtesting.

### Split logic

- Build up to 3 rolling splits.
- Each test segment size is horizon.
- No random shuffle.

### Metric

- Validation metric is MAE.

### Adjusted score used for ranking

The winner is chosen by minimizing:

`adjusted_score = validation_mae + complexity_penalty - preference_bonus`

Complexity penalties lightly favor simpler models:

- naive: 0.00
- seasonal_naive: 0.02
- ridge: 0.03
- ets: 0.05
- boosting: 0.06
- lightgbm/xgboost/catboost: 0.07
- sarimax: 0.08

Preference bonuses can offset this when data supports boosters:

- +0.05 for `lightgbm/xgboost/catboost` when exogenous signal and rows >= 40
- +0.03 for `lightgbm/xgboost/catboost` when panel-like data and rows >= 80
- +0.03 for `boosting` when exogenous signal and rows >= 40

## Step 7: Model execution after routing

After candidate ranking, the top model is retrained on the full available history.

### How each family is run

- `naive`: repeat last observed value.
- `seasonal_naive`: repeat last seasonal pattern.
- `ridge` / `boosting` / optional boosters:
  - Build supervised lag/calendar features.
  - Train regressor on full history.
  - Forecast recursively step-by-step for horizon.
- `ets`: fit Exponential Smoothing and forecast horizon.
- `sarimax`: fit SARIMAX(1,1,1) with seasonal order when applicable, then forecast horizon.

If a model fails at fit/forecast time, the agent falls back to a safe naive-style output.

## Step 8: Prediction intervals

When supported in the selected path, intervals are estimated from residual spread and a z-value from `interval_level`.

- Regressor paths: residual std on training fit.
- ETS/SARIMAX paths: residual/differenced spread based intervals.
- Naive/seasonal naive: currently point forecasts only.

## Step 9: Panel (multi-series) behavior

If a series/group column exists:

1. Split by group.
2. Route and score each group independently.
3. Train and forecast per group.
4. Concatenate all forecasts.

Selected model names may differ across groups.

## Step 10: Output format

Output CSV includes:

- `timestamp`
- `forecast`
- `model_name`
- optional `series_column` (for panel data)
- optional `lower_bound`, `upper_bound` (when interval path is available)

## Practical summary in plain language

- The agent connects to PostgreSQL locally and lists the tables.
- It loads a small preview first, then reloads only the required columns into memory on the local machine.
- It guesses the key time-series columns and lets the user correct them.
- It compares a small shortlist of models with rolling time validation.
- It retrains the winner on all available history and exports forecast rows to CSV.

## Where this is implemented

- Database input helpers: `src/automated_forecasting/db_source.py`
- Routing, scoring, model execution: `src/automated_forecasting/pipeline.py`
- CLI and interactive prompts: `src/automated_forecasting/cli.py`
