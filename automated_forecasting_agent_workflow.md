# Automated Forecasting Agent Workflow Plan

## Goal

Build an automated forecasting agent that takes a dataset in CSV form, performs preprocessing and exploratory analysis, selects a small number of suitable forecasting models, tunes them within a bounded compute budget, generates forecasts for the requested horizon, and returns the final prediction output as a CSV file.

## Privacy-first constraint

The agent must operate under a privacy-first rule:

- Raw customer data must never be sent to an external LLM or third-party service.
- Any optional language-model component may only receive sanitized metadata, schema summaries, and non-sensitive diagnostics.
- If privacy cannot be guaranteed, the agent must fall back to a fully local, deterministic workflow with no LLM involvement.
- The forecasting models themselves should run inside the controlled environment where the data already lives.

## Core idea

The agent should behave like a compact decision system:

1. Inspect the dataset structure.
2. Identify the time column, target column, optional series identifier, and optional exogenous features.
3. Perform data quality checks and time-series diagnostics.
4. Narrow the model family to a small shortlist.
5. Train and validate the shortlisted models with time-aware evaluation.
6. Choose the best model or small ensemble.
7. Produce forecast rows in CSV format.

## Inputs the agent should accept

Required:

- CSV file path or uploaded CSV.
- Forecast horizon, expressed as number of future time steps.
- Time frequency if it cannot be inferred reliably.

Usually required or strongly recommended:

- Target column name if the dataset has multiple numeric columns.
- Time column name if the schema is ambiguous.
- Series/group column name for panel or multi-entity forecasting.
- Forecast start date if the future index must be explicit.

Optional but useful:

- Exogenous feature columns.
- Confidence interval level, such as 80 percent or 95 percent.
- Evaluation metric preference, such as MAE, RMSE, MAPE, sMAPE, or pinball loss.
- Maximum training time or model budget.
- Whether the output should include intervals or only point forecasts.
- Whether the agent should forecast one target or multiple targets.

## Workflow

### Step 1: Schema detection

The agent should inspect the dataset and infer:

- Which column is the timestamp.
- Which column is the target to forecast.
- Whether the data is one series or multiple related series.
- Whether there are categorical or numeric exogenous variables.
- Whether the dataset is sorted and regularly spaced.

If the time or target field is ambiguous, the agent should use heuristics and, if still uncertain, choose the most likely candidate while warning the user.

### Step 2: Data validation

The agent should check:

- Missing values.
- Duplicate timestamps.
- Non-monotonic time order.
- Gaps in the time index.
- Constant or near-constant target values.
- Outliers and extreme spikes.
- Zero inflation or intermittent demand.
- Very short history relative to the requested horizon.

If needed, the agent should normalize the time index, de-duplicate records, and align to the inferred frequency.

### Step 3: Exploratory data analysis

The agent should compute and summarize:

- Trend and seasonality strength.
- Autocorrelation and partial autocorrelation.
- Cross-correlation between target and candidate exogenous variables.
- Rolling mean and rolling variance.
- Change points or structural breaks.
- Distribution shape and skewness.
- Intermittency measures such as zero ratio and average demand interval.
- Missingness patterns.
- Correlation between multiple target series if relevant.

This EDA should directly influence model selection rather than only being descriptive.

### Step 4: Feature engineering

The agent should create features such as:

- Lags of the target.
- Rolling statistics: mean, median, min, max, std, quantiles.
- Calendar features: day of week, month, quarter, weekend, holiday, hour, week of year.
- Fourier or seasonal basis terms when useful.
- Differencing features if stationarity is weak.
- Exogenous lags and lead-safe known future variables.
- Static series metadata for panel forecasting.

The agent must avoid leakage by only using information that would be available at prediction time.

### Step 5: Model narrowing logic

The agent should not test many models. It should narrow to a small shortlist, usually 2 to 4 models, based on the data diagnosis.

A practical selection rule is:

- Use naive or seasonal naive as a baseline always.
- Use ETS or SARIMA/SARIMAX when the series is short, seasonal, and relatively clean.
- Use gradient boosting when engineered lag and calendar features are informative.
- Use a deep model such as N-BEATS, N-HiTS, DeepAR, or a transformer only when there is enough data to justify it.
- Use intermittent-demand methods when the target contains many zeros.
- Use hierarchical reconciliation only when the data is explicitly hierarchical.

### Step 5a: No-LLM fallback mode

When the system runs in no-LLM mode, all routing decisions must be deterministic and based on local metadata and diagnostics.

The fallback router should use rules such as:

- If the series is short, stable, and seasonal, prefer ETS or SARIMA/SARIMAX.
- If the dataset has useful exogenous features and enough rows, prefer gradient boosting.
- If there are many related series and sufficient history, consider a global forecasting model.
- If the target is sparse or intermittent, prefer intermittent-demand methods.
- If the data is ambiguous or weakly structured, keep the shortlist small and include a baseline such as seasonal naive.

The no-LLM path should still perform:

- schema detection,
- preprocessing,
- feature engineering,
- diagnostics,
- shortlist generation,
- time-aware validation,
- bounded hyperparameter tuning,
- final forecasting,
- CSV export.

It should not require any external model to interpret the dataset or choose the final forecasting candidate.

Model choice should depend on:

- History length.
- Seasonality strength.
- Number of series.
- Exogenous feature richness.
- Horizon length.
- Compute budget.
- Desired uncertainty output.

### Step 6: Validation strategy

The agent should use time-aware validation only.

Recommended approaches:

- Rolling-origin backtesting.
- Expanding window validation.
- Sliding window validation when the series is non-stationary.

The agent should avoid random shuffling because it breaks temporal order.

### Step 7: Hyperparameter tuning

The agent should use bounded search strategies:

- Small grid search for statistical models.
- Random search or Bayesian optimization for tree and deep models.
- Early stopping for boosting and neural models.
- A strict cap on training trials.

Tuning should focus only on a small number of impactful parameters.

Examples:

- SARIMA: p, d, q, P, D, Q, seasonal period.
- ETS: additive or multiplicative trend and seasonality.
- Boosting: lag window, tree depth, learning rate, number of estimators, subsampling.
- Deep models: context length, hidden size, dropout, learning rate, batch size.
- Transformers: patch size or sequence length, attention heads, model width, dropout.

### Step 8: Model selection

The final choice should be based on a combination of:

- Validation score.
- Prediction stability across folds.
- Error distribution over time.
- Forecast calibration if intervals are produced.
- Compute cost.
- Interpretability requirements.

If two models are close, the agent may:

- Choose the simpler one if accuracy is similar.
- Build a small ensemble if the gains are meaningful and cheap.

### Step 9: Final training and forecasting

After selection, the agent should retrain the chosen model on the full history and generate forecasts for the full requested horizon.

The output should include:

- Timestamp or future index.
- Forecast value.
- Lower and upper bounds if intervals are requested.
- Series identifier if multiple series are forecasted.
- Model name used.
- Optional confidence score or quantile columns.

### Step 10: Export

The final output should be saved as CSV with a clear schema.

Recommended columns:

- `timestamp`
- `series_id` if relevant
- `forecast`
- `lower_bound` if relevant
- `upper_bound` if relevant
- `model_name`

If the user requests multiple horizons or multiple series, the file should be in long format rather than wide format unless explicitly requested otherwise.

## Practical model selection rules

### If the dataset is small and seasonal
- Start with seasonal naive, ETS, and SARIMA/SARIMAX.
- Use a simple boosted model only if there are useful covariates.

### If the dataset is tabular with strong exogenous variables
- Prioritize gradient boosting.
- Add linear or regularized regression as a baseline.
- Add SARIMAX only if the time structure is strong and interpretable.

### If there are many related series
- Use a global model such as DeepAR, N-BEATS, N-HiTS, or a transformer-based model if enough data exists.
- Consider tree-based global forecasting if feature engineering is practical.

### If the target is intermittent
- Use Croston-style methods or a model that explicitly handles sparse zeros.

### If the horizon is long
- Prefer direct multi-step approaches or models built for multi-horizon forecasting.
- Avoid fully recursive strategies unless the dataset is small and simple.

## Cheap discriminating checks before expensive training

Before training expensive candidates, the agent should use these low-cost checks:

- Compare naive vs seasonal naive.
- Measure seasonality and autocorrelation strength.
- Check if exogenous variables have meaningful cross-correlation with the target.
- Estimate series length relative to horizon.
- Check the number of related series.
- Detect intermittency.
- Check for trend breaks or regime shifts.

These checks should decide whether deep learning is even worth trying.

## Recommended system architecture

### Privacy and orchestration layer
- Enforces local-only execution rules.
- Sanitizes any optional metadata before it reaches a language model.
- Switches between no-LLM mode and optional local assist mode.
- Blocks external API calls for raw data processing.

### Orchestration layer
- Reads the CSV.
- Detects schema.
- Runs diagnostics.
- Chooses candidate models.
- Triggers training and evaluation.
- Writes outputs.

### Feature and diagnostics layer
- Handles preprocessing.
- Generates lags and calendar features.
- Computes correlations and time-series statistics.

### Model registry
- Stores available model wrappers.
- Provides a uniform interface for fit, predict, and forecast interval generation.

### Evaluation layer
- Runs backtests.
- Computes metrics.
- Tracks runtime and memory.

### Output layer
- Produces forecast CSV.
- Optionally writes a report with diagnostics and model selection rationale.

## Suggested output behavior

The agent should return:

- A forecast CSV.
- A short summary of which models were tested.
- The best model selected.
- The main reason for selection.
- The validation metric values.
- Any warnings about data quality or uncertainty.

## Failure handling

The agent should degrade gracefully when data quality is poor.

Examples:

- If the series is too short, use simple baselines and warn the user.
- If timestamps are irregular, resample or reject with a clear message.
- If exogenous features are unavailable, fall back to pure time-series models.
- If deep models fail or are too slow, keep the strongest statistical or boosting model.

## Practical recommendation

For an efficient first version, the agent should follow this default shortlist order:

1. Seasonal naive baseline.
2. ETS or SARIMA/SARIMAX if seasonal structure is clear.
3. Gradient boosting with lag and calendar features.
4. One deep model only if data volume justifies it.

This gives strong practical performance without testing too many expensive models.
