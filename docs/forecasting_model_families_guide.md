# Forecasting Models: Comparison and Selection Guide

This document summarizes the major forecasting model families, how they work, where they perform well, and what data characteristics usually make them a good or poor fit.

## 1. What a forecasting model is solving

Forecasting models map past observations, optionally plus external variables, to future values. The right model depends mainly on:

- Forecast horizon: short, medium, or long.
- Data frequency: hourly, daily, weekly, monthly, irregular.
- Seasonality: none, simple, multiple, changing.
- Trend and structural breaks: stable, drifting, regime shifts.
- Number of series: single-series, many related series, hierarchical.
- Exogenous drivers: weather, promotions, pricing, holidays, events.
- Data size: small tabular series vs large multivariate datasets.
- Noise level and intermittency: smooth demand vs sparse zeros.
- Need for interpretability or uncertainty intervals.
- Compute budget and latency constraints.

## 2. Baselines

These are not usually the final choice, but they are essential for sanity checks.

### 2.1 Naive and seasonal naive
- **How it works:** Predict the last observed value, or the last observed seasonal value.
- **Best for:** Fast benchmark, very short horizons, strong seasonality.
- **Strengths:** Simple, robust, hard to beat on very noisy data.
- **Weaknesses:** Cannot learn new structure or regressors.
- **Compute:** Minimal.

### 2.2 Moving average / rolling average
- **How it works:** Predict an average of recent values.
- **Best for:** Smoothing noisy but stable series.
- **Strengths:** Simple and stable.
- **Weaknesses:** Lags turning points, weak on trend and seasonality.
- **Compute:** Minimal.

## 3. Classical statistical forecasting

These are often the first serious candidates for univariate and low-dimensional series.

### 3.1 AR, MA, ARMA, ARIMA
- **How it works:** Models autocorrelation using past values and past errors; ARIMA adds differencing to handle non-stationarity.
- **Best for:** Single series with stable patterns and limited exogenous input.
- **Strengths:** Strong baseline, interpretable, efficient, works well with small datasets.
- **Weaknesses:** Limited nonlinear learning, weak with multiple seasonalities or many regressors.
- **Compute:** Low.

### 3.2 SARIMA / SARIMAX
- **How it works:** ARIMA plus seasonal terms; SARIMAX adds exogenous variables.
- **Best for:** Seasonal series with calendar effects or external drivers.
- **Strengths:** Very strong on structured seasonal data; interpretable.
- **Weaknesses:** Parameter search can be expensive; struggles with complex nonlinearities.
- **Compute:** Low to moderate.

### 3.3 ETS / exponential smoothing / Holt-Winters
- **How it works:** Separately smooths level, trend, and seasonality using exponential decay.
- **Best for:** Smooth trend + seasonality, operational forecasting, demand data.
- **Strengths:** Strong, robust, fast, often excellent on business time series.
- **Weaknesses:** Limited flexibility for complex regressors and nonlinear patterns.
- **Compute:** Low.

### 3.4 Theta method
- **How it works:** Decomposes and combines trend-like components, often strong for simple series.
- **Best for:** Short-term business series with trend.
- **Strengths:** Simple and surprisingly competitive.
- **Weaknesses:** Less flexible than modern ML.
- **Compute:** Low.

### 3.5 Prophet-style models
- **How it works:** Additive trend + seasonality + holiday effects + changepoints.
- **Best for:** Business data with holidays, trend shifts, and moderate seasonality.
- **Strengths:** Easy to use, handles missing data and changepoints reasonably well.
- **Weaknesses:** Not always top accuracy; can be outperformed by tuned statistical or ML models.
- **Compute:** Low to moderate.

### 3.6 State-space models / Kalman filter models
- **How it works:** Latent states evolve over time; observations are noisy emissions.
- **Best for:** Noisy series, missing values, dynamic systems.
- **Strengths:** Probabilistic, handles irregularities well.
- **Weaknesses:** Model specification can be complex.
- **Compute:** Low to moderate.

### 3.7 Dynamic harmonic regression
- **How it works:** Fourier terms for seasonality combined with ARIMA-style residual modeling.
- **Best for:** Multiple or long seasonality periods.
- **Strengths:** Efficient for complex seasonality.
- **Weaknesses:** Requires good seasonal period selection.
- **Compute:** Low to moderate.

## 4. Multivariate and feature-based machine learning

These methods convert forecasting into supervised learning by using lag features, rolling statistics, and exogenous variables.

### 4.1 Linear regression / ridge / lasso / elastic net
- **How it works:** Predict future values from engineered lag and calendar features.
- **Best for:** Small to medium datasets with good feature engineering.
- **Strengths:** Fast, interpretable, often strong with good features.
- **Weaknesses:** Can miss nonlinear dynamics.
- **Compute:** Low.

### 4.2 Support vector regression
- **How it works:** Margin-based regression with kernels.
- **Best for:** Smaller datasets with nonlinear structure.
- **Strengths:** Can model nonlinearities.
- **Weaknesses:** Scales poorly on large datasets; feature scaling is important.
- **Compute:** Moderate to high.

### 4.3 k-nearest neighbors regression
- **How it works:** Forecast using similar historical windows.
- **Best for:** Small datasets with repeating patterns.
- **Strengths:** Simple local pattern matching.
- **Weaknesses:** Weak extrapolation, sensitive to feature design.
- **Compute:** Moderate at prediction time.

### 4.4 Random forest
- **How it works:** Ensemble of decision trees over lagged and exogenous features.
- **Best for:** Nonlinear structured data with enough engineered features.
- **Strengths:** Robust, good baseline for tabular forecasting.
- **Weaknesses:** Poor at long-horizon recursive forecasting if not designed carefully.
- **Compute:** Moderate.

### 4.5 Gradient boosting: XGBoost, LightGBM, CatBoost, HistGradientBoosting
- **How it works:** Sequential tree boosting over supervised forecasting features.
- **Best for:** Many practical forecasting tasks with rich features.
- **Strengths:** Often among the best tabular forecasting methods; strong accuracy; handles nonlinear effects.
- **Weaknesses:** Needs careful feature creation and validation; recursive rollout can accumulate error.
- **Compute:** Moderate.

### 4.6 Extra trees / boosted trees variants
- **How it works:** Randomized tree ensembles.
- **Best for:** Fast strong baselines on structured data.
- **Strengths:** Robust, easy to use.
- **Weaknesses:** Same feature-engineering dependency as other tree models.
- **Compute:** Moderate.

### 4.7 Generalized additive models / GAMs
- **How it works:** Additive smooth effects of time, seasonality, and covariates.
- **Best for:** Interpretability with moderate nonlinearity.
- **Strengths:** Good balance of flexibility and explainability.
- **Weaknesses:** Not ideal for highly complex dynamics.
- **Compute:** Low to moderate.

## 5. Deep learning forecasting

These models are useful when there is enough data, many related series, complex nonlinear patterns, or the need for representation learning.

### 5.1 MLP / feedforward neural networks
- **How it works:** Uses lagged and engineered inputs to learn nonlinear mappings.
- **Best for:** Feature-rich forecasting and medium datasets.
- **Strengths:** Simple deep baseline.
- **Weaknesses:** No built-in temporal structure.
- **Compute:** Moderate.

### 5.2 RNNs: vanilla RNN, LSTM, GRU
- **How it works:** Sequence models that process one time step at a time.
- **Best for:** Sequential patterns and multivariate series.
- **Strengths:** Learn temporal dependencies without heavy manual lag engineering.
- **Weaknesses:** Harder to train than tree models; can underperform on tabular business forecasting.
- **Compute:** Moderate to high.

### 5.3 CNN / temporal convolutional networks / dilated convolutions
- **How it works:** Convolutions over time capture local and longer-range patterns through dilation.
- **Best for:** Large datasets and multivariate signals.
- **Strengths:** Efficient, parallelizable, strong temporal feature extraction.
- **Weaknesses:** Needs more data and tuning.
- **Compute:** Moderate to high.

### 5.4 N-BEATS / N-HiTS
- **How it works:** Deep residual architectures designed specifically for forecasting.
- **Best for:** Strong univariate or multivariate forecasting with ample data.
- **Strengths:** Often very competitive on benchmarks; strong direct forecasting.
- **Weaknesses:** Can be heavy and harder to explain.
- **Compute:** Moderate to high.

### 5.5 DeepAR / probabilistic autoregressive RNNs
- **How it works:** Trains on many related series with probabilistic outputs.
- **Best for:** Many similar series, retail, inventory, demand forecasting.
- **Strengths:** Learns across-series patterns; outputs uncertainty.
- **Weaknesses:** Less ideal for a single short series.
- **Compute:** Moderate to high.

### 5.6 Seq2Seq / encoder-decoder models
- **How it works:** Encodes past context and decodes future steps in one shot.
- **Best for:** Multi-step forecasting with complex dynamics.
- **Strengths:** Direct multi-horizon forecasting.
- **Weaknesses:** More complex to train and tune.
- **Compute:** High.

### 5.7 Temporal Fusion Transformer-style hybrids
- **How it works:** Combines attention, gating, static features, known future inputs, and sequence modeling.
- **Best for:** Rich multivariate forecasting with covariates and multiple horizons.
- **Strengths:** Strong on complex business data; can use static and dynamic features.
- **Weaknesses:** Requires more data, tuning, and compute.
- **Compute:** High.

## 6. Transformer-based forecasting models

Transformers are useful when datasets are large, multivariate, or have long-range dependencies.

### 6.1 Vanilla time-series transformers
- **How it works:** Attention across time steps for long-range dependency modeling.
- **Best for:** Long context windows and complex series.
- **Strengths:** Captures global patterns.
- **Weaknesses:** Can be expensive and data-hungry.
- **Compute:** High.

### 6.2 Informer / Autoformer / FEDformer / PatchTST / iTransformer / TimesNet and related families
- **How it works:** Transformer variants optimized for long horizons, seasonality, decomposition, patching, or efficient attention.
- **Best for:** Long-horizon forecasting, large multivariate datasets, long context series.
- **Strengths:** Often strong on benchmark tasks with sufficient data.
- **Weaknesses:** Typically more complex to configure; may not beat simpler models on small business datasets.
- **Compute:** High.

### 6.3 Foundation-style forecasting models
- **How it works:** Large pretrained models adapted to new series.
- **Best for:** When transfer learning is available and fast adaptation is needed.
- **Strengths:** Can work well with limited target data if the pretraining distribution matches.
- **Weaknesses:** Availability, cost, and reproducibility vary.
- **Compute:** High.

## 7. Specialized forecasting families

### 7.1 Intermittent demand models
- **Examples:** Croston, SBA, TSB.
- **Best for:** Sparse demand with many zeros.
- **Strengths:** Designed for intermittent series.
- **Weaknesses:** Not suitable for smooth continuous signals.
- **Compute:** Low.

### 7.2 Hierarchical forecasting
- **How it works:** Forecasts at multiple aggregation levels and reconciles them.
- **Best for:** Product-store-region-family hierarchies.
- **Strengths:** Consistent forecasts across levels.
- **Weaknesses:** More complex pipeline.
- **Compute:** Moderate.

### 7.3 Multiseries global models
- **How it works:** One model trained across many related series.
- **Best for:** Many items, customers, locations, sensors.
- **Strengths:** Shares information across series.
- **Weaknesses:** Needs consistent schema and enough series.
- **Compute:** Moderate to high.

### 7.4 Probabilistic forecasting
- **How it works:** Predicts distributions or quantiles instead of single values.
- **Examples:** Quantile regression, distributional neural nets, Bayesian models.
- **Best for:** Risk-sensitive planning and interval forecasting.
- **Strengths:** Uncertainty-aware.
- **Weaknesses:** More evaluation complexity.
- **Compute:** Low to high depending on model.

## 8. Which model family is usually best

This is a practical ranking, not a universal rule.

### Small data, one series, strong seasonality
- Best candidates: seasonal naive, ETS, SARIMA/SARIMAX, Prophet.
- Why: These methods are data-efficient and stable.

### Small to medium data, rich covariates, tabular features
- Best candidates: ridge/lasso, random forest, XGBoost, LightGBM, CatBoost.
- Why: Feature-based ML often wins when the target depends on external drivers.

### Many related series
- Best candidates: global tree models, DeepAR, N-BEATS/N-HiTS, TFT, transformer variants.
- Why: Shared learning across series improves accuracy.

### Long horizon or long memory dependence
- Best candidates: transformer variants, TCNs, N-HiTS, TFT.
- Why: Better at long context and multi-step structure.

### Sparse or intermittent demand
- Best candidates: Croston family, specialized probabilistic models, zero-inflated approaches.
- Why: Generic models often overfit zeros or smooth away bursts.

### Need interpretability and operational simplicity
- Best candidates: ETS, SARIMA/SARIMAX, GAMs, linear models, Prophet.
- Why: Easier to explain to stakeholders.

### Best pure accuracy on tabular business data
- Best candidates: gradient boosting on engineered lag and calendar features.
- Why: In many practical cases, tuned boosting models are the strongest and cheapest accuracy winners.

## 9. Selection criteria the agent should use

The agent should compare candidate models using the following characteristics:

- Sample size.
- Seasonality strength and number of seasonal periods.
- Trend stability or structural breaks.
- Number and quality of exogenous variables.
- Multi-series vs single-series setup.
- Forecast horizon length.
- Missing data rate.
- Intermittency or zero inflation.
- Need for prediction intervals.
- Training and inference budget.
- Interpretability requirements.

## 10. Practical shortlist strategy

To keep compute bounded, the agent should not test everything.

### Stage 1: Cheap baselines
- Seasonal naive.
- ETS or simple exponential smoothing.
- SARIMA/SARIMAX if the series is low-dimensional and seasonal.
- A simple gradient boosting model if useful lags and covariates exist.

### Stage 2: Strong candidates based on diagnostics
- LightGBM/XGBoost/CatBoost for structured covariate-rich problems.
- N-BEATS or N-HiTS for strong deep forecasting on enough data.
- TFT or transformer variants if the dataset is large and multivariate.

### Stage 3: Only if justified by data
- DeepAR for many related series.
- Specialized intermittent-demand methods for sparse data.
- Hierarchical reconciliation if the data has a hierarchy.

## 11. Summary

A good automated forecasting system should usually start with statistical baselines, then move to feature-based machine learning, and only then use deep learning or transformers when the data volume and structure justify the extra complexity. In many real business settings, the most efficient and accurate solution is often a well-engineered gradient boosting model plus strong time-series features, while SARIMA/ETS remain excellent for smaller seasonal series and transformer-style models are most useful for larger multivariate problems.
