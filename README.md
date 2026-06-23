# Automated Forecasting Agent

Privacy-first, local-only forecasting agent that uses deterministic heuristics to infer the dataset shape, shortlist candidate models, validate them with time-aware splits, and export forecasts to CSV.

## Install

Use the installed Python interpreter directly if `python` is not on your PATH yet:

```bash
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m pip install -e .
```

If Python is already on PATH, this also works:

```bash
pip install -e .
```

## Dependencies

Runtime dependencies are:

- numpy
- pandas
- scikit-learn
- statsmodels
- lightgbm
- xgboost
- catboost

The agent is designed to run locally with no external LLM calls.

Boosted-tree models are considered automatically when the dataset is large enough and the schema is feature-rich enough to justify them.

## Run

Windows launcher:

```bash
run_forecast.bat --csv data.csv --horizon 30 --output forecast.csv
```

Direct CLI:

```bash
automated-forcasting-agent --csv data.csv --horizon 30 --output forecast.csve
```

Optional arguments:

- `--time-column`
- `--target-column`
- `--series-column`
- `--frequency`
- `--interval-level`
