# Automated Forecasting Agent

Privacy-first, local-only forecasting agent that connects to PostgreSQL on the local machine, inspects table metadata, lets you confirm the inferred time-series columns, shortlists candidate models with deterministic heuristics, validates them with time-aware splits, and exports forecasts to CSV.

The database connection stays on your machine. No raw database rows need to be sent to any third-party service.

## Client Database Safety

By default, the database workflow uses lightweight metadata and avoids exact `COUNT(*)` and full-table per-column statistics. Heavy profiling and exact row counts are opt-in with `--pro-max`.

For sensitive client systems, prefer a read-only replica, a client-approved Parquet export, or a bounded date range with `--start-date` and `--end-date`. The final forecasting query still has to aggregate the selected time/target columns, so run that only on an approved table/window and ideally on an indexed time column.

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

Database-first workflow on Windows:

```bash
run_forecast.bat --db-host localhost --db-port 5432 --db-name mydb --db-user myuser --db-password mypass --db-schema public
```

The agent will list tables, show metadata, infer the time/target/series columns, and then ask for the forecast horizon and output file path.

Direct CLI:

```bash
automated-forecasting-agent --db-host localhost --db-port 5432 --db-name mydb --db-user myuser --db-password mypass --db-schema public
```

Optional arguments:

- `--db-table`
- `--db-sslmode`
- `--csv` for compatibility with older workflows
- `--time-column`
- `--target-column`
- `--series-column`
- `--frequency`
- `--interval-level`

If you omit the database arguments, the CLI will prompt for them interactively.
