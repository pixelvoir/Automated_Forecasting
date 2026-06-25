from __future__ import annotations

import argparse
import itertools
import sys
import threading
import time
from contextlib import contextmanager
from getpass import getpass
from pathlib import Path

import pandas as pd

from .calendar_features import CalendarFeatureConfig
from .db_source import (
    PostgresConnectionConfig,
    connect,
    fetch_group_count_feature,
    fetch_indexes,
    fetch_table_metadata,
    fetch_table_stats,
    list_tables,
    load_grouped_aggregated_frame,
)
from .diagnostics import RoutingDiagnostics
from .pipeline import ForecastRequest, ForecastingAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Privacy-first automated forecasting agent")
    parser.add_argument("--csv", default=None, help="Optional input dataset CSV path")
    parser.add_argument("--parquet", default=None, help="Optional input dataset Parquet path")
    parser.add_argument("--db-host", default=None, help="PostgreSQL server host")
    parser.add_argument("--db-port", default=None, type=int, help="PostgreSQL server port")
    parser.add_argument("--db-name", default=None, help="PostgreSQL database name")
    parser.add_argument("--db-user", default=None, help="PostgreSQL username")
    parser.add_argument("--db-password", default=None, help="PostgreSQL password")
    parser.add_argument("--db-schema", default=None, help="PostgreSQL schema name")
    parser.add_argument("--db-table", default=None, help="PostgreSQL table name")
    parser.add_argument("--db-sslmode", default=None, help="Optional PostgreSQL sslmode")
    parser.add_argument("--pro-max", action="store_true", help="Opt in to heavier full-table SQL profiling and exact row counts")
    parser.add_argument("--start-date", default=None, help="Optional inclusive lower bound for database aggregation")
    parser.add_argument("--end-date", default=None, help="Optional exclusive upper bound for database aggregation")
    parser.add_argument("--horizon", default=None, type=int, help="Forecast horizon in steps")
    parser.add_argument("--output", default=None, help="Output forecast CSV path")
    parser.add_argument("--time-column", default=None, help="Explicit time column name")
    parser.add_argument("--target-column", default=None, help="Explicit target column name")
    parser.add_argument("--series-column", default=None, help="Optional series/group column")
    parser.add_argument("--frequency", default=None, help="Optional frequency override")
    parser.add_argument("--interval-level", default=0.9, type=float, help="Prediction interval level")
    parser.add_argument("--imputation-strategy", default=None, help="Override the missing-value handling strategy")
    parser.add_argument("--calendar-feature-set", default="basic", help="Calendar feature set: basic, holiday, seasonal, fiscal, all")
    parser.add_argument("--calendar-region", default=None, help="Optional holidays region code, for example IN or US")
    parser.add_argument("--drift-threshold", default=0.2, type=float, help="Drift threshold for retrain triggering")
    return parser


def _prompt(prompt_text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt_text}{suffix}: ").strip()
    return value or (default or "")


def _prompt_path(prompt_text: str, default: str | None = None) -> Path:
    return Path(_prompt(prompt_text, default))


def _prompt_optional(prompt_text: str, default: str | None = None) -> str | None:
    value = _prompt(prompt_text, default)
    return value or None


def _print_metadata(frame: pd.DataFrame, profile, total_rows: int | None = None, column_stats: dict[str, dict[str, object]] | None = None) -> None:
    print("\nDetected dataset metadata:")
    print(f"- total rows: {total_rows if total_rows is not None else profile.row_count}")
    print(f"- preview rows: {profile.row_count}")
    print(f"- columns: {len(frame.columns)}")
    print(f"- time column: {profile.time_column}")
    print(f"- target column: {profile.target_column}")
    print(f"- series column: {profile.series_column or 'none'}")
    print(f"- frequency: {profile.frequency or 'unknown'}")
    if hasattr(profile, "parse_success_rate") and profile.parse_success_rate is not None:
        print(f"- time parse success rate: {profile.parse_success_rate:.4f}")
    print("- feature columns:")
    for column in frame.columns:
        print(f"  - {column}: {frame[column].dtype}")
        if column_stats and column in column_stats:
            stats = column_stats[column]
            details = []
            for key in ("null_rate", "distinct_rows", "variance", "stddev", "parse_success_rows"):
                if key in stats and stats[key] is not None:
                    details.append(f"{key}={stats[key]}")
            if details:
                print(f"    stats: {', '.join(details)}")


def _progress(message: str) -> None:
    sys.stdout.write("\r")
    print(f"[progress] {message}", flush=True)


@contextmanager
def _activity_indicator(message: str):
    stop_event = threading.Event()

    def run() -> None:
        for marker in itertools.cycle("|/-\\"):
            if stop_event.is_set():
                break
            sys.stdout.write(f"\r[working] {message} {marker}")
            sys.stdout.flush()
            time.sleep(0.5)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
        sys.stdout.write(f"\r[done] {message}    \n")
        sys.stdout.flush()


def _choose_column(label: str, inferred: str | None, frame: pd.DataFrame, optional: bool = False) -> str | None:
    print(f"\nInferred {label}: {inferred or 'none'}")
    if optional:
        prompt_text = f"Enter a different {label} name, or press Enter to keep none"
    else:
        prompt_text = f"Enter a different {label} or press Enter to keep"
    override = _prompt_optional(prompt_text, inferred)
    if override and override not in frame.columns:
        raise ValueError(f"{label} '{override}' was not found in the loaded data.")
    return override or inferred


def _choose_stat_based_column(label: str, options: list[tuple[str, str]], default: str | None) -> str | None:
    print(f"\n{label} candidates:")
    for index, (name, summary) in enumerate(options, start=1):
        print(f"  {index}. {name}: {summary}")
    if default:
        print(f"Default: {default}")
    choice = _prompt_optional(f"Enter a different {label} or press Enter to keep", default)
    if choice and choice not in {name for name, _ in options}:
        raise ValueError(f"{label} '{choice}' was not found among the SQL-derived candidates.")
    return choice or default


def _infer_target_from_stats(stats, time_column: str | None) -> str | None:
    numeric_types = {"smallint", "integer", "bigint", "numeric", "real", "double precision", "decimal"}
    candidates = []
    for name, value in stats.columns.items():
        if name == time_column:
            continue
        if str(value.data_type).lower() in numeric_types:
            candidates.append((value.variance if value.variance is not None else -1.0, value.non_null_rows, name))
    return max(candidates, default=(0.0, 0, None))[2]


def _infer_column_from_names(columns: list[str], preferred: tuple[str, ...], excluded: set[str] | None = None) -> str | None:
    excluded = excluded or set()
    normalized = {column.lower(): column for column in columns if column not in excluded}
    for token in preferred:
        for lower, original in normalized.items():
            if token in lower:
                return original
    return None


def _build_postgres_config(args: argparse.Namespace) -> PostgresConnectionConfig:
    host = args.db_host or _prompt("PostgreSQL host", "localhost")
    port = args.db_port or int(_prompt("PostgreSQL port", "5432"))
    database = args.db_name or _prompt("PostgreSQL database name")
    user = args.db_user or _prompt("PostgreSQL username")
    password = args.db_password or getpass("PostgreSQL password: ")
    schema = args.db_schema or _prompt("PostgreSQL schema", "public")
    sslmode = args.db_sslmode or _prompt_optional("PostgreSQL sslmode (optional)")
    return PostgresConnectionConfig(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        schema=schema,
        sslmode=sslmode,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.csv or args.parquet:
        input_path = Path(args.csv or args.parquet)
        if args.parquet:
            print("Loading Parquet input...", flush=True)
            frame = pd.read_parquet(input_path)
        else:
            print("Loading CSV input...", flush=True)
            frame = pd.read_csv(input_path)

        source_label = str(input_path)
        preview_request = ForecastRequest(
            horizon=args.horizon or 1,
            output_path=Path(args.output or "_preview.csv"),
            input_frame=frame,
            time_column=args.time_column,
            target_column=args.target_column,
            series_column=args.series_column,
            frequency=args.frequency,
            interval_level=args.interval_level,
            source_label=source_label,
            progress_callback=_progress,
        )
        with _activity_indicator("inferring columns"):
            preview_profile = ForecastingAgent()._profile_dataset(frame, preview_request)
        _print_metadata(frame, preview_profile)
        time_column = _choose_column("time column", preview_profile.time_column, frame)
        target_column = _choose_column("target column", preview_profile.target_column, frame)
        series_column = _choose_column("series column", preview_profile.series_column, frame, optional=True)
        frequency = args.frequency or preview_profile.frequency
        horizon = args.horizon or int(_prompt("Forecast horizon in steps"))
        output_path = args.output or _prompt("Output CSV file path")
    else:
        print("Preparing PostgreSQL connection...", flush=True)
        config = _build_postgres_config(args)
        print(f"Connecting to {config.host}:{config.port}/{config.database}...", flush=True)
        connection = connect(config)
        try:
            print(f"Listing tables in schema '{config.schema}'...", flush=True)
            tables = list_tables(connection, schema=config.schema)
            if not tables:
                raise ValueError(f"No tables found in schema '{config.schema}'.")

            print("\nAvailable tables:")
            for table_name in tables:
                print(f"- {table_name}")

            table_name = args.db_table or _prompt("Table to forecast from")
            if table_name not in tables:
                raise ValueError(f"Table '{table_name}' was not found in schema '{config.schema}'.")

            print(f"Reading lightweight metadata for {config.schema}.{table_name}...", flush=True)
            with _activity_indicator("reading table metadata"):
                metadata = fetch_table_metadata(connection, table_name, schema=config.schema, exact_row_count=args.pro_max)
            stats = None
            if args.pro_max:
                print("Pro-max mode: computing SQL-pushdown statistics across the full table...", flush=True)
                with _activity_indicator("profiling full table columns"):
                    stats = fetch_table_stats(connection, table_name, schema=config.schema)
            else:
                print("Safe DB mode: skipping exact row count and full-table column statistics.", flush=True)
            print(f"\nSelected table: {metadata.schema}.{metadata.table_name}")
            row_label = "exact row count" if args.pro_max else "estimated row count"
            print(f"{row_label}: {metadata.row_count}")
            print("Columns:")
            for column in metadata.columns:
                print(f"- {column['name']} ({column['data_type']})")

            source_label = f"{config.database}.{config.schema}.{table_name}"

            column_names = [column["name"] for column in metadata.columns]
            candidate_time = args.time_column or (stats.time_column if stats else None) or _infer_column_from_names(column_names, ("date", "datetime", "timestamp", "time", "ds"))
            if candidate_time:
                index_info = fetch_indexes(connection, config.schema, table_name)
                warning = f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{config.schema}_{table_name}_{candidate_time} ON {config.schema}.{table_name} ({candidate_time});"
                if not index_info.has_index:
                    print(f"Warning: no index found for {candidate_time}. Suggested index: {warning}", flush=True)

            inferred_candidates = []
            if stats:
                for name, value in stats.columns.items():
                    summary = f"null_rate={value.null_rate:.4f}, distinct={value.distinct_rows or 0}"
                    if value.parse_success_rate is not None:
                        summary += f", parse_success={value.parse_success_rate:.4f}"
                    if value.variance is not None:
                        summary += f", variance={value.variance:.4f}"
                    inferred_candidates.append((name, summary))
            else:
                for column in metadata.columns:
                    inferred_candidates.append((column["name"], f"type={column['data_type']}"))
            print("Column candidates:", flush=True)
            for name, summary in inferred_candidates:
                print(f"- {name}: {summary}")

            inferred_time = _choose_stat_based_column("time column", inferred_candidates, candidate_time)
            inferred_target = _choose_stat_based_column(
                "target column",
                inferred_candidates,
                args.target_column
                or (_infer_target_from_stats(stats, candidate_time) if stats else None)
                or _infer_column_from_names(column_names, ("target", "y", "value", "sales", "demand", "load", "price", "volume"), {candidate_time} if candidate_time else set()),
            )
            inferred_series = _choose_column("series column", args.series_column or None, pd.DataFrame(columns=[name for name, _ in inferred_candidates]), optional=True)
            frequency = args.frequency or (stats.time_stats.frequency if stats and stats.time_stats else None)
            if not frequency:
                frequency = _prompt("Aggregation frequency (H, D, W, MS, QS, YS)", "D")
            print("Would you like to engineer a new feature by aggregating the count of one column grouped by another column?", flush=True)
            engineered_feature = None
            if _prompt_optional("Type yes to add a grouped count feature", "no") in {"yes", "y"}:
                group_by_column = _prompt("Group-by column")
                count_column = _prompt("Count column")
                engineered_feature = (f"count_{group_by_column}_{count_column}", count_column)
                print(f"Building SQL GROUP BY count feature for {group_by_column} / {count_column}...", flush=True)
                fetch_group_count_feature(connection, table_name, config.schema, group_by_column, count_column)

            time_column = inferred_time
            target_column = inferred_target
            series_column = inferred_series
            if args.frequency:
                frequency = args.frequency
            horizon = args.horizon or int(_prompt("Forecast horizon in steps"))
            output_path = args.output or _prompt("Output CSV file path")

            print("Loading SQL-aggregated modeling frame...", flush=True)
            with _activity_indicator("loading aggregated modeling frame"):
                frame = load_grouped_aggregated_frame(
                    connection,
                    table_name,
                    schema=config.schema,
                    time_column=time_column,
                    target_column=target_column,
                    frequency=frequency or (stats.time_stats.frequency if stats and stats.time_stats else "D"),
                    series_column=series_column,
                    engineered_feature=engineered_feature,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
            time_column = "timestamp"
            target_column = "forecast_target"
            series_column = "series" if series_column else None
            frequency = frequency or (stats.time_stats.frequency if stats and stats.time_stats else "D")
        finally:
            connection.close()

    print("Starting forecast run...", flush=True)

    request = ForecastRequest(
        horizon=horizon,
        output_path=Path(output_path),
        input_frame=frame,
        time_column=time_column,
        target_column=target_column,
        series_column=series_column,
        frequency=frequency,
        interval_level=args.interval_level,
        source_label=source_label,
        imputation_strategy=args.imputation_strategy,
        calendar_feature_config=CalendarFeatureConfig(feature_set=args.calendar_feature_set, region=args.calendar_region),
        drift_threshold=args.drift_threshold,
        progress_callback=_progress,
    )
    with _activity_indicator("running model selection and forecast"):
        result = ForecastingAgent().run(request)
    print(result.summary)
    print()
    print("Model selection explanation:")
    print(result.selection_explanation)


if __name__ == "__main__":
    main()
