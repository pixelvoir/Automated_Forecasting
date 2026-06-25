from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


@dataclass
class PostgresConnectionConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    schema: str = "public"
    sslmode: str | None = None


@dataclass
class TableMetadata:
    schema: str
    table_name: str
    row_count: int
    columns: list[dict[str, Any]]


@dataclass
class ColumnStats:
    name: str
    data_type: str
    total_rows: int
    non_null_rows: int
    null_rows: int
    null_rate: float
    distinct_rows: int | None
    min_value: Any | None
    max_value: Any | None
    variance: float | None = None
    stddev: float | None = None
    parse_success_rows: int | None = None
    parse_failure_rows: int | None = None

    @property
    def parse_success_rate(self) -> float | None:
        if self.parse_success_rows is None:
            return None
        return float(self.parse_success_rows) / max(self.total_rows, 1)


@dataclass
class TimeStats:
    frequency: str | None
    median_delta_seconds: float | None
    min_timestamp: Any | None
    max_timestamp: Any | None
    parse_success_rate: float


@dataclass
class TableStats:
    row_count: int
    time_column: str | None
    columns: dict[str, ColumnStats]
    time_stats: TimeStats | None
    preview_rows: int = 0


@dataclass
class IndexInfo:
    has_index: bool
    matching_indexes: list[str]


def connect(config: PostgresConnectionConfig):
    try:
        import psycopg2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "psycopg2-binary is required for PostgreSQL connections. Install dependencies with 'pip install -e .' or 'pip install psycopg2-binary'."
        ) from exc

    connection_kwargs: dict[str, Any] = {
        "host": config.host,
        "port": config.port,
        "dbname": config.database,
        "user": config.user,
        "password": config.password,
    }
    if config.sslmode:
        connection_kwargs["sslmode"] = config.sslmode
    return psycopg2.connect(**connection_kwargs)


def list_tables(connection, schema: str = "public") -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """,
            (schema,),
        )
        return [row[0] for row in cursor.fetchall()]


def fetch_table_metadata(connection, table_name: str, schema: str = "public", exact_row_count: bool = False) -> TableMetadata:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name, data_type, is_nullable, character_maximum_length,
                   numeric_precision, numeric_scale
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """,
            (schema, table_name),
        )
        columns = [
            {
                "name": row[0],
                "data_type": row[1],
                "is_nullable": row[2],
                "character_maximum_length": row[3],
                "numeric_precision": row[4],
                "numeric_scale": row[5],
            }
            for row in cursor.fetchall()
        ]

        if exact_row_count:
            from psycopg2 import sql

            cursor.execute(sql.SQL("SELECT COUNT(*) FROM {}.{}").format(sql.Identifier(schema), sql.Identifier(table_name)))
            row_count = int(cursor.fetchone()[0])
        else:
            cursor.execute(
                """
                SELECT COALESCE(c.reltuples::bigint, 0)
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = %s AND c.relname = %s
                """,
                (schema, table_name),
            )
            row_count = int(cursor.fetchone()[0] or 0)

    return TableMetadata(schema=schema, table_name=table_name, row_count=row_count, columns=columns)


def _qualified_identifier(schema: str, table_name: str):
    from psycopg2 import sql

    return sql.Identifier(schema), sql.Identifier(table_name)


def fetch_table_stats(connection, table_name: str, schema: str = "public") -> TableStats:
    metadata = fetch_table_metadata(connection, table_name, schema=schema, exact_row_count=True)
    column_stats: dict[str, ColumnStats] = {}
    time_candidates = []
    with connection.cursor() as cursor:
        for column in metadata.columns:
            name = column["name"]
            data_type = str(column["data_type"])
            stats = _fetch_column_stats(cursor, schema, table_name, name, data_type, metadata.row_count)
            column_stats[name] = stats
            if stats.parse_success_rate is not None and stats.parse_success_rate >= 0.8:
                time_candidates.append((stats.parse_success_rate, stats.distinct_rows or 0, name))

        time_column = max(time_candidates, default=(0.0, 0, None))[2]
        time_stats = _fetch_time_stats(cursor, schema, table_name, time_column) if time_column else None

    return TableStats(
        row_count=metadata.row_count,
        time_column=time_column,
        columns=column_stats,
        time_stats=time_stats,
    )


def _fetch_column_stats(cursor, schema: str, table_name: str, column_name: str, data_type: str, row_count: int) -> ColumnStats:
    from psycopg2 import sql

    column_sql = sql.Identifier(column_name)
    table_sql = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table_name))
    numeric_types = {"smallint", "integer", "bigint", "numeric", "real", "double precision", "decimal"}
    datetime_types = {"timestamp without time zone", "timestamp with time zone", "date", "time without time zone", "time with time zone"}

    if data_type.lower() in numeric_types:
        cursor.execute(
            sql.SQL(
                """
                SELECT
                    COUNT(*) AS total_rows,
                    COUNT({col}) AS non_null_rows,
                    COUNT(*) - COUNT({col}) AS null_rows,
                    COUNT(DISTINCT {col}) AS distinct_rows,
                    MIN({col}) AS min_value,
                    MAX({col}) AS max_value,
                    VAR_SAMP({col}) AS variance,
                    STDDEV_SAMP({col}) AS stddev
                FROM {table}
                """
            ).format(col=column_sql, table=table_sql)
        )
        total_rows, non_null_rows, null_rows, distinct_rows, min_value, max_value, variance, stddev = cursor.fetchone()
        return ColumnStats(
            name=column_name,
            data_type=data_type,
            total_rows=int(total_rows or 0),
            non_null_rows=int(non_null_rows or 0),
            null_rows=int(null_rows or 0),
            null_rate=(float(null_rows or 0) / max(row_count, 1)),
            distinct_rows=int(distinct_rows or 0),
            min_value=min_value,
            max_value=max_value,
            variance=float(variance) if variance is not None else None,
            stddev=float(stddev) if stddev is not None else None,
        )

    if data_type.lower() in datetime_types:
        cursor.execute(
            sql.SQL(
                """
                SELECT
                    COUNT(*) AS total_rows,
                    COUNT({col}) AS non_null_rows,
                    COUNT(*) - COUNT({col}) AS null_rows,
                    COUNT(DISTINCT {col}) AS distinct_rows,
                    MIN({col}) AS min_value,
                    MAX({col}) AS max_value
                FROM {table}
                """
            ).format(col=column_sql, table=table_sql)
        )
        total_rows, non_null_rows, null_rows, distinct_rows, min_value, max_value = cursor.fetchone()
        return ColumnStats(
            name=column_name,
            data_type=data_type,
            total_rows=int(total_rows or 0),
            non_null_rows=int(non_null_rows or 0),
            null_rows=int(null_rows or 0),
            null_rate=(float(null_rows or 0) / max(row_count, 1)),
            distinct_rows=int(distinct_rows or 0),
            min_value=min_value,
            max_value=max_value,
        )

    cursor.execute(
        sql.SQL(
            """
            SELECT
                COUNT(*) AS total_rows,
                COUNT({col}) AS non_null_rows,
                COUNT(*) - COUNT({col}) AS null_rows,
                COUNT(DISTINCT {col}) AS distinct_rows,
                MIN({col}::text) AS min_value,
                MAX({col}::text) AS max_value,
                SUM(CASE WHEN {col}::text ~ '^[0-9]{{4}}-[0-9]{{1,2}}-[0-9]{{1,2}}|^[0-9]{{1,2}}/[0-9]{{1,2}}/[0-9]{{2,4}}' THEN 1 ELSE 0 END) AS parse_success_rows,
                SUM(CASE WHEN {col} IS NOT NULL AND NOT ({col}::text ~ '^[0-9]{{4}}-[0-9]{{1,2}}-[0-9]{{1,2}}|^[0-9]{{1,2}}/[0-9]{{1,2}}/[0-9]{{2,4}}') THEN 1 ELSE 0 END) AS parse_failure_rows
            FROM {table}
            """
        ).format(col=column_sql, table=table_sql)
    )
    total_rows, non_null_rows, null_rows, distinct_rows, min_value, max_value, parse_success_rows, parse_failure_rows = cursor.fetchone()
    parse_success_rows = int(parse_success_rows or 0)
    parse_failure_rows = int(parse_failure_rows or 0)
    non_null_rows = int(non_null_rows or 0)
    return ColumnStats(
        name=column_name,
        data_type=data_type,
        total_rows=int(total_rows or 0),
        non_null_rows=non_null_rows,
        null_rows=int(null_rows or 0),
        null_rate=(float(null_rows or 0) / max(row_count, 1)),
        distinct_rows=int(distinct_rows or 0),
        min_value=min_value,
        max_value=max_value,
        parse_success_rows=parse_success_rows,
        parse_failure_rows=parse_failure_rows,
    )


def _fetch_time_stats(cursor, schema: str, table_name: str, time_column: str | None) -> TimeStats | None:
    if not time_column:
        return None
    from psycopg2 import sql

    table_sql = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table_name))
    column_sql = sql.Identifier(time_column)
    cursor.execute(
        sql.SQL(
            """
            WITH ordered AS (
                SELECT {col}::timestamp AS ts
                FROM {table}
                WHERE {col} IS NOT NULL
                ORDER BY {col}::timestamp
            ),
            deltas AS (
                SELECT EXTRACT(EPOCH FROM ts - LAG(ts) OVER (ORDER BY ts)) AS delta_seconds
                FROM ordered
            )
            SELECT
                (SELECT MIN(ts) FROM ordered) AS min_timestamp,
                (SELECT MAX(ts) FROM ordered) AS max_timestamp,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY delta_seconds) AS median_delta_seconds,
                1.0 AS parse_success_rate
            FROM deltas
            WHERE delta_seconds IS NOT NULL
            """
        ).format(col=column_sql, table=table_sql)
    )
    min_timestamp, max_timestamp, median_delta_seconds, parse_success_rate = cursor.fetchone()
    frequency = _frequency_from_delta(median_delta_seconds)
    return TimeStats(
        frequency=frequency,
        median_delta_seconds=float(median_delta_seconds) if median_delta_seconds is not None else None,
        min_timestamp=min_timestamp,
        max_timestamp=max_timestamp,
        parse_success_rate=float(parse_success_rate) if parse_success_rate is not None else 0.0,
    )


def _frequency_from_delta(delta_seconds: float | None) -> str | None:
    if delta_seconds is None:
        return None
    if delta_seconds <= 7200:
        return "H"
    if delta_seconds <= 172800:
        return "D"
    if delta_seconds <= 1209600:
        return "W"
    if delta_seconds <= 3456000:
        return "MS"
    if delta_seconds <= 9072000:
        return "QS"
    return "YS"


def fetch_preview_sample(connection, table_name: str, schema: str = "public", limit: int = 1000) -> pd.DataFrame:
    from psycopg2 import sql

    query = sql.SQL("SELECT * FROM {}.{} TABLESAMPLE SYSTEM (%s)").format(sql.Identifier(schema), sql.Identifier(table_name))
    try:
        return pd.read_sql_query(query.as_string(connection), connection, params=(min(max(limit / 1000, 0.1), 100.0),))
    except Exception:
        fallback = sql.SQL("SELECT * FROM {}.{} LIMIT %s").format(sql.Identifier(schema), sql.Identifier(table_name))
        return pd.read_sql_query(fallback.as_string(connection), connection, params=(limit,))

    query = sql.SQL("SELECT * FROM {}.{} LIMIT %s").format(sql.Identifier(schema), sql.Identifier(table_name))
    return pd.read_sql_query(query.as_string(connection), connection, params=(limit,))


def load_table(connection, table_name: str, schema: str = "public", columns: list[str] | None = None) -> pd.DataFrame:
    from psycopg2 import sql

    if columns:
        column_list = sql.SQL(", ").join(sql.Identifier(column) for column in columns)
        query = sql.SQL("SELECT {} FROM {}.{}").format(column_list, sql.Identifier(schema), sql.Identifier(table_name))
    else:
        query = sql.SQL("SELECT * FROM {}.{}").format(sql.Identifier(schema), sql.Identifier(table_name))
    return pd.read_sql_query(query.as_string(connection), connection)


def load_grouped_aggregated_frame(
    connection,
    table_name: str,
    schema: str,
    time_column: str,
    target_column: str,
    frequency: str,
    series_column: str | None = None,
    engineered_feature: tuple[str, str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    from psycopg2 import sql

    postgres_frequency = _postgres_date_trunc_frequency(frequency)
    bucket_expr = sql.SQL("date_trunc(%s, {}::timestamp)").format(sql.Identifier(time_column))
    select_items = [bucket_expr.as_string(connection) + " AS timestamp"]
    group_items = ["1"]
    order_items = ["1"]
    params: list[Any] = [postgres_frequency]
    where_clauses = []
    if start_date:
        where_clauses.append(sql.SQL("{}::timestamp >= %s").format(sql.Identifier(time_column)).as_string(connection))
        params.append(start_date)
    if end_date:
        where_clauses.append(sql.SQL("{}::timestamp < %s").format(sql.Identifier(time_column)).as_string(connection))
        params.append(end_date)
    where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    if series_column:
        select_items.append(sql.Identifier(series_column).as_string(connection) + " AS series")
        group_items.append("series")
        order_items.append("series")

    target_identifier = sql.Identifier(target_column).as_string(connection)
    cleaned_target = (
        "NULLIF(regexp_replace("
        f"{target_identifier}::text, '[^0-9.+-]', '', 'g'"
        "), '')"
    )
    numeric_target = (
        "CASE "
        f"WHEN {cleaned_target} ~ '^[-+]?[0-9]*\\.?[0-9]+$' "
        f"THEN {cleaned_target}::numeric "
        "ELSE NULL "
        "END"
    )
    select_items.append(f"SUM({numeric_target}) AS forecast_target")
    if engineered_feature is not None:
        feature_name, count_column = engineered_feature
        select_items.append(f"COUNT({sql.Identifier(count_column).as_string(connection)}) AS {sql.Identifier(feature_name).as_string(connection)}")

    query = sql.SQL(
        "SELECT {select_list} FROM {table}{where_sql} GROUP BY {group_list} ORDER BY {order_list}"
    ).format(
        select_list=sql.SQL(", ").join(sql.SQL(item) for item in select_items),
        table=sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table_name)),
        where_sql=sql.SQL(where_sql),
        group_list=sql.SQL(", ").join(sql.SQL(item) for item in group_items),
        order_list=sql.SQL(", ").join(sql.SQL(item) for item in order_items),
    )
    return pd.read_sql_query(query.as_string(connection), connection, params=params)


def _postgres_date_trunc_frequency(frequency: str | None) -> str:
    if not frequency:
        return "day"
    normalized = str(frequency).upper()
    if normalized.startswith("H"):
        return "hour"
    if normalized.startswith("D"):
        return "day"
    if normalized.startswith("W"):
        return "week"
    if normalized.startswith("M"):
        return "month"
    if normalized.startswith("Q"):
        return "quarter"
    if normalized.startswith("Y") or normalized.startswith("A"):
        return "year"
    return "day"


def fetch_group_count_feature(
    connection,
    table_name: str,
    schema: str,
    group_by_column: str,
    count_column: str,
) -> pd.DataFrame:
    from psycopg2 import sql

    query = sql.SQL(
        """
        SELECT
            {group_by} AS group_value,
            COUNT({count_col}) AS engineered_count_feature
        FROM {table}
        GROUP BY {group_by}
        ORDER BY {group_by}
        """
    ).format(
        group_by=sql.Identifier(group_by_column),
        count_col=sql.Identifier(count_column),
        table=sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table_name)),
    )
    return pd.read_sql_query(query.as_string(connection), connection)


def fetch_indexes(connection, schema: str, table_name: str) -> IndexInfo:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = %s
              AND tablename = %s
            ORDER BY indexname
            """,
            (schema, table_name),
        )
        rows = cursor.fetchall()
    return IndexInfo(
        has_index=bool(rows),
        matching_indexes=[row[1] for row in rows],
    )


def warn_missing_time_index(index_info: IndexInfo, schema: str, table_name: str, time_column: str) -> str | None:
    if index_info.has_index:
        return None
    return f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{schema}_{table_name}_{time_column} ON {schema}.{table_name} ({time_column});"


def stream_raw_rows(connection, table_name: str, schema: str = "public", columns: list[str] | None = None, fetch_size: int = 10000) -> Iterable[pd.DataFrame]:
    from psycopg2 import sql

    if columns:
        column_list = sql.SQL(", ").join(sql.Identifier(column) for column in columns)
        query = sql.SQL("SELECT {} FROM {}.{}").format(column_list, sql.Identifier(schema), sql.Identifier(table_name))
    else:
        query = sql.SQL("SELECT * FROM {}.{}").format(sql.Identifier(schema), sql.Identifier(table_name))

    cursor = connection.cursor(name=f"stream_{table_name}")
    cursor.itersize = fetch_size
    cursor.execute(query)
    try:
        while True:
            rows = cursor.fetchmany(fetch_size)
            if not rows:
                break
            yield pd.DataFrame.from_records(rows, columns=[desc[0] for desc in cursor.description])
    finally:
        cursor.close()
