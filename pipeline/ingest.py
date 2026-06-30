"""Stage 1: pull data from PostgreSQL, save locally as Parquet, extract metadata."""
import json
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import connectorx as cx
import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.pool import NullPool

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
RAW_DIR = ROOT / "data" / "raw"
CONFIG_PATH = ROOT / "config" / "settings.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("ingest", {})


# ── Connection helpers ──────────────────────────────────────────────────────

def _build_conn_str(host: str, port: int, db: str, user: str, password: str) -> str:
    """Build a connectorx-compatible PostgreSQL connection string (URL-encodes credentials)."""
    return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{db}"


def build_engine(host: str, port: int, db: str, user: str, password: str):
    """Create a read-only SQLAlchemy engine (used for lightweight metadata queries only)."""
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=user,
        password=password,
        host=host,
        port=port,
        database=db,
    )
    return create_engine(
        url,
        connect_args={
            "options": "-c default_transaction_read_only=on",
            "connect_timeout": 5,
        },
        poolclass=NullPool,  # every connection is physically closed on release — no idle connections
    )


def _credentials() -> dict:
    return {
        "host": os.environ["DB_HOST"],
        "port": int(os.environ.get("DB_PORT", 5432)),
        "db": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
    }


# ── Query helpers ───────────────────────────────────────────────────────────

def _validate_table_name(table: str) -> None:
    """Allow schema.table or table — alphanumeric, underscores, dots only."""
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", table):
        raise ValueError(
            f"Invalid table name '{table}'. Use letters, digits, underscores, "
            "or schema.table notation."
        )


def _strip_sql_comments(query: str) -> str:
    query = re.sub(r"--[^\n]*", "", query)
    query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)
    return query.strip()


def _build_table_sql(table: str, row_limit: int = None) -> str:
    """Build a properly quoted SELECT for a plain or schema-qualified table name."""
    _validate_table_name(table)
    if "." in table:
        schema, tbl = table.split(".", 1)
        from_clause = f'"{schema}"."{tbl}"'
    else:
        from_clause = f'"{table}"'
    limit = f" LIMIT {int(row_limit)}" if row_limit else ""
    return f"SELECT * FROM {from_clause}{limit}"


# ── Data fetch ──────────────────────────────────────────────────────────────

def fetch_data(
    conn_str: str,
    table: str = None,
    query: str = None,
    row_limit: int = None,
) -> pd.DataFrame:
    """Pull data from the database via connectorx (Rust-backed, parallel fetch).
    Only SELECT statements are permitted — enforced before any data is fetched.
    """
    if query:
        clean = _strip_sql_comments(query)
        if not clean.upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are permitted.")
        sql = clean
    elif table:
        sql = _build_table_sql(table, row_limit)
    else:
        raise ValueError("Provide either 'table' or 'query'.")

    return cx.read_sql(conn_str, sql, return_type="pandas")


def list_tables(credentials: dict = None) -> list[dict]:
    """Return all user tables in the connected DB. Queries information_schema only (read-only).
    credentials dict: {host, port, db, user, password}. Falls back to env vars if None.
    Engine is disposed immediately after the query — no persistent connection kept."""
    creds = credentials if credentials is not None else _credentials()
    engine = build_engine(**creds)
    sql = text("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
          AND table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [{"schema": r[0], "table": r[1], "qualified": f"{r[0]}.{r[1]}"} for r in rows]
    finally:
        engine.dispose()  # disconnect immediately — never hold an idle connection


def _read_file(file_path: str, row_limit: int = None) -> pd.DataFrame:
    """Read a local CSV, Excel, or Parquet file into a DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=row_limit)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, nrows=row_limit)
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        return df.head(row_limit) if row_limit else df
    raise ValueError(f"Unsupported file type '{suffix}'. Supported: .csv, .xlsx, .xls, .parquet")


# ── Metadata extraction ─────────────────────────────────────────────────────

def _infer_dtype(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if series.dtype == object:
        sample = series.dropna().head(50)
        try:
            pd.to_datetime(sample)
            return "datetime"
        except Exception:
            pass
    return "categorical"


def _infer_frequency(series: pd.Series) -> str:
    try:
        s = pd.to_datetime(series.dropna()).sort_values()
        if len(s) < 2:
            return "unknown"
        median_days = s.diff().dropna().median().days
        if median_days == 0:
            return "hourly"
        if median_days == 1:
            return "daily"
        if 5 <= median_days <= 8:
            return "weekly"
        if 25 <= median_days <= 32:
            return "monthly"
        if 85 <= median_days <= 95:
            return "quarterly"
        if 360 <= median_days <= 370:
            return "yearly"
        return f"every_{median_days}_days"
    except Exception:
        return "unknown"


def extract_metadata(df: pd.DataFrame, top_n: int = 5) -> dict:
    """Compute Stage 1 statistics. No raw data values are included in the output."""
    dtype_map = {col: _infer_dtype(df[col]) for col in df.columns}
    datetime_cols = [col for col, dt in dtype_map.items() if dt == "datetime"]

    schema = [
        {"col": col, "dtype_raw": str(df[col].dtype), "dtype_inferred": dtype_map[col]}
        for col in df.columns
    ]

    nulls = {
        col: {
            "count": int(df[col].isna().sum()),
            "pct": round(float(df[col].isna().mean()) * 100, 2),
        }
        for col in df.columns
    }

    cardinality = {}
    for col, dt in dtype_map.items():
        if dt == "categorical":
            vc = df[col].value_counts(dropna=False).head(top_n)
            cardinality[col] = {
                "unique_count": int(df[col].nunique()),
                f"top_{top_n}": [{"value": str(v), "freq": int(c)} for v, c in vc.items()],
            }

    numeric_stats = {}
    for col, dt in dtype_map.items():
        if dt == "numeric":
            s = df[col].dropna()
            if len(s) == 0:
                continue
            q1, q3 = float(np.percentile(s, 25)), float(np.percentile(s, 75))
            numeric_stats[col] = {
                "min": float(s.min()), "max": float(s.max()),
                "mean": float(s.mean()), "median": float(s.median()),
                "std": float(s.std()), "skew": float(s.skew()),
                "kurtosis": float(s.kurtosis()),
                "q1": q1, "q3": q3, "iqr": round(q3 - q1, 6),
            }

    return {
        "shape": {
            "rows": len(df),
            "cols": len(df.columns),
            "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 3),
        },
        "schema": schema,
        "nulls": nulls,
        "cardinality": cardinality,
        "numeric_stats": numeric_stats,
        "datetime_cols": datetime_cols,
        "frequency": {col: _infer_frequency(df[col]) for col in datetime_cols},
        "duplicates": {"row_count": int(df.duplicated().sum())},
    }


# ── Stage 1 entry point ─────────────────────────────────────────────────────

def run(
    table: str = None,
    query: str = None,
    credentials: dict = None,
    file_path: str = None,
) -> dict:
    """Stage 1 entry point: ingest from PostgreSQL or a local file and save locally.

    DB mode: credentials dict {host, port, db, user, password} or falls back to env vars.
    File mode: file_path to a local .csv / .xlsx / .parquet file.
    """
    cfg = _load_config()

    if file_path:
        df = _read_file(file_path, row_limit=cfg.get("row_limit"))
        source_info = {"file": str(file_path)}
    else:
        creds = credentials if credentials is not None else _credentials()
        conn_str = _build_conn_str(**creds)
        df = fetch_data(conn_str, table=table, query=query, row_limit=cfg.get("row_limit"))
        source_info = {"table": table, "query": query}

    run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / f"{run_id}_raw.parquet"
    df.to_parquet(raw_path, index=False)

    metadata = extract_metadata(df, top_n=cfg.get("top_n_categories", 5))
    metadata["run_id"] = run_id
    metadata["source"] = source_info

    meta_path = run_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    return {
        "run_id": run_id,
        "rows": metadata["shape"]["rows"],
        "cols": metadata["shape"]["cols"],
        "raw_path": str(raw_path),
        "metadata_path": str(meta_path),
    }
