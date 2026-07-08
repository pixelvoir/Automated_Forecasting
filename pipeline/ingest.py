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

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency
    pa = None
    pq = None

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


def _write_csv_parquet(file_path, raw_path: Path, block_bytes: int,
                       convert_options, row_limit: int = None) -> int:
    """Stream a CSV to Parquet block-by-block. pyarrow's streaming reader infers ONE schema
    for the whole file and applies it to every block (unlike pandas' per-chunk inference,
    which diverges across chunk boundaries and made the old ParquetWriter crash). Peak memory
    is one block, not the whole file. Returns the rows written."""
    import pyarrow.csv as pacsv
    read_options = pacsv.ReadOptions(block_size=block_bytes)
    convert_options = convert_options or pacsv.ConvertOptions()
    reader = pacsv.open_csv(file_path, read_options=read_options, convert_options=convert_options)
    writer = None
    total = 0
    try:
        for batch in reader:
            if row_limit is not None:
                if total >= row_limit:
                    break
                if total + batch.num_rows > row_limit:
                    batch = batch.slice(0, row_limit - total)
            if writer is None:
                writer = pq.ParquetWriter(raw_path, batch.schema, compression="snappy")
            writer.write_table(pa.Table.from_batches([batch]))
            total += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"CSV file is empty: {file_path}")
    return total


def _stream_csv_to_parquet(
    file_path: str,
    raw_path: Path,
    row_limit: int = None,
    block_size_mb: int = 64,
    top_n: int = 5,
) -> dict:
    """Convert a (potentially large) CSV to Parquet with bounded memory, then compute EXACT
    metadata from the columnar Parquet — one column at a time — so a 254MB CSV never has to
    live in RAM as a pandas DataFrame (which is what OOM-crashed the old path).

    Type handling: pyarrow infers proper column types for clean CSVs; if a column's values
    conflict under inference (pure-int early, a real string later), we retry reading every
    column as string. That's safe here — Stage 1 re-classifies numeric-as-string columns and
    Stage 2/3 re-coerce them (the pipeline is built for string-typed input, see CLAUDE.md)."""
    if pa is None or pq is None:
        raise RuntimeError(
            "CSV streaming requires 'pyarrow'. Install it to enable large file uploads.")

    import pyarrow.csv as pacsv
    block_bytes = max(int(block_size_mb), 1) * 1024 * 1024
    try:
        row_count = _write_csv_parquet(file_path, raw_path, block_bytes, None, row_limit)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
        # Inference conflict across blocks — fall back to reading everything as string.
        Path(raw_path).unlink(missing_ok=True)
        cols = pd.read_csv(file_path, nrows=0).columns.tolist()
        convert = pacsv.ConvertOptions(column_types={c: pa.string() for c in cols})
        row_count = _write_csv_parquet(file_path, raw_path, block_bytes, convert, row_limit)

    return {"metadata": extract_metadata_from_parquet(
        raw_path, source_bytes=Path(file_path).stat().st_size, top_n=top_n)}


def _count_duplicates_parquet(pf, batch_rows: int = 131_072) -> int:
    """Exact duplicate-row count with bounded memory: hash each row (batched) and keep only
    the uint64 hashes (8 bytes/row), not the rows themselves. 64-bit collisions are
    negligible, so this is effectively exact without ever materializing the full frame."""
    hashes = []
    for batch in pf.iter_batches(batch_size=batch_rows):
        dfb = batch.to_pandas()
        hashes.append(pd.util.hash_pandas_object(dfb, index=False).to_numpy())
    if not hashes:
        return 0
    allh = np.concatenate(hashes)
    return int(len(allh) - len(np.unique(allh)))


def extract_metadata_from_parquet(raw_path: Path, source_bytes: int, top_n: int = 5) -> dict:
    """Stage 1 statistics read column-by-column from a Parquet file, so peak memory is one
    column (not the whole dataset). Same output shape as extract_metadata(). No raw values."""
    pf = pq.ParquetFile(raw_path)
    try:
        cols = list(pf.schema_arrow.names)
        n_rows = pf.metadata.num_rows
        dup_count = _count_duplicates_parquet(pf)

        schema_rows, nulls, cardinality, numeric_stats = [], {}, {}, {}
        datetime_cols, frequency = [], {}
        for col in cols:
            s = pf.read(columns=[col]).column(0).to_pandas()
            dt = _infer_dtype(s)
            schema_rows.append({"col": col, "dtype_raw": str(s.dtype), "dtype_inferred": dt})
            null_count = int(s.isna().sum())
            nulls[col] = {"count": null_count,
                          "pct": round(null_count / max(n_rows, 1) * 100, 2)}
            if dt == "datetime":
                datetime_cols.append(col)
                frequency[col] = _infer_frequency(s)
            elif dt == "categorical":
                vc = s.value_counts(dropna=False).head(top_n)
                cardinality[col] = {
                    "unique_count": int(s.nunique()),
                    f"top_{top_n}": [{"value": str(v), "freq": int(c)} for v, c in vc.items()],
                }
            elif dt == "numeric":
                sn = pd.to_numeric(s, errors="coerce").dropna()
                if len(sn):
                    q1, q3 = float(np.percentile(sn, 25)), float(np.percentile(sn, 75))
                    numeric_stats[col] = {
                        "min": float(sn.min()), "max": float(sn.max()),
                        "mean": float(sn.mean()), "median": float(sn.median()),
                        "std": float(sn.std()), "skew": float(sn.skew()),
                        "kurtosis": float(sn.kurtosis()),
                        "q1": q1, "q3": q3, "iqr": round(q3 - q1, 6),
                    }
            del s

        return {
            "shape": {
                "rows": int(n_rows),
                "cols": int(len(cols)),
                "memory_mb": round(float(source_bytes) / 1e6, 3),
            },
            "schema": schema_rows,
            "nulls": nulls,
            "cardinality": cardinality,
            "numeric_stats": numeric_stats,
            "datetime_cols": datetime_cols,
            "frequency": frequency,
            "duplicates": {"row_count": dup_count},
        }
    finally:
        try:
            pf.close()
        except Exception:
            pass


# ── Metadata extraction ─────────────────────────────────────────────────────

def _infer_dtype(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if series.dtype == object:
        sample = series.dropna().head(200)
        if len(sample) == 0:
            return "categorical"
        # Numeric-as-string check first (matches Stage 2's _dtype_issues precedence) —
        # CSV/DB columns are commonly read as strings even when the values are numeric.
        # Getting this right here matters: numeric_stats below (and Stage 2's outlier
        # detection, which only looks at columns classified "numeric") both depend on it —
        # miss it here and a numeric column silently gets zero outlier analysis downstream.
        if pd.to_numeric(sample, errors="coerce").notna().mean() > 0.8:
            return "numeric"
        try:
            pd.to_datetime(sample)
            return "datetime"
        except Exception:
            pass
    return "categorical"


def _infer_frequency(series: pd.Series) -> str:
    """Infer the frequency of the time GRID from *distinct* timestamps.

    Deduplicating first is load-bearing: panel/multi-series data has many rows per
    timestamp (e.g. ~1,800 store×product rows per day), which drives the median raw
    row-to-row diff to 0 and mislabeled daily panels as "hourly". The grid frequency
    is what seasonal periods/windows need, regardless of how many series share it.
    Full Timedelta resolution (not .days) distinguishes hourly from daily properly,
    and errors="coerce" stops a few malformed values from forcing "unknown"."""
    try:
        s = pd.to_datetime(series.dropna(), errors="coerce").dropna()
        s = s.drop_duplicates().sort_values()
        if len(s) < 2:
            return "unknown"
        med = s.diff().dropna().median()
        if med <= pd.Timedelta(minutes=90):
            return "hourly"
        if med <= pd.Timedelta(hours=36):
            return "daily"
        median_days = med.days
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
            # to_numeric is a no-op for already-numeric dtypes and safely coerces columns
            # that were classified "numeric" via the string-detection path above.
            s = pd.to_numeric(df[col], errors="coerce").dropna()
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
    top_n = int(cfg.get("top_n_categories", 5) or 5)

    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            RAW_DIR.mkdir(parents=True, exist_ok=True)  # the pending parquet is written now
            streamed = _stream_csv_to_parquet(
                file_path=file_path,
                raw_path=RAW_DIR / "_pending_raw.parquet",
                row_limit=cfg.get("row_limit"),
                block_size_mb=int(cfg.get("csv_block_size_mb", 64) or 64),
                top_n=top_n,
            )
            metadata = streamed["metadata"]
            source_info = {"file": str(file_path)}
            df = None
        else:
            df = _read_file(file_path, row_limit=cfg.get("row_limit"))
            metadata = extract_metadata(df, top_n=top_n)
            source_info = {"file": str(file_path)}
    else:
        creds = credentials if credentials is not None else _credentials()
        conn_str = _build_conn_str(**creds)
        df = fetch_data(conn_str, table=table, query=query, row_limit=cfg.get("row_limit"))
        metadata = extract_metadata(df, top_n=top_n)
        source_info = {"table": table, "query": query}

    run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / f"{run_id}_raw.parquet"
    if file_path and Path(file_path).suffix.lower() == ".csv":
        pending_path = RAW_DIR / "_pending_raw.parquet"
        if pending_path.exists():
            pending_path.replace(raw_path)
    elif df is not None:
        df.to_parquet(raw_path, index=False)

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
