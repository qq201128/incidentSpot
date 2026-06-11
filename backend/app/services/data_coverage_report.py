from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import sqlite3
from typing import Any

from app.db.session import get_conn

MS_PER_SECOND = 1000
COVERAGE_DECIMALS = 2
STATUS_HEALTHY = "healthy"
STATUS_PARTIAL = "partial"
STATUS_MISSING = "missing"
STATUS_UNAVAILABLE = "unavailable"
OPEN_TIME_COLUMN = "open_time"


@dataclass(frozen=True)
class CoverageOptions:
    symbol: str = "BTCUSDT"
    interval: str = "10m"
    primary_only: bool = False


@dataclass(frozen=True)
class TableSpec:
    name: str
    group_columns: tuple[str, ...]
    time_column: str | None = OPEN_TIME_COLUMN


TABLE_SPECS = (
    TableSpec("klines", ("symbol", "interval")),
    TableSpec("klines_multi", ("symbol", "interval")),
    TableSpec("orderbook_features", ("symbol",)),
    TableSpec("orderbook_ticks", ("symbol",), "quote_time"),
    TableSpec("funding_features", ("symbol",)),
    TableSpec("futures_positioning_features", ("symbol",)),
    TableSpec("market_sentiment_features", ("source",)),
    TableSpec("onchain_features", ("symbol",)),
)


def build_data_coverage_report(
    options: CoverageOptions | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    opts = options or CoverageOptions()
    active_conn, should_close = _active_connection(conn)
    try:
        main = _main_range(active_conn, opts)
        return {
            "generatedAt": _utc_now(),
            "symbol": opts.symbol.strip().upper(),
            "interval": opts.interval,
            "mainRange": main,
            "tables": [_table_report(active_conn, opts, main, spec) for spec in TABLE_SPECS],
        }
    finally:
        if should_close:
            active_conn.close()


def _active_connection(conn: sqlite3.Connection | None) -> tuple[sqlite3.Connection, bool]:
    if conn is not None:
        return conn, False
    return get_conn(), True


def _main_range(conn: sqlite3.Connection, options: CoverageOptions) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT open_time) AS row_count,
               MIN(open_time) AS min_open_time,
               MAX(open_time) AS max_open_time
        FROM klines
        WHERE symbol = ? AND interval = ?
        """,
        (options.symbol.strip().upper(), options.interval),
    ).fetchone()
    count = _int_value(row, "row_count")
    return {
        "status": STATUS_HEALTHY if count > 0 else STATUS_MISSING,
        "rowCount": count,
        "minOpenTime": _row_value(row, "min_open_time"),
        "maxOpenTime": _row_value(row, "max_open_time"),
        "minTimeUtc": _ms_to_iso(_row_value(row, "min_open_time")),
        "maxTimeUtc": _ms_to_iso(_row_value(row, "max_open_time")),
        "missingReason": None if count > 0 else "no_rows",
    }


def _table_report(
    conn: sqlite3.Connection,
    options: CoverageOptions,
    main: dict[str, Any],
    spec: TableSpec,
) -> dict[str, Any]:
    if not _table_exists(conn, spec.name):
        return _missing_table(spec)
    rows = _summary_rows(conn, options, spec)
    if not rows:
        rows = [_empty_scope(options, spec)]
    return {
        "table": spec.name,
        "timeColumn": spec.time_column,
        "rows": [_summary_payload(conn, options, main, spec, row) for row in rows],
    }


def _summary_rows(
    conn: sqlite3.Connection,
    options: CoverageOptions,
    spec: TableSpec,
) -> list[sqlite3.Row]:
    where_sql, params = _target_where(spec, options)
    group_sql = ", ".join(spec.group_columns)
    select_sql = ", ".join(spec.group_columns)
    time_column = spec.time_column or OPEN_TIME_COLUMN
    sql = f"""
        SELECT {select_sql}, COUNT(*) AS row_count,
               MIN({time_column}) AS min_open_time,
               MAX({time_column}) AS max_open_time
        FROM {spec.name}
        {where_sql}
        GROUP BY {group_sql}
        ORDER BY {group_sql}
    """
    return list(conn.execute(sql, params).fetchall())


def _summary_payload(
    conn: sqlite3.Connection,
    options: CoverageOptions,
    main: dict[str, Any],
    spec: TableSpec,
    row: sqlite3.Row | dict[str, Any],
) -> dict[str, Any]:
    base = {
        **_group_values(row, spec),
        "rowCount": _int_value(row, "row_count"),
        "minOpenTime": _row_value(row, "min_open_time"),
        "maxOpenTime": _row_value(row, "max_open_time"),
        "minTimeUtc": _ms_to_iso(_row_value(row, "min_open_time")),
        "maxTimeUtc": _ms_to_iso(_row_value(row, "max_open_time")),
    }
    coverage = _coverage(conn, options, main, spec, row)
    return {**base, **coverage}


def _coverage(
    conn: sqlite3.Connection,
    options: CoverageOptions,
    main: dict[str, Any],
    spec: TableSpec,
    row: sqlite3.Row | dict[str, Any],
) -> dict[str, Any]:
    if _int_value(row, "row_count") == 0:
        return _coverage_payload(STATUS_MISSING, None, "no_rows")
    if _is_primary_klines_scope(options, spec, row):
        return _coverage_payload(STATUS_HEALTHY, 100.0, None)
    if spec.time_column != OPEN_TIME_COLUMN:
        return _coverage_payload(STATUS_UNAVAILABLE, None, "no_open_time_column")
    if int(main["rowCount"]) == 0:
        return _coverage_payload(STATUS_UNAVAILABLE, None, "main_range_no_rows")
    matched = _matched_open_times(conn, options, spec, row)
    pct = round(matched / int(main["rowCount"]) * 100.0, COVERAGE_DECIMALS)
    return _coverage_payload(_coverage_status(matched, int(main["rowCount"])), pct, None)


def _matched_open_times(
    conn: sqlite3.Connection,
    options: CoverageOptions,
    spec: TableSpec,
    row: sqlite3.Row | dict[str, Any],
) -> int:
    where_sql, params = _coverage_where(spec, row)
    sql = f"""
        SELECT COUNT(DISTINCT t.open_time) AS matched_count
        FROM {spec.name} t
        INNER JOIN klines k ON k.open_time = t.open_time
        WHERE k.symbol = ? AND k.interval = ? {where_sql}
    """
    result = conn.execute(sql, (options.symbol.strip().upper(), options.interval, *params)).fetchone()
    return _int_value(result, "matched_count")


def _coverage_where(
    spec: TableSpec,
    row: sqlite3.Row | dict[str, Any],
) -> tuple[str, tuple[Any, ...]]:
    clauses = [f"t.{column} = ?" for column in spec.group_columns]
    params = tuple(_row_value(row, column) for column in spec.group_columns)
    return "AND " + " AND ".join(clauses), params


def _target_where(spec: TableSpec, options: CoverageOptions) -> tuple[str, tuple[Any, ...]]:
    clauses: list[str] = []
    params: list[Any] = []
    if "symbol" in spec.group_columns:
        clauses.append("symbol = ?")
        params.append(options.symbol.strip().upper())
    if options.primary_only and "interval" in spec.group_columns:
        clauses.append("interval = ?")
        params.append(options.interval)
    if not clauses:
        return "", ()
    return "WHERE " + " AND ".join(clauses), tuple(params)


def _is_primary_klines_scope(
    options: CoverageOptions,
    spec: TableSpec,
    row: sqlite3.Row | dict[str, Any],
) -> bool:
    return (
        spec.name == "klines"
        and _row_value(row, "symbol") == options.symbol.strip().upper()
        and _row_value(row, "interval") == options.interval
    )


def _empty_scope(options: CoverageOptions, spec: TableSpec) -> dict[str, Any]:
    row: dict[str, Any] = {"row_count": 0, "min_open_time": None, "max_open_time": None}
    for column in spec.group_columns:
        row[column] = _default_group_value(options, column)
    return row


def _default_group_value(options: CoverageOptions, column: str) -> str | None:
    if column == "symbol":
        return options.symbol.strip().upper()
    if column == "interval":
        return options.interval
    if column == "source":
        return None
    raise ValueError(f"unsupported coverage group column: {column}")


def _missing_table(spec: TableSpec) -> dict[str, Any]:
    return {
        "table": spec.name,
        "timeColumn": spec.time_column,
        "rows": [{"status": STATUS_UNAVAILABLE, "rowCount": 0, "missingReason": "table_missing"}],
    }


def _coverage_payload(status: str, pct: float | None, reason: str | None) -> dict[str, Any]:
    return {"status": status, "coveragePct": pct, "missingReason": reason}


def _coverage_status(matched: int, expected: int) -> str:
    if matched == expected:
        return STATUS_HEALTHY
    if matched > 0:
        return STATUS_PARTIAL
    return STATUS_MISSING


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _group_values(row: sqlite3.Row | dict[str, Any], spec: TableSpec) -> dict[str, Any]:
    return {column: _row_value(row, column) for column in spec.group_columns}


def _row_value(row: sqlite3.Row | dict[str, Any] | None, key: str) -> Any:
    if row is None:
        return None
    return row[key] if isinstance(row, sqlite3.Row) else row.get(key)


def _int_value(row: sqlite3.Row | dict[str, Any] | None, key: str) -> int:
    value = _row_value(row, key)
    return 0 if value is None else int(value)


def _ms_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value) / MS_PER_SECOND, tz=timezone.utc).isoformat()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
