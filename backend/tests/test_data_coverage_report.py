from __future__ import annotations

import sqlite3

from app.services.data_coverage_report import CoverageOptions, build_data_coverage_report

TEN_MINUTES_MS = 600_000
START_TIME = 1_700_000_000_000


def test_empty_feature_tables_report_missing_rows() -> None:
    conn = _connect()
    _create_schema(conn)
    _insert_klines(conn, count=3)

    report = build_data_coverage_report(_options(), conn)

    assert report["mainRange"]["rowCount"] == 3
    orderbook = _first_row(report, "orderbook_features")
    assert orderbook["status"] == "missing"
    assert orderbook["rowCount"] == 0
    assert orderbook["coveragePct"] is None
    assert orderbook["missingReason"] == "no_rows"


def test_complete_feature_table_reports_healthy_coverage() -> None:
    conn = _connect()
    _create_schema(conn)
    _insert_klines(conn, count=3)
    for index in range(3):
        _insert_orderbook_feature(conn, START_TIME + index * TEN_MINUTES_MS)

    report = build_data_coverage_report(_options(), conn)

    orderbook = _first_row(report, "orderbook_features")
    assert orderbook["status"] == "healthy"
    assert orderbook["rowCount"] == 3
    assert orderbook["coveragePct"] == 100.0
    assert orderbook["missingReason"] is None


def test_partial_feature_table_reports_partial_coverage() -> None:
    conn = _connect()
    _create_schema(conn)
    _insert_klines(conn, count=4)
    _insert_orderbook_feature(conn, START_TIME)
    _insert_orderbook_feature(conn, START_TIME + TEN_MINUTES_MS)

    report = build_data_coverage_report(_options(), conn)

    orderbook = _first_row(report, "orderbook_features")
    assert orderbook["status"] == "partial"
    assert orderbook["rowCount"] == 2
    assert orderbook["coveragePct"] == 50.0


def test_quote_time_table_reports_unavailable_coverage() -> None:
    conn = _connect()
    _create_schema(conn)
    _insert_klines(conn, count=1)
    conn.execute(
        "INSERT INTO orderbook_ticks(symbol, quote_time) VALUES('BTCUSDT', ?)",
        (START_TIME,),
    )

    report = build_data_coverage_report(_options(), conn)

    ticks = _first_row(report, "orderbook_ticks")
    assert ticks["status"] == "unavailable"
    assert ticks["rowCount"] == 1
    assert ticks["coveragePct"] is None
    assert ticks["missingReason"] == "no_open_time_column"


def test_primary_only_limits_interval_scoped_tables() -> None:
    conn = _connect()
    _create_schema(conn)
    _insert_klines(conn, count=2)
    conn.execute(
        "INSERT INTO klines(symbol, interval, open_time) VALUES('BTCUSDT', '30m', ?)",
        (START_TIME,),
    )
    conn.execute(
        "INSERT INTO klines_multi(symbol, interval, open_time) VALUES('BTCUSDT', '30m', ?)",
        (START_TIME,),
    )

    report = build_data_coverage_report(
        CoverageOptions(symbol="BTCUSDT", interval="10m", primary_only=True),
        conn,
    )

    klines_rows = _table_rows(report, "klines")
    multi_rows = _table_rows(report, "klines_multi")
    assert [row["interval"] for row in klines_rows] == ["10m"]
    assert [row["interval"] for row in multi_rows] == ["10m"]
    assert klines_rows[0]["status"] == "healthy"
    assert klines_rows[0]["coveragePct"] == 100.0


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE klines (
          symbol TEXT NOT NULL,
          interval TEXT NOT NULL,
          open_time INTEGER NOT NULL
        );
        CREATE TABLE klines_multi (
          symbol TEXT NOT NULL,
          interval TEXT NOT NULL,
          open_time INTEGER NOT NULL
        );
        CREATE TABLE orderbook_features (
          symbol TEXT NOT NULL,
          open_time INTEGER NOT NULL
        );
        CREATE TABLE orderbook_ticks (
          symbol TEXT NOT NULL,
          quote_time INTEGER NOT NULL
        );
        CREATE TABLE funding_features (
          symbol TEXT NOT NULL,
          open_time INTEGER NOT NULL
        );
        CREATE TABLE futures_positioning_features (
          symbol TEXT NOT NULL,
          open_time INTEGER NOT NULL
        );
        CREATE TABLE market_sentiment_features (
          source TEXT NOT NULL,
          open_time INTEGER NOT NULL
        );
        CREATE TABLE onchain_features (
          symbol TEXT NOT NULL,
          open_time INTEGER NOT NULL
        );
        """
    )


def _insert_klines(conn: sqlite3.Connection, *, count: int) -> None:
    rows = [("BTCUSDT", "10m", START_TIME + index * TEN_MINUTES_MS) for index in range(count)]
    conn.executemany("INSERT INTO klines(symbol, interval, open_time) VALUES(?, ?, ?)", rows)


def _insert_orderbook_feature(conn: sqlite3.Connection, open_time: int) -> None:
    conn.execute(
        "INSERT INTO orderbook_features(symbol, open_time) VALUES('BTCUSDT', ?)",
        (open_time,),
    )


def _first_row(report: dict, table: str) -> dict:
    return _table_rows(report, table)[0]


def _table_rows(report: dict, table: str) -> list[dict]:
    table_report = next(item for item in report["tables"] if item["table"] == table)
    return table_report["rows"]


def _options() -> CoverageOptions:
    return CoverageOptions(symbol="BTCUSDT", interval="10m")
