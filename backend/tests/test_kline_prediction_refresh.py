from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.services.kline_prediction_refresh import (
    KlineRefreshDeps,
    KlineRefreshRequest,
    refresh_required_klines,
)

ONE_MINUTE_MS = 60_000


def test_refresh_required_klines_fills_forward_gap(tmp_path: Path) -> None:
    db_path = _init_db(tmp_path)
    _insert_rows(db_path, [0])
    fetches = []

    def fetch(symbol: str, interval: str, **kwargs) -> list[dict]:
        fetches.append((symbol, interval, kwargs))
        return _rows_between(kwargs["start_time"], kwargs["end_time"], ONE_MINUTE_MS)

    refresh_required_klines(
        KlineRefreshRequest("BTCUSDT", "1m", 3 * ONE_MINUTE_MS),
        _deps(db_path, fetch),
    )

    assert fetches == [
        (
            "BTCUSDT",
            "1m",
            {
                "limit": 3,
                "start_time": ONE_MINUTE_MS,
                "end_time": (4 * ONE_MINUTE_MS) - 1,
            },
        )
    ]
    assert _open_times(db_path) == [0, ONE_MINUTE_MS, 2 * ONE_MINUTE_MS, 3 * ONE_MINUTE_MS]


def test_refresh_required_klines_fills_internal_gap(tmp_path: Path) -> None:
    db_path = _init_db(tmp_path)
    _insert_rows(db_path, [0, ONE_MINUTE_MS, 3 * ONE_MINUTE_MS])
    fetches = []

    def fetch(symbol: str, interval: str, **kwargs) -> list[dict]:
        fetches.append(kwargs)
        return [_row(2 * ONE_MINUTE_MS)]

    refresh_required_klines(
        KlineRefreshRequest("BTCUSDT", "1m", 3 * ONE_MINUTE_MS),
        _deps(db_path, fetch),
    )

    assert fetches == [
        {
            "limit": 1,
            "start_time": 2 * ONE_MINUTE_MS,
            "end_time": (3 * ONE_MINUTE_MS) - 1,
        }
    ]
    assert _open_times(db_path) == [0, ONE_MINUTE_MS, 2 * ONE_MINUTE_MS, 3 * ONE_MINUTE_MS]


def test_refresh_required_klines_raises_when_range_fetch_empty(tmp_path: Path) -> None:
    db_path = _init_db(tmp_path)
    _insert_rows(db_path, [0])

    with pytest.raises(ValueError, match="no 1m klines returned.*range"):
        refresh_required_klines(
            KlineRefreshRequest("BTCUSDT", "1m", ONE_MINUTE_MS),
            _deps(db_path, lambda *_args, **_kwargs: []),
        )


def _deps(db_path: Path, fetch) -> KlineRefreshDeps:
    return KlineRefreshDeps(
        connect=lambda: _connect(db_path),
        fetch=fetch,
        upsert=lambda symbol, interval, rows: _upsert(db_path, symbol, interval, rows),
    )


def _init_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "klines.db"
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE klines (
              symbol TEXT NOT NULL,
              interval TEXT NOT NULL,
              open_time INTEGER NOT NULL,
              open REAL NOT NULL,
              high REAL NOT NULL,
              low REAL NOT NULL,
              close REAL NOT NULL,
              volume REAL NOT NULL,
              close_time INTEGER NOT NULL,
              PRIMARY KEY (symbol, interval, open_time)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _insert_rows(db_path: Path, open_times: list[int]) -> None:
    _upsert(db_path, "BTCUSDT", "1m", [_row(open_time) for open_time in open_times])


def _upsert(db_path: Path, symbol: str, interval: str, rows: list[dict]) -> None:
    conn = _connect(db_path)
    try:
        for row in rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO klines(
                  symbol, interval, open_time, open, high, low, close, volume, close_time
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    interval,
                    row["openTime"],
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                    row["closeTime"],
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _open_times(db_path: Path) -> list[int]:
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT open_time FROM klines ORDER BY open_time").fetchall()
    finally:
        conn.close()
    return [int(row["open_time"]) for row in rows]


def _rows_between(start_time: int, end_time: int, step_ms: int) -> list[dict]:
    rows = []
    current = int(start_time)
    while current <= int(end_time):
        rows.append(_row(current))
        current += step_ms
    return rows


def _row(open_time: int) -> dict:
    return {
        "openTime": int(open_time),
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        "close": 1.0,
        "volume": 1.0,
        "closeTime": int(open_time) + ONE_MINUTE_MS - 1,
    }
