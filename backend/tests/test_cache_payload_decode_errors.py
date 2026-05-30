from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from app.services import factor_cache_metadata
from app.services import factor_combo_backtest_cache_service
from app.services import factor_combination_signal_cache_service
from app.services import factor_ranking_cache_service
from app.services import high_winrate_combo_cache_service
from app.services.cache_payloads import CachePayloadDecodeError

UPDATED_AT = "2026-05-14T00:00:00+00:00"
CACHE_DECODE_SCHEMA = """
CREATE TABLE klines (
  symbol TEXT NOT NULL,
  interval TEXT NOT NULL,
  open_time INTEGER NOT NULL
);
CREATE TABLE factor_ranking_cache (
  symbol TEXT NOT NULL,
  duration TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  total INTEGER NOT NULL,
  payload TEXT NOT NULL,
  PRIMARY KEY (symbol, duration)
);
CREATE TABLE high_winrate_combo_ranking_cache (
  symbol TEXT NOT NULL,
  duration TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  total INTEGER NOT NULL,
  search_config TEXT NOT NULL,
  payload TEXT NOT NULL,
  PRIMARY KEY (symbol, duration)
);
CREATE TABLE factor_combo_backtest_cache (
  symbol TEXT NOT NULL,
  duration TEXT NOT NULL,
  factor_name TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  payload TEXT NOT NULL,
  PRIMARY KEY (symbol, duration, factor_name)
);
CREATE TABLE factor_combo_signal_cache (
  symbol TEXT NOT NULL PRIMARY KEY,
  updated_at TEXT NOT NULL,
  top_per_duration INTEGER NOT NULL,
  limit_count INTEGER NOT NULL,
  payload TEXT NOT NULL
);
"""


def test_corrupt_factor_ranking_cache_payload_is_exposed(monkeypatch, tmp_path: Path) -> None:
    db_path = _init_db(tmp_path)
    _patch_cache_db(monkeypatch, db_path)
    _insert_factor_ranking_payload(db_path, "{broken")

    _assert_cache_decode_error(
        lambda: factor_ranking_cache_service.get_cached_ranking("btcusdt", "10m"),
        {"cacheName": "factor_ranking_cache", "symbol": "BTCUSDT", "duration": "10m"},
    )


def test_corrupt_high_winrate_combo_cache_payload_is_exposed(monkeypatch, tmp_path: Path) -> None:
    db_path = _init_db(tmp_path)
    _patch_cache_db(monkeypatch, db_path)
    _insert_high_winrate_payload(db_path, "{broken")

    _assert_cache_decode_error(
        lambda: high_winrate_combo_cache_service.get_cached_high_winrate_combo_ranking("btcusdt", "10m"),
        {"cacheName": "high_winrate_combo_ranking_cache", "symbol": "BTCUSDT", "duration": "10m"},
    )


def test_corrupt_combo_backtest_cache_payload_is_exposed(monkeypatch, tmp_path: Path) -> None:
    db_path = _init_db(tmp_path)
    _patch_cache_db(monkeypatch, db_path)
    _insert_combo_backtest_payload(db_path, "{broken")

    _assert_cache_decode_error(
        lambda: factor_combo_backtest_cache_service.get_usable_combo_backtest(
            "btcusdt",
            "10m",
            "combo__a__b",
        ),
        {
            "cacheName": "factor_combo_backtest_cache",
            "symbol": "BTCUSDT",
            "duration": "10m",
            "factorName": "combo__a__b",
        },
    )


def test_corrupt_combo_signal_cache_payload_is_exposed(monkeypatch, tmp_path: Path) -> None:
    db_path = _init_db(tmp_path)
    _patch_cache_db(monkeypatch, db_path)
    _insert_combo_signal_payload(db_path, "{broken")

    _assert_cache_decode_error(
        lambda: factor_combination_signal_cache_service.get_cached_combination_signals("btcusdt"),
        {"cacheName": "factor_combo_signal_cache", "symbol": "BTCUSDT"},
    )


def _assert_cache_decode_error(action: Callable[[], Any], expected: dict[str, Any]) -> None:
    with pytest.raises(CachePayloadDecodeError) as raised:
        action()
    for key, value in expected.items():
        assert raised.value.details[key] == value
    assert raised.value.details["exceptionType"] == "JSONDecodeError"


def _patch_cache_db(monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
    monkeypatch.setattr(factor_cache_metadata, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(factor_combo_backtest_cache_service, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(factor_combination_signal_cache_service, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(factor_ranking_cache_service, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(high_winrate_combo_cache_service, "get_conn", lambda: _connect(db_path))


def _init_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "cache_decode.db"
    conn = _connect(db_path)
    try:
        conn.executescript(CACHE_DECODE_SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return db_path


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _insert_factor_ranking_payload(db_path: Path, payload: str) -> None:
    _execute(
        db_path,
        "INSERT INTO factor_ranking_cache(symbol, duration, updated_at, total, payload) VALUES(?, ?, ?, ?, ?)",
        ("BTCUSDT", "10m", UPDATED_AT, 1, payload),
    )


def _insert_high_winrate_payload(db_path: Path, payload: str) -> None:
    _execute(
        db_path,
        """
        INSERT INTO high_winrate_combo_ranking_cache(
          symbol, duration, updated_at, total, search_config, payload
        ) VALUES(?, ?, ?, ?, ?, ?)
        """,
        ("BTCUSDT", "10m", UPDATED_AT, 1, "{}", payload),
    )


def _insert_combo_backtest_payload(db_path: Path, payload: str) -> None:
    _execute(
        db_path,
        """
        INSERT INTO factor_combo_backtest_cache(symbol, duration, factor_name, updated_at, payload)
        VALUES(?, ?, ?, ?, ?)
        """,
        ("BTCUSDT", "10m", "combo__a__b", UPDATED_AT, payload),
    )


def _insert_combo_signal_payload(db_path: Path, payload: str) -> None:
    _execute(
        db_path,
        """
        INSERT INTO factor_combo_signal_cache(symbol, updated_at, top_per_duration, limit_count, payload)
        VALUES(?, ?, ?, ?, ?)
        """,
        ("BTCUSDT", UPDATED_AT, 2, 5, payload),
    )


def _execute(db_path: Path, query: str, values: tuple[Any, ...]) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(query, values)
        conn.commit()
    finally:
        conn.close()
