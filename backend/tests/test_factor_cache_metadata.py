from __future__ import annotations

import sqlite3
from pathlib import Path

from app.services import factor_cache_metadata
from app.services import factor_combination_cache_service
from app.services import factor_ranking_cache_service

ONE_MINUTE_MS = 60_000


def test_factor_ranking_cache_becomes_stale_when_market_data_changes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    db_path = _init_db(tmp_path)
    _patch_cache_db(monkeypatch, db_path)
    _insert_kline(db_path, 0)

    factor_ranking_cache_service.save_cached_ranking(
        "BTCUSDT",
        "10m",
        [{"factorName": "factor_a", "totalPeriods": 100}],
    )

    assert factor_ranking_cache_service.get_cached_ranking("BTCUSDT", "10m")["cacheStatus"]["usable"] is True

    _insert_kline(db_path, ONE_MINUTE_MS)
    cached = factor_ranking_cache_service.get_cached_ranking("BTCUSDT", "10m")

    assert cached["cacheStatus"]["usable"] is False
    assert cached["cacheStatus"]["reason"] == "market_data_changed"


def test_legacy_combination_cache_is_marked_stale(monkeypatch, tmp_path: Path) -> None:
    db_path = _init_db(tmp_path)
    _patch_cache_db(monkeypatch, db_path)
    _insert_kline(db_path, 0)
    _insert_legacy_combo_cache(db_path)

    cached = factor_combination_cache_service.get_cached_combination_ranking("BTCUSDT", "10m")

    assert cached["ranking"][0]["factorName"] == "combo_a"
    assert cached["cacheStatus"]["usable"] is False
    assert cached["cacheStatus"]["reason"] == "legacy_without_fingerprint"


def _patch_cache_db(monkeypatch, db_path: Path) -> None:
    monkeypatch.setattr(factor_cache_metadata, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(factor_ranking_cache_service, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(factor_combination_cache_service, "get_conn", lambda: _connect(db_path))


def _init_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "cache.db"
    conn = _connect(db_path)
    try:
        conn.executescript(
            """
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
            CREATE TABLE factor_combo_ranking_cache (
              symbol TEXT NOT NULL,
              duration TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              total INTEGER NOT NULL,
              search_config TEXT NOT NULL,
              payload TEXT NOT NULL,
              PRIMARY KEY (symbol, duration)
            );
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


def _insert_kline(db_path: Path, open_time: int) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO klines(symbol, interval, open_time) VALUES('BTCUSDT', '1m', ?)",
            (open_time,),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_legacy_combo_cache(db_path: Path) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO factor_combo_ranking_cache(
              symbol, duration, updated_at, total, search_config, payload
            ) VALUES(
              'BTCUSDT', '10m', '2026-05-14T00:00:00+00:00', 1, '{}',
              '{"symbol":"BTCUSDT","duration":"10m","ranking":[{"factorName":"combo_a"}]}'
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
