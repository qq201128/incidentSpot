from __future__ import annotations

import sqlite3
from pathlib import Path

from app.services import factor_cache_metadata
from app.services import factor_combination_cache_service
from app.services import factor_ranking_cache_service
from app.services.auto_trade_types import AutoTradeSettings

ONE_MINUTE_MS = 60_000
TEN_MINUTES_MS = 10 * ONE_MINUTE_MS


def test_factor_ranking_cache_becomes_stale_when_market_data_changes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    db_path = _init_db(tmp_path)
    _patch_cache_db(monkeypatch, db_path)
    _insert_kline(db_path, "1m", 0)
    _insert_kline(db_path, "10m", 0)

    factor_ranking_cache_service.save_cached_ranking(
        "BTCUSDT",
        "10m",
        [{"factorName": "factor_a", "totalPeriods": 100}],
    )

    assert factor_ranking_cache_service.get_cached_ranking("BTCUSDT", "10m")["cacheStatus"]["usable"] is True

    _insert_kline(db_path, "1m", ONE_MINUTE_MS)
    cached = factor_ranking_cache_service.get_cached_ranking("BTCUSDT", "10m")

    assert cached["cacheStatus"]["usable"] is True

    _insert_kline(db_path, "10m", TEN_MINUTES_MS)
    cached = factor_ranking_cache_service.get_cached_ranking("BTCUSDT", "10m")

    assert cached["cacheStatus"]["usable"] is False
    assert cached["cacheStatus"]["reason"] == "market_data_changed"


def test_factor_ranking_cache_preserves_diagnostics(monkeypatch, tmp_path: Path) -> None:
    db_path = _init_db(tmp_path)
    _patch_cache_db(monkeypatch, db_path)
    _insert_kline(db_path, "10m", 0)

    factor_ranking_cache_service.save_cached_ranking(
        "BTCUSDT",
        "10m",
        [{"factorName": "factor_a", "totalPeriods": 100}],
        diagnostics={"factorDefinitionCount": 2, "failureCount": 1},
        failures=[{"factorName": "agent_a", "error": "formula column not found: missing"}],
    )

    cached = factor_ranking_cache_service.get_cached_ranking("BTCUSDT", "10m")

    assert cached["rankingDiagnostics"]["failureCount"] == 1
    assert cached["rankingFailures"][0]["factorName"] == "agent_a"


def test_factor_ranking_precomputed_symbols_include_enabled_auto_trade_symbols(monkeypatch) -> None:
    monkeypatch.setattr(
        factor_ranking_cache_service,
        "list_auto_trade_settings",
        lambda: [
            _auto_trade_settings("BTCUSDT", enabled=False),
            _auto_trade_settings("ETHUSDT", enabled=True),
        ],
    )

    assert factor_ranking_cache_service.factor_ranking_precomputed_symbols() == ["BTCUSDT", "ETHUSDT"]


def test_legacy_combination_cache_is_marked_stale(monkeypatch, tmp_path: Path) -> None:
    db_path = _init_db(tmp_path)
    _patch_cache_db(monkeypatch, db_path)
    _insert_kline(db_path, "10m", 0)
    _insert_legacy_combo_cache(db_path)

    cached = factor_combination_cache_service.get_cached_combination_ranking("BTCUSDT", "10m")

    assert cached["ranking"][0]["factorName"] == "combo_a"
    assert cached["cacheStatus"]["usable"] is False
    assert cached["cacheStatus"]["reason"] == "legacy_without_fingerprint"


def test_empty_combination_report_does_not_overwrite_nonempty_cache(monkeypatch, tmp_path: Path) -> None:
    db_path = _init_db(tmp_path)
    _patch_cache_db(monkeypatch, db_path)
    _insert_kline(db_path, "10m", 0)

    factor_combination_cache_service.save_cached_combination_ranking(
        {
            "symbol": "BTCUSDT",
            "duration": "10m",
            "ranking": [{"factorName": "combo__a__b"}],
            "searchConfig": {},
        }
    )
    factor_combination_cache_service.save_cached_combination_ranking(
        {
            "symbol": "BTCUSDT",
            "duration": "10m",
            "ranking": [],
            "searchConfig": {},
        }
    )

    cached = factor_combination_cache_service.get_cached_combination_ranking("BTCUSDT", "10m")

    assert [row["factorName"] for row in cached["ranking"]] == ["combo__a__b"]


def test_diagnostic_empty_combination_report_overwrites_nonempty_cache(monkeypatch, tmp_path: Path) -> None:
    db_path = _init_db(tmp_path)
    _patch_cache_db(monkeypatch, db_path)
    _insert_kline(db_path, "10m", 0)

    factor_combination_cache_service.save_cached_combination_ranking(
        {
            "symbol": "BTCUSDT",
            "duration": "10m",
            "ranking": [{"factorName": "combo__a__b"}],
            "searchConfig": {},
        }
    )
    factor_combination_cache_service.save_cached_combination_ranking(
        {
            "symbol": "BTCUSDT",
            "duration": "10m",
            "ranking": [],
            "searchConfig": {},
            "searchDiagnostics": {"evaluatedCombinationCount": 800},
        }
    )

    cached = factor_combination_cache_service.get_cached_combination_ranking("BTCUSDT", "10m")

    assert cached["ranking"] == []
    assert cached["searchDiagnostics"]["evaluatedCombinationCount"] == 800


def test_append_only_market_change_is_allowed_for_live_signal() -> None:
    payload = {
        "cacheStatus": {
            "usable": False,
            "reason": "market_data_changed",
            "cachedMarketData": {"rowCount": 1000, "maxOpenTime": 100},
            "currentMarketData": {"rowCount": 1001, "maxOpenTime": 200},
        }
    }

    assert factor_cache_metadata.cache_is_usable(payload) is False
    assert factor_cache_metadata.cache_is_usable_for_live_signal(payload) is True
    assert factor_cache_metadata.live_signal_cache_reason(payload) == "market_data_appended"


def test_rewritten_market_change_is_not_allowed_for_live_signal() -> None:
    payload = {
        "cacheStatus": {
            "usable": False,
            "reason": "market_data_changed",
            "cachedMarketData": {"rowCount": 1000, "maxOpenTime": 200},
            "currentMarketData": {"rowCount": 999, "maxOpenTime": 100},
        }
    }

    assert factor_cache_metadata.cache_is_usable_for_live_signal(payload) is False


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


def _insert_kline(db_path: Path, interval: str, open_time: int) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO klines(symbol, interval, open_time) VALUES('BTCUSDT', ?, ?)",
            (interval, open_time),
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


def _auto_trade_settings(symbol: str, *, enabled: bool) -> AutoTradeSettings:
    return AutoTradeSettings(
        strategy_key="factor_combo_ranker_v1",
        enabled=enabled,
        symbol=symbol,
        duration="10m",
        duration_minutes=10,
        qty=5.0,
        live_trading_enabled=False,
    )
