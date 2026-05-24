from __future__ import annotations

import sqlite3

from app.db.session import _ensure_auto_trade_strategies
from app.services import auto_trade_service, auto_trade_status
from app.services.auto_trade_types import AutoTradeSettings
from app.services.auto_trade_service import AUTO_TRADE_SLOT_DURATIONS
from app.services.rule_config import DURATION_TO_MINUTES
from app.services.strategy_registry import (
    FACTOR_COMBO_STRATEGY_KEY,
    HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
    strategy_payloads,
)
from app.services.model_family_config import MODEL_FAMILIES, model_family_strategy_key


def test_db_seed_enables_existing_factor_combo_sim_slots() -> None:
    conn = _auto_trade_conn()
    for duration in AUTO_TRADE_SLOT_DURATIONS:
        _insert_strategy(conn, FACTOR_COMBO_STRATEGY_KEY, duration, enabled=0, live=0)

    _ensure_auto_trade_strategies(conn)

    rows = conn.execute(
        """
        SELECT duration, enabled, live_trading_enabled
        FROM auto_trade_strategies
        WHERE strategy_key = ?
        """,
        (FACTOR_COMBO_STRATEGY_KEY,),
    ).fetchall()
    by_duration = {row["duration"]: row for row in rows}
    assert set(by_duration) == set(AUTO_TRADE_SLOT_DURATIONS)
    assert all(by_duration[duration]["enabled"] == 1 for duration in AUTO_TRADE_SLOT_DURATIONS)
    assert all(by_duration[duration]["live_trading_enabled"] == 0 for duration in AUTO_TRADE_SLOT_DURATIONS)
    assert _strategy_count(conn, "orderbook_notional_40m") == 0


def test_db_seed_removes_legacy_execution_slots() -> None:
    conn = _auto_trade_conn()
    _insert_strategy(conn, HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, "10m", enabled=1, live=0)
    _insert_strategy(conn, model_family_strategy_key("lstm", "10m"), "10m", enabled=1, live=0)

    _ensure_auto_trade_strategies(conn)

    assert _strategy_count(conn, HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY) == 0
    assert _strategy_count(conn, model_family_strategy_key("lstm", "10m")) == 0


def test_strategy_payloads_only_expose_factor_combo_execution_item() -> None:
    keys = {payload["key"] for payload in strategy_payloads()}

    assert keys == {FACTOR_COMBO_STRATEGY_KEY}


def test_current_bucket_prediction_is_treated_as_fresh(monkeypatch) -> None:
    settings = AutoTradeSettings(
        strategy_key=FACTOR_COMBO_STRATEGY_KEY,
        enabled=True,
        symbol="BTCUSDT",
        duration="10m",
        duration_minutes=10,
        qty=5.0,
        live_trading_enabled=False,
    )
    monkeypatch.setattr(
        auto_trade_service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: 1778922000000,
    )

    assert auto_trade_service._is_fresh_prediction(
        {"open_time": 1778922000000},
        settings,
    ) is True


def test_status_treats_current_bucket_prediction_as_fresh(monkeypatch) -> None:
    settings = AutoTradeSettings(
        strategy_key=FACTOR_COMBO_STRATEGY_KEY,
        enabled=True,
        symbol="BTCUSDT",
        duration="10m",
        duration_minutes=10,
        qty=5.0,
        live_trading_enabled=False,
    )
    monkeypatch.setattr(
        auto_trade_status,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: 1778922000000,
    )
    prediction = _status_prediction(open_time=1778922000000, quality_passed=False)

    status = {
        "settings": settings.to_response(),
        "openPosition": False,
        "latestPrediction": auto_trade_status._prediction_status(prediction, settings),
    }

    assert status["latestPrediction"]["fresh"] is True
    assert auto_trade_status._reason(status) == "signal_condition_not_met"


def _auto_trade_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE auto_trade_strategies (
          strategy_key TEXT NOT NULL,
          duration TEXT NOT NULL DEFAULT '10m',
          enabled INTEGER NOT NULL DEFAULT 0,
          live_trading_enabled INTEGER NOT NULL DEFAULT 0,
          symbol TEXT NOT NULL DEFAULT 'BTCUSDT',
          duration_minutes INTEGER NOT NULL DEFAULT 10,
          qty REAL NOT NULL DEFAULT 5,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (strategy_key, duration)
        )
        """
    )
    return conn


def _insert_strategy(
    conn: sqlite3.Connection,
    strategy_key: str,
    duration: str,
    *,
    enabled: int,
    live: int,
) -> None:
    conn.execute(
        """
        INSERT INTO auto_trade_strategies(
          strategy_key, duration, enabled, live_trading_enabled, symbol, duration_minutes, qty, updated_at
        )
        VALUES(?, ?, ?, ?, 'BTCUSDT', ?, 5, '2026-01-01T00:00:00+00:00')
        """,
        (strategy_key, duration, enabled, live, DURATION_TO_MINUTES[duration]),
    )
    conn.commit()


def _strategy_count(conn: sqlite3.Connection, strategy_key: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS total FROM auto_trade_strategies WHERE strategy_key = ?",
        (strategy_key,),
    ).fetchone()
    return int(row["total"])


def _status_prediction(*, open_time: int, quality_passed: bool) -> dict:
    return {
        "id": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "open_time": open_time,
        "probability_up": 0.8,
        "direction": "up",
        "trade_quality_score": 0.5,
        "trade_quality_passed": int(quality_passed),
    }
