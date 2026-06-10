from __future__ import annotations

import sqlite3

from app.db.session import _ensure_auto_trade_strategies
from app.services import auto_trade_service, auto_trade_status
from app.services.runtime_symbols import configured_runtime_symbols
from app.services.auto_trade_types import AutoTradeSettings
from app.services.auto_trade_service import AUTO_TRADE_SLOT_DURATIONS
from app.services.factor_candidate_signal_keys import factor_candidate_signal_key
from app.services.factor_combo_simulation_keys import simulation_strategy_key_for_factor_name
from app.services.event_final_decision_service import EVENT_FINAL_DECISION_STRATEGY_KEY
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
        SELECT symbol, duration, enabled, live_trading_enabled
        FROM auto_trade_strategies
        WHERE strategy_key = ?
        """,
        (FACTOR_COMBO_STRATEGY_KEY,),
    ).fetchall()
    by_duration = {row["duration"]: row for row in rows if row["symbol"] == "BTCUSDT"}
    assert set(by_duration) == set(AUTO_TRADE_SLOT_DURATIONS)
    assert all(by_duration[duration]["enabled"] == 1 for duration in AUTO_TRADE_SLOT_DURATIONS)
    assert all(by_duration[duration]["live_trading_enabled"] == 0 for duration in AUTO_TRADE_SLOT_DURATIONS)
    assert _strategy_count(conn, "orderbook_notional_40m") == 0
    eth_rows = conn.execute(
        """
        SELECT duration, enabled, live_trading_enabled
        FROM auto_trade_strategies
        WHERE strategy_key = ? AND symbol = 'ETHUSDT'
        """,
        (FACTOR_COMBO_STRATEGY_KEY,),
    ).fetchall()
    assert {row["duration"] for row in eth_rows} == set(AUTO_TRADE_SLOT_DURATIONS)


def test_db_seed_keeps_model_family_simulation_slots() -> None:
    conn = _auto_trade_conn()
    _insert_strategy(conn, HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, "10m", enabled=1, live=0)
    model_key = model_family_strategy_key("lstm", "10m")
    _insert_strategy(conn, model_key, "10m", enabled=1, live=1)

    _ensure_auto_trade_strategies(conn)

    assert _strategy_count(conn, HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY) == 0
    assert _strategy_count(conn, model_key) == 2
    assert _strategy_live_enabled(conn, model_key, "BTCUSDT") == 1


def test_strategy_payloads_expose_factor_combo_and_model_family_simulation_items() -> None:
    keys = {payload["key"] for payload in strategy_payloads()}
    expected_model_keys = {
        model_family_strategy_key(family, duration)
        for family in MODEL_FAMILIES
        for duration in AUTO_TRADE_SLOT_DURATIONS
    }

    assert keys == {FACTOR_COMBO_STRATEGY_KEY, EVENT_FINAL_DECISION_STRATEGY_KEY, *expected_model_keys}


def test_default_runtime_symbols_include_btc_and_eth(monkeypatch) -> None:
    monkeypatch.delenv("FACTOR_RANKING_SYMBOLS", raising=False)

    assert configured_runtime_symbols() == ("BTCUSDT", "ETHUSDT")


def test_auto_trade_strategy_slots_allow_same_strategy_duration_for_btc_and_eth(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "slots.db"
    _auto_trade_conn(db_path).close()
    monkeypatch.setattr(auto_trade_service, "get_conn", lambda: _connect(db_path))
    monkeypatch.delenv("FACTOR_RANKING_SYMBOLS", raising=False)

    btc = AutoTradeSettings(FACTOR_COMBO_STRATEGY_KEY, True, "BTCUSDT", "10m", 10, 5.0, False)
    eth = AutoTradeSettings(FACTOR_COMBO_STRATEGY_KEY, True, "ETHUSDT", "10m", 10, 7.0, False)
    auto_trade_service.update_auto_trade_settings(btc)
    auto_trade_service.update_auto_trade_settings(eth)

    conn = _connect(db_path)
    try:
        rows = conn.execute(
        """
        SELECT symbol, qty
        FROM auto_trade_strategies
        WHERE strategy_key = ? AND duration = ?
        ORDER BY symbol
        """,
            (FACTOR_COMBO_STRATEGY_KEY, "10m"),
        ).fetchall()
    finally:
        conn.close()
    assert [(row["symbol"], row["qty"]) for row in rows] == [("BTCUSDT", 5.0), ("ETHUSDT", 7.0)]


def test_updating_eth_auto_trade_settings_does_not_overwrite_btc(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "slots.db"
    _auto_trade_conn(db_path).close()
    monkeypatch.setattr(auto_trade_service, "get_conn", lambda: _connect(db_path))
    btc = AutoTradeSettings(FACTOR_COMBO_STRATEGY_KEY, True, "BTCUSDT", "10m", 10, 5.0, False)
    eth = AutoTradeSettings(FACTOR_COMBO_STRATEGY_KEY, False, "ETHUSDT", "10m", 10, 8.0, False)
    auto_trade_service.update_auto_trade_settings(btc)
    auto_trade_service.update_auto_trade_settings(eth)

    updated_eth = AutoTradeSettings(FACTOR_COMBO_STRATEGY_KEY, True, "ETHUSDT", "10m", 10, 9.0, False)
    auto_trade_service.update_auto_trade_settings(updated_eth)

    conn = _connect(db_path)
    try:
        btc_row = _strategy_row(conn, FACTOR_COMBO_STRATEGY_KEY, "BTCUSDT", "10m")
        eth_row = _strategy_row(conn, FACTOR_COMBO_STRATEGY_KEY, "ETHUSDT", "10m")
    finally:
        conn.close()
    assert btc_row["enabled"] == 1
    assert btc_row["qty"] == 5.0
    assert eth_row["enabled"] == 1
    assert eth_row["qty"] == 9.0


def test_run_auto_trade_once_creates_btc_and_eth_sim_events(monkeypatch) -> None:
    current_open = 1778922000000
    settings = [
        AutoTradeSettings(FACTOR_COMBO_STRATEGY_KEY, True, "BTCUSDT", "10m", 10, 5.0, False),
        AutoTradeSettings(FACTOR_COMBO_STRATEGY_KEY, True, "ETHUSDT", "10m", 10, 5.0, False),
    ]
    predictions = {
        "BTCUSDT": _auto_trade_prediction("BTCUSDT", current_open, 0.8),
        "ETHUSDT": _auto_trade_prediction("ETHUSDT", current_open, 0.2),
    }
    created = []
    monkeypatch.setattr(auto_trade_service, "list_auto_trade_settings", lambda: settings)
    monkeypatch.setattr(auto_trade_service, "_has_open_position", lambda *_args: False)
    monkeypatch.setattr(auto_trade_service, "_latest_prediction_row", lambda item: predictions[item.symbol])
    monkeypatch.setattr(auto_trade_service, "evaluate_market_regime_trade_gate", _allowed_regime_gate)
    monkeypatch.setattr(
        auto_trade_service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: current_open,
    )

    def create(settings: AutoTradeSettings, prediction: dict) -> dict:
        created.append((settings.symbol, prediction["symbol"]))
        return {"eventId": len(created), "orderId": len(created), "symbol": settings.symbol}

    monkeypatch.setattr(auto_trade_service, "_create_trade", create)

    results = auto_trade_service.run_auto_trade_once()

    assert [row["symbol"] for row in results] == ["BTCUSDT", "ETHUSDT"]
    assert created == [("BTCUSDT", "BTCUSDT"), ("ETHUSDT", "ETHUSDT")]


def test_run_auto_trade_once_skips_when_market_regime_blocks(monkeypatch) -> None:
    current_open = 1778922000000
    settings = [AutoTradeSettings(FACTOR_COMBO_STRATEGY_KEY, True, "BTCUSDT", "10m", 10, 5.0, False)]
    monkeypatch.setattr(auto_trade_service, "list_auto_trade_settings", lambda: settings)
    monkeypatch.setattr(auto_trade_service, "_has_open_position", lambda *_args: False)
    monkeypatch.setattr(
        auto_trade_service,
        "_latest_prediction_row",
        lambda _settings: _auto_trade_prediction("BTCUSDT", current_open, 0.2),
    )
    monkeypatch.setattr(
        auto_trade_service,
        "current_rule_entry_open_time_for_duration",
        lambda _duration, _now_ms=None: current_open,
    )
    monkeypatch.setattr(auto_trade_service, "evaluate_market_regime_trade_gate", _blocked_regime_gate)
    monkeypatch.setattr(
        auto_trade_service,
        "_create_trade",
        lambda *_args: (_ for _ in ()).throw(AssertionError("trade should be blocked")),
    )

    assert auto_trade_service.run_auto_trade_once() == []


def test_list_auto_trade_settings_includes_dynamic_simulation_slots(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "slots.db"
    conn = _auto_trade_conn(db_path)
    factor_key = factor_candidate_signal_key("agent__alpha")
    combo_key = simulation_strategy_key_for_factor_name("combo__a__b")
    _insert_dynamic_strategy(conn, factor_key, "BTCUSDT", "10m")
    _insert_dynamic_strategy(conn, combo_key, "BTCUSDT", "30m")
    conn.close()
    monkeypatch.setattr(auto_trade_service, "get_conn", lambda: _connect(db_path))
    monkeypatch.delenv("FACTOR_RANKING_SYMBOLS", raising=False)

    settings = auto_trade_service.list_auto_trade_settings()
    by_key = {(item.strategy_key, item.symbol, item.duration): item for item in settings}

    assert by_key[(factor_key, "BTCUSDT", "10m")].enabled is True
    assert by_key[(combo_key, "BTCUSDT", "30m")].enabled is True
    assert by_key[(factor_key, "BTCUSDT", "10m")].live_trading_enabled is True
    assert by_key[(combo_key, "BTCUSDT", "30m")].live_trading_enabled is True


def test_factor_candidate_signal_live_trading_setting_is_saved(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "slots.db"
    _auto_trade_conn(db_path).close()
    monkeypatch.setattr(auto_trade_service, "get_conn", lambda: _connect(db_path))
    factor_key = factor_candidate_signal_key("agent__alpha")
    settings = AutoTradeSettings(factor_key, True, "BTCUSDT", "10m", 10, 5.0, True)

    updated = auto_trade_service.update_auto_trade_settings(settings)

    assert updated.live_trading_enabled is True
    persisted = auto_trade_service.list_auto_trade_settings()
    by_key = {(item.strategy_key, item.symbol, item.duration): item for item in persisted}
    assert by_key[(factor_key, "BTCUSDT", "10m")].live_trading_enabled is True


def test_factor_combo_live_trading_setting_is_saved(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "slots.db"
    _auto_trade_conn(db_path).close()
    monkeypatch.setattr(auto_trade_service, "get_conn", lambda: _connect(db_path))
    settings = AutoTradeSettings(FACTOR_COMBO_STRATEGY_KEY, True, "BTCUSDT", "10m", 10, 5.0, True)

    updated = auto_trade_service.update_auto_trade_settings(settings)

    assert updated.live_trading_enabled is True
    persisted = auto_trade_service.list_auto_trade_settings()
    by_key = {(item.strategy_key, item.symbol, item.duration): item for item in persisted}
    assert by_key[(FACTOR_COMBO_STRATEGY_KEY, "BTCUSDT", "10m")].live_trading_enabled is True


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


def _auto_trade_conn(path=None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or ":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE auto_trade_strategies (
          strategy_key TEXT NOT NULL,
          symbol TEXT NOT NULL DEFAULT 'BTCUSDT',
          duration TEXT NOT NULL DEFAULT '10m',
          enabled INTEGER NOT NULL DEFAULT 0,
          live_trading_enabled INTEGER NOT NULL DEFAULT 0,
          duration_minutes INTEGER NOT NULL DEFAULT 10,
          qty REAL NOT NULL DEFAULT 5,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (strategy_key, symbol, duration)
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
          strategy_key, symbol, duration, enabled, live_trading_enabled, duration_minutes, qty, updated_at
        )
        VALUES(?, 'BTCUSDT', ?, ?, ?, ?, 5, '2026-01-01T00:00:00+00:00')
        """,
        (strategy_key, duration, enabled, live, DURATION_TO_MINUTES[duration]),
    )
    conn.commit()


def _insert_dynamic_strategy(conn: sqlite3.Connection, strategy_key: str, symbol: str, duration: str) -> None:
    conn.execute(
        """
        INSERT INTO auto_trade_strategies(
          strategy_key, symbol, duration, enabled, live_trading_enabled, duration_minutes, qty, updated_at
        )
        VALUES(?, ?, ?, 1, 1, ?, 5, '2026-01-01T00:00:00+00:00')
        """,
        (strategy_key, symbol, duration, DURATION_TO_MINUTES[duration]),
    )
    conn.commit()


def _strategy_count(conn: sqlite3.Connection, strategy_key: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS total FROM auto_trade_strategies WHERE strategy_key = ?",
        (strategy_key,),
    ).fetchone()
    return int(row["total"])


def _strategy_live_enabled(conn: sqlite3.Connection, strategy_key: str, symbol: str) -> int:
    row = conn.execute(
        "SELECT live_trading_enabled FROM auto_trade_strategies WHERE strategy_key = ? AND symbol = ?",
        (strategy_key, symbol),
    ).fetchone()
    return int(row["live_trading_enabled"])


def _connect(path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _strategy_row(conn: sqlite3.Connection, strategy_key: str, symbol: str, duration: str) -> sqlite3.Row:
    return conn.execute(
        """
        SELECT *
        FROM auto_trade_strategies
        WHERE strategy_key = ? AND symbol = ? AND duration = ?
        """,
        (strategy_key, symbol, duration),
    ).fetchone()


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


def _auto_trade_prediction(symbol: str, open_time: int, probability_up: float) -> dict:
    return {
        "id": 1,
        "strategy_key": FACTOR_COMBO_STRATEGY_KEY,
        "symbol": symbol,
        "duration": "10m",
        "open_time": open_time,
        "probability_up": probability_up,
        "direction": "up" if probability_up >= 0.5 else "down",
        "trade_quality_score": 0.8,
        "trade_quality_passed": 1,
    }


def _allowed_regime_gate(**_kwargs):
    return _RegimeDecision(True, "trend_up_aligned", "trend")


def _blocked_regime_gate(**_kwargs):
    return _RegimeDecision(False, "counter_trend_down_vs_up", "skip")


class _RegimeDecision:
    def __init__(self, allowed: bool, reason: str, mode: str) -> None:
        self.allowed = allowed
        self.reason = reason
        self.mode = mode
        self.regime = {"ready": True, "trendState": "trend_up"}
