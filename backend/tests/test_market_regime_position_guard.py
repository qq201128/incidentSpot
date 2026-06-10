from __future__ import annotations

import sqlite3
from uuid import uuid4

from app.api import event_quick_trade
from app.api.events_models import EventCreate, OrderCreate, QuickTradeCreate
from app.services.event_search_index import ensure_event_search_index
from app.services.position_guard import has_open_position
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY


def test_position_guard_can_ignore_pre_market_regime_open_events() -> None:
    conn = _conn()
    _insert_open_event(conn, event_id=1, gate_passed=None)

    assert has_open_position(conn, "btcusdt", FACTOR_COMBO_STRATEGY_KEY, event_interval="10m") is True
    assert (
        has_open_position(
            conn,
            "btcusdt",
            FACTOR_COMBO_STRATEGY_KEY,
            event_interval="10m",
            require_market_regime_gate_passed=True,
        )
        is False
    )

    _insert_open_event(conn, event_id=2, gate_passed=1)

    assert (
        has_open_position(
            conn,
            "btcusdt",
            FACTOR_COMBO_STRATEGY_KEY,
            event_interval="10m",
            require_market_regime_gate_passed=True,
        )
        is True
    )


def test_strategy_quick_trade_ignores_pre_market_regime_open_event(monkeypatch) -> None:
    db_uri = f"file:market_regime_position_guard_{uuid4().hex}?mode=memory&cache=shared"
    keeper = _connect(db_uri)
    _create_quick_trade_schema(keeper)
    _insert_open_event(keeper, event_id=1, gate_passed=None)
    monkeypatch.setattr(event_quick_trade, "get_conn", lambda: _connect(db_uri))
    monkeypatch.setattr(event_quick_trade, "evaluate_market_regime_trade_gate", _allowed_regime_gate)

    result = event_quick_trade.create_quick_trade_record(
        event_quick_trade.QuickTradeContext(
            payload=_quick_trade_payload(),
            strategy_key=FACTOR_COMBO_STRATEGY_KEY,
            symbol="BTCUSDT",
            side="BUY",
            event_interval="10m",
            rule_type="ABOVE",
            predicted="up",
            entry_price=100.0,
            live_trading_enabled=False,
            prediction_open_time=1_700_000_000_000,
        )
    )

    row = keeper.execute("SELECT market_regime_gate_passed FROM events WHERE id = ?", (result["eventId"],)).fetchone()
    assert row["market_regime_gate_passed"] == 1


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _create_events_schema(conn)
    return conn


def _connect(db_uri: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _create_quick_trade_schema(conn: sqlite3.Connection) -> None:
    _create_events_schema(conn)
    conn.execute(
        """
        CREATE TABLE orders(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          event_id INTEGER NOT NULL,
          side TEXT NOT NULL,
          price REAL NOT NULL,
          qty REAL NOT NULL,
          status TEXT NOT NULL,
          created_at TEXT NOT NULL,
          external_order_id TEXT,
          external_status TEXT,
          external_response TEXT
        )
        """
    )
    ensure_event_search_index(conn)
    conn.commit()


def _create_events_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE events(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          strategy_key TEXT,
          symbol TEXT NOT NULL,
          title TEXT NOT NULL,
          event_interval TEXT,
          rule_type TEXT,
          strike_value REAL,
          upper_bound REAL,
          start_time TEXT,
          end_time TEXT,
          status TEXT,
          result TEXT,
          settlement_source TEXT,
          prediction_open_time INTEGER,
          ai_probability_up REAL,
          ai_predicted_direction TEXT,
          ai_quality_score REAL,
          ai_quality_passed INTEGER,
          ai_high_winrate_gate TEXT,
          ai_high_winrate_rule TEXT,
          ai_high_winrate_passed INTEGER,
          ai_high_winrate_value REAL,
          market_regime_gate_version TEXT,
          market_regime_gate_passed INTEGER,
          market_regime_gate_reason TEXT,
          market_regime_gate_mode TEXT,
          market_regime_label TEXT
        )
        """
    )


def _insert_open_event(conn: sqlite3.Connection, *, event_id: int, gate_passed: int | None) -> None:
    conn.execute(
        """
        INSERT INTO events(
          id, strategy_key, symbol, title, event_interval, rule_type, strike_value,
          start_time, end_time, status, market_regime_gate_passed
        )
        VALUES(
          ?, ?, 'BTCUSDT', 'test', '10m', 'ABOVE', 100.0,
          '2026-05-13T00:00:00+00:00', '2026-05-13T00:10:00+00:00', 'OPEN', ?
        )
        """,
        (event_id, FACTOR_COMBO_STRATEGY_KEY, gate_passed),
    )
    conn.commit()


def _quick_trade_payload() -> QuickTradeCreate:
    return QuickTradeCreate(
        liveTradingEnabled=False,
        event=EventCreate(
            strategyKey=FACTOR_COMBO_STRATEGY_KEY,
            symbol="BTCUSDT",
            title="strategy quick trade",
            eventInterval="10m",
            ruleType="ABOVE",
            strikeValue=100.0,
            endTime="2026-05-31T00:10:00+00:00",
            aiProbabilityUp=0.8,
            aiPredictedDirection="up",
            aiQualityScore=0.9,
            aiQualityPassed=True,
            aiHighWinratePassed=True,
        ),
        order=OrderCreate(side="BUY", qty=1.0, price=0.5),
    )


class _RegimeDecision:
    allowed = True
    reason = "trend_up_aligned"
    mode = "trend"
    regime = {"ready": True, "trendState": "trend_up", "regimeLabel": "trend_up:normal_vol"}


def _allowed_regime_gate(**_kwargs):
    return _RegimeDecision()
