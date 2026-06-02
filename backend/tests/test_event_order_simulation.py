from __future__ import annotations

import json
import sqlite3
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api import event_quick_trade
from app.api.events import create_quick_trade
from app.api.event_writes import _insert_order
from app.api.events_models import EventCreate, OrderCreate, QuickTradeCreate
from app.services.event_search_index import ensure_event_search_index


def test_manual_order_insert_is_explicitly_simulated() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
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
          external_status TEXT,
          external_response TEXT
        )
        """
    )

    order_id = _insert_order(conn, 7, "BUY", OrderCreate(side="BUY", qty=1.0, price=0.5))

    row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    response = json.loads(row["external_response"])
    assert row["external_status"] == "SIMULATED"
    assert response["simulation"] is True
    assert "未调用 Binance" in response["message"]


def test_quick_trade_is_simulated_and_does_not_call_binance(monkeypatch: pytest.MonkeyPatch) -> None:
    db_uri = _quick_trade_db_uri()
    keeper = _quick_trade_conn(db_uri)
    monkeypatch.setattr(event_quick_trade, "get_conn", lambda: _connect_quick_trade(db_uri))
    monkeypatch.setattr(event_quick_trade, "has_open_position", lambda *_args, **_kwargs: False)

    result = create_quick_trade(_quick_trade_payload(live=False))

    order = keeper.execute("SELECT * FROM orders WHERE id = ?", (result["orderId"],)).fetchone()
    external = json.loads(order["external_response"])
    assert result["simulated"] is True
    assert result["binanceCalled"] is False
    assert result["externalStatus"] == "SIMULATED"
    assert order["external_status"] == "SIMULATED"
    assert external["response"]["simulation"] is True
    assert "未调用 Binance" in external["response"]["message"]


def test_quick_trade_live_trading_calls_binance_and_writes_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    db_uri = _quick_trade_db_uri()
    keeper = _quick_trade_conn(db_uri)
    placed = []
    monkeypatch.setattr(event_quick_trade, "get_conn", lambda: _connect_quick_trade(db_uri))
    monkeypatch.setattr(event_quick_trade, "has_open_position", lambda *_args, **_kwargs: False)

    def place_order(**kwargs):
        placed.append(kwargs)
        return {"externalOrderId": "live-1", "externalStatus": "PLACED", "response": {"success": True}}

    monkeypatch.setattr(event_quick_trade, "place_event_contract_order", place_order)

    result = create_quick_trade(_quick_trade_payload(live=True))

    order = keeper.execute("SELECT * FROM orders WHERE id = ?", (result["orderId"],)).fetchone()
    external = json.loads(order["external_response"])
    assert placed == [
        {
            "symbol": "BTCUSDT",
            "event_interval": "10m",
            "side": "BUY",
            "amount": 1.0,
            "payout_ratio": 0.8,
        }
    ]
    assert result["simulated"] is False
    assert result["binanceCalled"] is True
    assert result["externalStatus"] == "PLACED"
    assert result["externalOrderId"] == "live-1"
    assert order["external_status"] == "PLACED"
    assert external["externalOrderId"] == "live-1"


def test_quick_trade_live_failure_does_not_write_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    db_uri = _quick_trade_db_uri()
    keeper = _quick_trade_conn(db_uri)
    failures = []
    monkeypatch.setattr(event_quick_trade, "get_conn", lambda: _connect_quick_trade(db_uri))
    monkeypatch.setattr(event_quick_trade, "has_open_position", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(event_quick_trade, "log_live_order_failure", lambda ctx, exc: failures.append((ctx, exc)))

    def place_order(**_kwargs):
        raise RuntimeError("binance order rejected")

    monkeypatch.setattr(event_quick_trade, "place_event_contract_order", place_order)

    with pytest.raises(RuntimeError, match="binance order rejected"):
        create_quick_trade(_quick_trade_payload(live=True))

    assert len(failures) == 1
    assert _row_count(keeper, "events") == 0
    assert _row_count(keeper, "orders") == 0


def _quick_trade_payload(*, live: bool) -> QuickTradeCreate:
    return QuickTradeCreate(
        liveTradingEnabled=live,
        event=EventCreate(
            symbol="BTCUSDT",
            title="manual quick trade",
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


def _quick_trade_db_uri() -> str:
    return f"file:quick_trade_sim_{uuid4().hex}?mode=memory&cache=shared"


def _quick_trade_conn(db_uri: str) -> sqlite3.Connection:
    conn = _connect_quick_trade(db_uri)
    conn.row_factory = sqlite3.Row
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
          ai_high_winrate_value REAL
        )
        """
    )
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
    return conn


def _connect_quick_trade(db_uri: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _row_count(conn: sqlite3.Connection, table: str) -> int:
    if table == "events":
        row = conn.execute("SELECT COUNT(*) AS total FROM events").fetchone()
    elif table == "orders":
        row = conn.execute("SELECT COUNT(*) AS total FROM orders").fetchone()
    else:
        raise ValueError(f"unsupported table: {table}")
    return int(row["total"])
