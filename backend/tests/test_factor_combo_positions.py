from __future__ import annotations

import sqlite3

from app.services.factor_combo_simulation_keys import factor_combo_shadow_strategy_key
from app.services.factor_combo_position_service import factor_combo_positions_payload
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY


def test_factor_combo_positions_filters_to_selected_combo() -> None:
    conn = _conn()
    _insert_event(
        conn, event_id=1, duration="10m", status="OPEN", rule="combo__a__b", pnl=None
    )
    _insert_event(
        conn, event_id=2, duration="10m", status="SETTLED", rule="combo__old", pnl=4.0
    )
    _insert_event(
        conn, event_id=3, duration="30m", status="OPEN", rule="combo__a__b", pnl=None
    )
    _insert_event(
        conn, event_id=4, duration="10m", status="SETTLED", rule="combo__a__b", pnl=2.5
    )

    payload = factor_combo_positions_payload(
        conn,
        symbol="btcusdt",
        duration="10m",
        factor_name="combo__a__b",
    )

    assert payload["total"] == 2
    assert payload["openCount"] == 1
    assert payload["settledCount"] == 1
    assert payload["currentFactorCount"] == 2
    assert payload["totalPnl"] == 2.5
    assert [event["id"] for event in payload["events"]] == [4, 1]
    assert {event["aiHighWinrateRule"] for event in payload["events"]} == {"combo__a__b"}


def test_factor_combo_positions_reads_shadow_strategy_keys() -> None:
    conn = _conn()
    _insert_event(
        conn,
        event_id=1,
        duration="10m",
        status="OPEN",
        rule="combo__top_2",
        pnl=None,
        strategy_key=factor_combo_shadow_strategy_key(2),
    )

    payload = factor_combo_positions_payload(
        conn,
        symbol="BTCUSDT",
        duration="10m",
        factor_name="combo__top_2",
    )

    assert payload["total"] == 1
    assert payload["events"][0]["strategyKey"] == factor_combo_shadow_strategy_key(2)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE events (
          id INTEGER PRIMARY KEY,
          strategy_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          title TEXT NOT NULL,
          event_interval TEXT NOT NULL,
          rule_type TEXT NOT NULL,
          strike_value REAL NOT NULL,
          upper_bound REAL,
          start_time TEXT NOT NULL,
          end_time TEXT NOT NULL,
          status TEXT NOT NULL,
          result TEXT,
          settlement_price REAL,
          settlement_quote_time INTEGER,
          settlement_source TEXT,
          ai_probability_up REAL,
          ai_predicted_direction TEXT,
          ai_prediction_correct INTEGER,
          ai_quality_score REAL,
          ai_quality_passed INTEGER,
          ai_high_winrate_gate TEXT,
          ai_high_winrate_rule TEXT,
          ai_high_winrate_passed INTEGER,
          ai_high_winrate_value REAL
        );
        CREATE TABLE orders (
          id INTEGER PRIMARY KEY,
          event_id INTEGER NOT NULL,
          side TEXT NOT NULL,
          price REAL NOT NULL,
          qty REAL NOT NULL,
          status TEXT NOT NULL,
          created_at TEXT NOT NULL,
          external_order_id TEXT,
          external_status TEXT,
          external_response TEXT
        );
        CREATE TABLE settlements (event_id INTEGER NOT NULL, pnl REAL NOT NULL);
        """
    )
    return conn


def _insert_event(
    conn: sqlite3.Connection,
    *,
    event_id: int,
    duration: str,
    status: str,
    rule: str,
    pnl: float | None,
    strategy_key: str = FACTOR_COMBO_STRATEGY_KEY,
) -> None:
    conn.execute(
        """
        INSERT INTO events VALUES(
          ?, ?, 'BTCUSDT', 'test', ?, 'ABOVE', 100.0, NULL,
          '2026-05-13T00:00:00+00:00', '2026-05-13T00:10:00+00:00', ?,
          NULL, NULL, NULL, NULL, 0.6, 'up', NULL, 0.6, 1, NULL, ?, NULL, NULL
        )
        """,
        (event_id, strategy_key, duration, status, rule),
    )
    conn.execute(
        """
        INSERT INTO orders VALUES(
          ?, ?, 'BUY', 0.8, 5.0, 'OPEN', '2026-05-13T00:00:00+00:00', NULL, 'SIMULATED', NULL
        )
        """,
        (event_id, event_id),
    )
    if pnl is not None:
        conn.execute("INSERT INTO settlements VALUES(?, ?)", (event_id, pnl))
