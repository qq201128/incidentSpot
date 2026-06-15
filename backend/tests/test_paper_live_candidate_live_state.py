from __future__ import annotations

import sqlite3
from pathlib import Path

from app.services import paper_live_candidate_live_state as service


def test_live_trading_overview_groups_enabled_candidates(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "live-state.db"
    _create_db(db_path)
    _insert_live_rows(db_path)
    monkeypatch.setattr(service, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(
        service,
        "_runtime_status_by_slot",
        lambda: {
            ("factor_alpha", "BTCUSDT", "10m"): {
                "reason": "ready_to_place_order",
                "latestPrediction": {"createdAt": "2026-06-12T00:05:00+00:00", "fresh": True, "ageMs": 0},
            },
            ("factor_beta", "ETHUSDT", "30m"): {
                "reason": "waiting_fresh_prediction",
                "latestPrediction": {"createdAt": "2026-06-12T00:06:00+00:00", "fresh": False, "ageMs": 60000},
            },
        },
    )

    overview = service.live_trading_overview()

    assert overview["activeCount"] == 2
    assert [(row["symbol"], row["duration"], row["activeCount"]) for row in overview["groups"]] == [
        ("BTCUSDT", "10m", 1),
        ("ETHUSDT", "30m", 1),
    ]
    btc = overview["groups"][0]["candidates"][0]
    assert btc["candidateName"] == "alpha"
    assert btc["candidateKey"] == "factor_alpha"
    assert btc["liveTradingEnabled"] is True
    assert btc["runtimeReason"] == "ready_to_place_order"
    assert btc["latestPredictionFresh"] is True
    eth = overview["groups"][1]["candidates"][0]
    assert eth["runtimeReason"] == "waiting_fresh_prediction"
    assert eth["latestPredictionFresh"] is False


def _create_db(path: Path) -> None:
    conn = _connect(path)
    conn.executescript(
        """
        CREATE TABLE auto_trade_strategies (
          strategy_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          enabled INTEGER NOT NULL DEFAULT 0,
          live_trading_enabled INTEGER NOT NULL DEFAULT 0,
          duration_minutes INTEGER NOT NULL DEFAULT 10,
          qty REAL NOT NULL DEFAULT 5.0,
          updated_at TEXT NOT NULL,
          PRIMARY KEY(strategy_key, symbol, duration)
        );
        CREATE TABLE predictions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          signal_key TEXT NOT NULL,
          strategy_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          open_time INTEGER NOT NULL,
          high_winrate_rule TEXT,
          model_family TEXT,
          model_version TEXT
        );
        """
    )
    conn.close()


def _insert_live_rows(path: Path) -> None:
    conn = _connect(path)
    conn.executemany(
        """
        INSERT INTO auto_trade_strategies(
          strategy_key, symbol, duration, enabled, live_trading_enabled,
          duration_minutes, qty, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("factor_alpha", "BTCUSDT", "10m", 1, 1, 10, 5.0, "2026-06-12T00:00:00+00:00"),
            ("factor_beta", "ETHUSDT", "30m", 1, 1, 30, 8.0, "2026-06-12T00:01:00+00:00"),
            ("factor_disabled", "BTCUSDT", "60m", 1, 0, 60, 5.0, "2026-06-12T00:02:00+00:00"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO predictions(signal_key, strategy_key, symbol, duration, open_time, high_winrate_rule)
        VALUES(?, ?, ?, ?, ?, ?)
        """,
        [
            ("factor_alpha", "factor_alpha", "BTCUSDT", "10m", 10, "alpha"),
            ("factor_beta", "factor_beta", "ETHUSDT", "30m", 20, "beta"),
        ],
    )
    conn.commit()
    conn.close()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn
