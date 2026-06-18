from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.services import paper_live_candidate_service as service

FULL_SAMPLE_COUNT = 130


def test_candidate_report_uses_all_settled_samples(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "candidates.db"
    _create_db(db_path)
    _insert_predictions(db_path)
    _insert_event_outcomes(db_path)
    monkeypatch.setattr(service, "get_conn", lambda: _connect(db_path))

    report = service.refresh_paper_live_candidate_states("BTCUSDT", "10m")
    candidate = report["allCandidates"][0]

    assert candidate["candidateKey"] == "factor_bulk"
    assert candidate["paperLiveSampleCount"] == FULL_SAMPLE_COUNT
    assert candidate["metrics"]["paperLiveWindows"]["recent100"]["sampleCount"] == 100


def _create_db(path: Path) -> None:
    conn = _connect(path)
    conn.executescript(
        """
        CREATE TABLE predictions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          signal_key TEXT NOT NULL,
          strategy_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          open_time INTEGER NOT NULL,
          direction TEXT NOT NULL,
          high_winrate_rule TEXT,
          high_winrate_gate_value REAL,
          high_winrate_gate_min REAL,
          model_family TEXT,
          model_version TEXT,
          validation_win_rate REAL,
          feature_window INTEGER,
          model_duration TEXT,
          model_trained_at TEXT,
          oos_win_rate REAL,
          walk_forward_result TEXT,
          recent_rolling_result TEXT,
          data_freshness_status TEXT,
          missing_feature_status TEXT,
          entry_price REAL,
          exit_price REAL,
          actual_return REAL,
          prediction_correct INTEGER,
          settled_at TEXT,
          created_at TEXT NOT NULL
        );
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
        CREATE TABLE events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          strategy_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          event_interval TEXT NOT NULL,
          start_time TEXT NOT NULL,
          end_time TEXT NOT NULL,
          status TEXT NOT NULL,
          result TEXT,
          ai_predicted_direction TEXT,
          ai_prediction_correct INTEGER,
          ai_high_winrate_rule TEXT,
          prediction_open_time INTEGER,
          prediction_id INTEGER,
          market_regime_gate_passed INTEGER
        );
        CREATE TABLE orders (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          event_id INTEGER NOT NULL,
          side TEXT NOT NULL,
          qty REAL NOT NULL,
          price REAL NOT NULL
        );
        """
    )
    conn.close()


def _insert_predictions(path: Path) -> None:
    rows = [_prediction_row(index) for index in range(FULL_SAMPLE_COUNT)]
    conn = _connect(path)
    conn.executemany(
        """
        INSERT INTO predictions(
          signal_key, strategy_key, symbol, duration, open_time, direction,
          high_winrate_rule, high_winrate_gate_value, oos_win_rate,
          walk_forward_result, recent_rolling_result, data_freshness_status,
          missing_feature_status, actual_return, prediction_correct, settled_at, created_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def _prediction_row(index: int) -> tuple:
    correct = index % 2 == 0
    return (
        "factor_bulk",
        "factor_bulk",
        "BTCUSDT",
        "10m",
        index,
        "up",
        "bulk",
        0.70,
        0.60,
        json.dumps({"stabilityScore": 0.5, "oosWinRate": 0.60}),
        json.dumps({"winRate": 0.60}),
        "fresh",
        "complete",
        0.01 if correct else -0.01,
        int(correct),
        "2026-05-26T00:00:00+00:00",
        "2026-05-26T00:00:00+00:00",
    )


def _insert_event_outcomes(path: Path) -> None:
    conn = _connect(path)
    for index in range(FULL_SAMPLE_COUNT):
        correct = index % 2 == 0
        cursor = conn.execute(
            """
            INSERT INTO events(
              strategy_key, symbol, event_interval, start_time, end_time, status,
              result, ai_predicted_direction, ai_prediction_correct,
              ai_high_winrate_rule, prediction_open_time, prediction_id, market_regime_gate_passed
            )
            VALUES('factor_bulk', 'BTCUSDT', '10m', ?, ?, 'SETTLED', ?, 'up', ?, 'bulk', ?, ?, 1)
            """,
            (
                f"2026-05-26T00:{index % 60:02d}:00+00:00",
                f"2026-05-26T00:{index % 60:02d}:30+00:00",
                "YES" if correct else "NO",
                int(correct),
                index,
                index + 1,
            ),
        )
        conn.execute(
            "INSERT INTO orders(event_id, side, qty, price) VALUES(?, 'BUY', 5.0, 0.8)",
            (cursor.lastrowid,),
        )
    conn.commit()
    conn.close()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn
