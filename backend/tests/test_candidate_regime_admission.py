from __future__ import annotations

import sqlite3

import pytest

from app.services import candidate_regime_admission as admission


def test_candidate_regime_admission_allows_exploration_without_bucket_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _db()
    monkeypatch.setattr(admission, "get_conn", lambda: conn)
    monkeypatch.setattr(admission, "market_regime_status", _ready_regime)

    decision = admission.evaluate_candidate_regime_admission(_prediction(direction="up"))

    assert decision.allowed is True
    assert decision.mode == "exploration"
    assert decision.reason == "regime_exploration_sample_count_below_50"
    assert decision.sample_count == 0


def test_candidate_regime_admission_keeps_collecting_until_120_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _db()
    _insert_bucket(conn, total=119, wins=40)
    monkeypatch.setattr(admission, "get_conn", lambda: conn)
    monkeypatch.setattr(admission, "market_regime_status", _ready_regime)

    decision = admission.evaluate_candidate_regime_admission(_prediction(direction="up"))

    assert decision.allowed is True
    assert decision.mode == "collecting"
    assert decision.reason == "regime_collecting_sample_count_below_120"
    assert decision.sample_count == 119


def test_candidate_regime_admission_blocks_failed_evaluable_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _db()
    _insert_bucket(conn, total=120, wins=60)
    monkeypatch.setattr(admission, "get_conn", lambda: conn)
    monkeypatch.setattr(admission, "market_regime_status", _ready_regime)

    decision = admission.evaluate_candidate_regime_admission(_prediction(direction="up"))

    assert decision.allowed is False
    assert decision.mode == "evaluable"
    assert decision.reason == "regime_bucket_win_rate_below_min"
    assert decision.metrics["winRate"] == 0.5


def test_candidate_regime_admission_passes_evaluable_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _db()
    _insert_bucket(conn, total=120, wins=80)
    monkeypatch.setattr(admission, "get_conn", lambda: conn)
    monkeypatch.setattr(admission, "market_regime_status", _ready_regime)

    decision = admission.evaluate_candidate_regime_admission(_prediction(direction="up"))

    assert decision.allowed is True
    assert decision.mode == "evaluable"
    assert decision.reason == "regime_bucket_evaluable_passed"
    assert decision.metrics["recent50WinRate"] >= 0.54


def _ready_regime(*_args, **_kwargs) -> dict:
    return {
        "ready": True,
        "trendState": "trend_down",
        "volatilityState": "normal_vol",
        "regimeLabel": "trend_down:normal_vol",
    }


def _prediction(*, direction: str) -> dict:
    return {
        "signal_key": "combo_alpha_signal",
        "strategy_key": "combo_alpha_strategy",
        "high_winrate_rule": "combo_alpha",
        "symbol": "BTCUSDT",
        "duration": "10m",
        "open_time": 1_700_000_000_000,
        "direction": direction,
    }


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
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
          model_version TEXT
        );
        CREATE TABLE events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          strategy_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          event_interval TEXT NOT NULL,
          start_time TEXT NOT NULL,
          status TEXT NOT NULL,
          result TEXT,
          ai_predicted_direction TEXT,
          ai_prediction_correct INTEGER,
          prediction_id INTEGER,
          market_regime_gate_passed INTEGER,
          market_regime_label TEXT
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
    return conn


def _insert_bucket(conn: sqlite3.Connection, *, total: int, wins: int) -> None:
    for index in range(total):
        win = index >= total - wins
        prediction_id = conn.execute(
            """
            INSERT INTO predictions(
              signal_key, strategy_key, symbol, duration, open_time, direction,
              high_winrate_rule, model_version
            )
            VALUES('combo_alpha_signal', 'combo_alpha_strategy', 'BTCUSDT', '10m', ?, 'up', 'combo_alpha', NULL)
            """,
            (1_700_000_000_000 + index,),
        ).lastrowid
        event_id = conn.execute(
            """
            INSERT INTO events(
              strategy_key, symbol, event_interval, start_time, status, result,
              ai_predicted_direction, ai_prediction_correct, prediction_id,
              market_regime_gate_passed, market_regime_label
            )
            VALUES(
              'combo_alpha_strategy', 'BTCUSDT', '10m', ?, 'SETTLED', ?,
              'up', ?, ?, 1, 'trend_down:normal_vol'
            )
            """,
            (f"2026-06-01T00:{index % 60:02d}:00+00:00", "YES" if win else "NO", int(win), prediction_id),
        ).lastrowid
        conn.execute(
            "INSERT INTO orders(event_id, side, qty, price) VALUES(?, 'BUY', 5.0, 0.8)",
            (event_id,),
        )
    conn.commit()
