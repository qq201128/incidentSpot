from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from tempfile import gettempdir

import pytest

from app.services import paper_live_candidate_live_control as control
from app.services import paper_live_candidate_service as report_service
from app.services.factor_candidate_signal_keys import factor_candidate_signal_key


def test_stable_candidate_live_trading_can_be_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("paper-live-control") / "candidates.db"
    candidate_key = factor_candidate_signal_key("alpha")
    _create_db(db_path)
    _insert_predictions(db_path, candidate_key, correct_count=30)
    _patch_db(monkeypatch, db_path)

    result = control.set_candidate_live_trading(
        "BTCUSDT",
        "10m",
        candidate_key=candidate_key,
        live_trading_enabled=True,
    )

    slot = _slot_row(db_path, candidate_key)
    candidate = result["report"]["stable"][0]
    assert result["liveTradingEnabled"] is True
    assert slot["enabled"] == 1
    assert slot["live_trading_enabled"] == 1
    assert candidate["liveTradingEnabled"] is True
    assert result["report"]["realTradingEnabled"] is True


def test_non_stable_candidate_live_trading_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("paper-live-control-blocked") / "candidates.db"
    candidate_key = factor_candidate_signal_key("beta")
    _create_db(db_path)
    _insert_predictions(db_path, candidate_key, correct_count=1)
    _patch_db(monkeypatch, db_path)

    with pytest.raises(ValueError, match="candidate is not stable"):
        control.set_candidate_live_trading(
            "BTCUSDT",
            "10m",
            candidate_key=candidate_key,
            live_trading_enabled=True,
        )

    assert _slot_row(db_path, candidate_key) is None


def _patch_db(monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
    monkeypatch.setattr(control, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(report_service, "get_conn", lambda: _connect(db_path))


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
        """
    )
    conn.close()


def _insert_predictions(path: Path, signal_key: str, *, correct_count: int) -> None:
    conn = _connect(path)
    rows = [_prediction_row(signal_key, index, index < correct_count) for index in range(30)]
    conn.executemany(
        """
        INSERT INTO predictions(
          signal_key, strategy_key, symbol, duration, open_time, direction,
          high_winrate_rule, high_winrate_gate_value, data_freshness_status,
          missing_feature_status, actual_return, prediction_correct, settled_at, created_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def _prediction_row(signal_key: str, index: int, correct: bool) -> tuple:
    actual_return = 0.01 if correct else -0.02
    return (
        signal_key,
        signal_key,
        "BTCUSDT",
        "10m",
        index,
        "up",
        signal_key,
        0.7,
        "fresh",
        "complete",
        actual_return,
        int(correct),
        "2026-05-26T00:00:00+00:00",
        "2026-05-26T00:00:00+00:00",
    )


def _slot_row(path: Path, strategy_key: str) -> sqlite3.Row | None:
    conn = _connect(path)
    try:
        return conn.execute(
            "SELECT * FROM auto_trade_strategies WHERE strategy_key = ?",
            (strategy_key,),
        ).fetchone()
    finally:
        conn.close()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _runtime_path(name: str) -> Path:
    path = Path(gettempdir()) / "incidentSpot-pytest-temp" / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
