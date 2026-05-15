from __future__ import annotations

import sqlite3
from pathlib import Path

from app.services import high_winrate_strategy_demotion as demotion
from app.services.rule_config import DURATION_TO_MINUTES
from app.services.strategy_registry import HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY


def test_promotion_enables_simulation_slot(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "promotion.db"
    _init_db(db_path)
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))

    result = demotion.promote_high_winrate_strategy("btcusdt", "10m")

    row = _slot(db_path, "10m")
    assert result["status"] == demotion.STATUS_COLLECTING
    assert row["enabled"] == 1
    assert row["live_trading_enabled"] == 0


def test_active_after_live_samples_hit_target(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "active.db"
    _init_db(db_path)
    _insert_slot(db_path, "10m", enabled=1, live=1)
    _insert_predictions(db_path, "10m", [True] * demotion.ACTIVE_SAMPLE_COUNT)
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))

    result = demotion.evaluate_high_winrate_demotion("BTCUSDT", "10m")

    row = _slot(db_path, "10m")
    assert result["status"] == demotion.STATUS_ACTIVE
    assert result["reason"] == "stable_live_target_met"
    assert row["enabled"] == 1
    assert row["live_trading_enabled"] == 1


def test_demotion_disables_live_but_keeps_simulation_on_loss_streak(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "loss-streak.db"
    _init_db(db_path)
    _insert_slot(db_path, "10m", enabled=1, live=1)
    _insert_predictions(db_path, "10m", [False, False, False, False, False])
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))

    result = demotion.evaluate_high_winrate_demotion("BTCUSDT", "10m")

    row = _slot(db_path, "10m")
    assert result["status"] == demotion.STATUS_DEMOTED
    assert result["reason"] == "consecutive_losses"
    assert row["enabled"] == 1
    assert row["live_trading_enabled"] == 0


def test_demotion_below_live_target_keeps_collecting_predictions(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "below-target.db"
    _init_db(db_path)
    _insert_slot(db_path, "30m", enabled=1, live=1)
    _insert_predictions(db_path, "30m", ([True] * 2 + [False]) * 6 + [True, False])
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))

    result = demotion.evaluate_high_winrate_demotion("BTCUSDT", "30m")

    row = _slot(db_path, "30m")
    assert result["status"] == demotion.STATUS_DEMOTED
    assert result["reason"] == "live_win_rate_below_target"
    assert row["enabled"] == 1
    assert row["live_trading_enabled"] == 0


def test_paused_status_requires_new_promotion_to_clear(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "sticky-paused.db"
    _init_db(db_path)
    _insert_slot(db_path, "10m", enabled=1, live=0)
    _insert_status(db_path, "10m", demotion.STATUS_PAUSED, "manual_test_pause")
    _insert_predictions(db_path, "10m", [True] * demotion.ACTIVE_SAMPLE_COUNT)
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))

    result = demotion.evaluate_high_winrate_demotion("BTCUSDT", "10m")

    assert result["status"] == demotion.STATUS_PAUSED
    assert result["reason"] == "manual_test_pause"


def _init_db(path: Path) -> None:
    conn = _connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE auto_trade_strategies (
              strategy_key TEXT NOT NULL,
              duration TEXT NOT NULL,
              enabled INTEGER NOT NULL,
              live_trading_enabled INTEGER NOT NULL,
              symbol TEXT NOT NULL,
              duration_minutes INTEGER NOT NULL,
              qty REAL NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (strategy_key, duration)
            );
            CREATE TABLE predictions (
              strategy_key TEXT NOT NULL,
              symbol TEXT NOT NULL,
              duration TEXT NOT NULL,
              open_time INTEGER NOT NULL,
              prediction_correct INTEGER,
              actual_return REAL,
              high_winrate_rule TEXT,
              settled_at TEXT
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def _insert_slot(path: Path, duration: str, *, enabled: int, live: int) -> None:
    conn = _connect(path)
    try:
        conn.execute(
            """
            INSERT INTO auto_trade_strategies
            VALUES(?, ?, ?, ?, 'BTCUSDT', ?, 5.0, 'now')
            """,
            (
                HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
                duration,
                enabled,
                live,
                int(DURATION_TO_MINUTES[duration]),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_predictions(path: Path, duration: str, outcomes: list[bool]) -> None:
    conn = _connect(path)
    try:
        for index, correct in enumerate(outcomes):
            conn.execute(
                """
                INSERT INTO predictions
                VALUES(?, 'BTCUSDT', ?, ?, ?, ?, 'goal_combo__test', 'done')
                """,
                (
                    HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
                    duration,
                    index,
                    int(correct),
                    0.01 if correct else -0.01,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _insert_status(path: Path, duration: str, status: str, reason: str) -> None:
    conn = _connect(path)
    try:
        demotion._ensure_table(conn)
        conn.execute(
            """
            INSERT INTO high_winrate_strategy_status
            VALUES(?, 'BTCUSDT', ?, ?, ?, '{}', 0, NULL, NULL, 0, 'now', 'now')
            """,
            (HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, duration, status, reason),
        )
        conn.commit()
    finally:
        conn.close()


def _slot(path: Path, duration: str) -> sqlite3.Row:
    conn = _connect(path)
    try:
        return conn.execute(
            """
            SELECT * FROM auto_trade_strategies
            WHERE strategy_key = ? AND duration = ?
            """,
            (HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, duration),
        ).fetchone()
    finally:
        conn.close()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn
