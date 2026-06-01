from __future__ import annotations

import sqlite3
from pathlib import Path

from app.services import high_winrate_strategy_demotion as demotion
from app.services.batch_combo_event_demotion import evaluate_batch_combo_event_demotion
from app.services.high_winrate_strategy_metrics import ACTIVE_SAMPLE_COUNT
from app.services.high_winrate_strategy_rotation import ensure_high_winrate_status_table
from app.services.model_family_config import model_family_strategy_key
from app.services.rule_config import DURATION_TO_MINUTES
from app.services.shadow_event_deviation_service import shadow_event_deviation_report
from app.services.strategy_registry import HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY

BATCH_KEY = "high_winrate_factor_combo_v1_combo_deadbeef0001"


def test_shadow_event_deviation_flags_prediction_win_event_loss(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "deviation.db"
    _init_db(db_path)
    for index in range(5):
        open_time = 1_000 + index
        _insert_prediction(db_path, open_time=open_time, correct=True, actual_return=0.02)
        _insert_event(db_path, strategy_key=BATCH_KEY, open_time=open_time, direction="up", result="NO", side="BUY")
    monkeypatch.setattr("app.db.session.get_conn", lambda: _connect(db_path))

    report = shadow_event_deviation_report("BTCUSDT", "10m")

    assert report["summary"]["pairedCount"] == 5
    assert report["summary"]["shadowWinEventLossCount"] == 5
    assert "issues" not in report
    assert report["byStrategy"][0]["shadowWinEventLossCount"] == 5


def test_high_winrate_demotion_prefers_event_pnl_rows(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "event-demote.db"
    _init_db(db_path)
    _insert_slot(db_path, "10m")
    for index in range(ACTIVE_SAMPLE_COUNT):
        _insert_event(
            db_path,
            strategy_key=HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
            open_time=10_000 + index,
            direction="up",
            result="NO",
            side="BUY",
            rule="goal_combo__top1",
        )
    monkeypatch.setattr(demotion, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr("app.db.session.get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(demotion, "high_winrate_candidate_rule", lambda *_args: None)

    result = demotion.evaluate_high_winrate_demotion("BTCUSDT", "10m")

    assert result["metricsSource"] == "events"
    assert result["sampleCount"] == ACTIVE_SAMPLE_COUNT
    assert result["status"] == demotion.STATUS_DEMOTED
    assert result["reason"] == "consecutive_losses"


def test_batch_combo_event_demotion_marks_bad_strategy_without_disabling(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "batch-demote.db"
    _init_db(db_path)
    _insert_slot(db_path, "10m", strategy_key=BATCH_KEY, enabled=1)
    for index in range(ACTIVE_SAMPLE_COUNT):
        _insert_event(
            db_path,
            strategy_key=BATCH_KEY,
            open_time=20_000 + index,
            direction="up",
            result="NO",
            side="BUY",
            rule="goal_combo__alpha",
        )
    monkeypatch.setattr("app.db.session.get_conn", lambda: _connect(db_path))

    report = evaluate_batch_combo_event_demotion("BTCUSDT", "10m")
    row = _slot(db_path, "10m", BATCH_KEY)

    assert report["observeOnly"] is True
    assert report["watchlistCount"] == 1
    assert report["evaluations"][0]["status"] == "demoted"
    assert report["evaluations"][0]["autoTradeAction"] == "none"
    assert row["enabled"] == 1


FACTOR_CANDIDATE_KEY = "factor_candidate_signal_deadbeef0002"


def test_factor_candidate_event_demotion_marks_bad_strategy_without_disabling(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "single-demote.db"
    _init_db(db_path)
    _insert_slot(db_path, "10m", strategy_key=FACTOR_CANDIDATE_KEY, enabled=1)
    for index in range(ACTIVE_SAMPLE_COUNT):
        _insert_event(
            db_path,
            strategy_key=FACTOR_CANDIDATE_KEY,
            open_time=30_000 + index,
            direction="up",
            result="NO",
            side="BUY",
            rule="rsi_14",
        )
    monkeypatch.setattr("app.db.session.get_conn", lambda: _connect(db_path))

    from app.services.factor_candidate_event_demotion import evaluate_factor_candidate_event_demotion

    report = evaluate_factor_candidate_event_demotion("BTCUSDT", "10m")
    row = _slot(db_path, "10m", FACTOR_CANDIDATE_KEY)

    assert report["source"] == "factor_candidate"
    assert report["observeOnly"] is True
    assert report["watchlistCount"] == 1
    assert report["evaluations"][0]["status"] == "demoted"
    assert report["evaluations"][0]["displayRule"] == "rsi_14"
    assert row["enabled"] == 1


def test_combo_event_monitoring_includes_single_and_batch_sections(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "monitoring.db"
    _init_db(db_path)
    monkeypatch.setattr("app.db.session.get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(
        "app.services.combo_event_governance.shadow_event_deviation_report",
        lambda *_args, **_kwargs: {"summary": {"pairedCount": 0}},
    )

    from app.services.combo_event_governance import compute_combo_event_monitoring

    payload = compute_combo_event_monitoring("BTCUSDT", "10m")

    assert "batchComboDemotion" in payload
    assert "factorCandidateDemotion" in payload
    assert "modelShadowSimulation" in payload
    assert payload["simulationObservation"]["evaluatedCount"] == 0
    assert payload["simulationObservation"]["modelShadowEventCount"] == 0


def test_combo_event_monitoring_reports_model_shadow_simulation(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "model-shadow.db"
    model_key = model_family_strategy_key("lstm", "10m")
    _init_db(db_path)
    _insert_model_prediction(db_path, strategy_key=model_key)
    _insert_event(db_path, strategy_key=model_key, open_time=40_000, direction="up", result="YES", side="BUY")
    monkeypatch.setattr("app.db.session.get_conn", lambda: _connect(db_path))
    monkeypatch.setattr("app.services.model_shadow_simulation_monitor.get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(
        "app.services.combo_event_governance.shadow_event_deviation_report",
        lambda *_args, **_kwargs: {"summary": {"pairedCount": 0}},
    )

    from app.services.combo_event_governance import compute_combo_event_monitoring

    payload = compute_combo_event_monitoring("BTCUSDT", "10m")
    model = payload["modelShadowSimulation"]

    assert model["summary"]["qualityPassedCount"] == 1
    assert model["summary"]["simulationEventCount"] == 1
    assert payload["simulationObservation"]["modelShadowEventCount"] == 1


def _init_db(path: Path) -> None:
    conn = _connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE auto_trade_strategies (
              strategy_key TEXT NOT NULL,
              symbol TEXT NOT NULL,
              duration TEXT NOT NULL,
              enabled INTEGER NOT NULL,
              live_trading_enabled INTEGER NOT NULL,
              duration_minutes INTEGER NOT NULL,
              qty REAL NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (strategy_key, symbol, duration)
            );
            CREATE TABLE predictions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              signal_key TEXT NOT NULL,
              strategy_key TEXT NOT NULL,
              symbol TEXT NOT NULL,
              duration TEXT NOT NULL,
              open_time INTEGER NOT NULL,
              direction TEXT NOT NULL,
              trade_quality_passed INTEGER,
              prediction_correct INTEGER,
              actual_return REAL,
              settled_at TEXT,
              created_at TEXT
            );
            CREATE TABLE events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              strategy_key TEXT NOT NULL,
              symbol TEXT NOT NULL,
              title TEXT NOT NULL,
              event_interval TEXT NOT NULL,
              rule_type TEXT NOT NULL,
              strike_value REAL NOT NULL,
              start_time TEXT NOT NULL,
              end_time TEXT NOT NULL,
              status TEXT NOT NULL,
              result TEXT,
              prediction_open_time INTEGER,
              ai_predicted_direction TEXT,
              ai_prediction_correct INTEGER,
              ai_high_winrate_rule TEXT
            );
            CREATE TABLE orders (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              event_id INTEGER NOT NULL,
              side TEXT NOT NULL,
              price REAL NOT NULL,
              qty REAL NOT NULL,
              status TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            """
        )
        ensure_high_winrate_status_table(conn)
        conn.commit()
    finally:
        conn.close()


def _insert_prediction(path: Path, *, open_time: int, correct: bool, actual_return: float) -> None:
    conn = _connect(path)
    try:
        conn.execute(
            """
            INSERT INTO predictions(
              signal_key, strategy_key, symbol, duration, open_time, direction,
              prediction_correct, actual_return, settled_at
            )
            VALUES(?, ?, 'BTCUSDT', '10m', ?, 'up', ?, ?, 'done')
            """,
            (BATCH_KEY, BATCH_KEY, open_time, int(correct), actual_return),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_model_prediction(path: Path, *, strategy_key: str) -> None:
    conn = _connect(path)
    try:
        conn.execute(
            """
            INSERT INTO predictions(
              signal_key, strategy_key, symbol, duration, open_time, direction,
              trade_quality_passed, prediction_correct, actual_return, settled_at, created_at
            )
            VALUES(?, ?, 'BTCUSDT', '10m', 40000, 'up', 1, 1, 0.01, 'done', 'now')
            """,
            (strategy_key, strategy_key),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_event(
    path: Path,
    *,
    strategy_key: str,
    open_time: int,
    direction: str,
    result: str,
    side: str,
    rule: str = "goal_combo__alpha",
) -> None:
    conn = _connect(path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO events(
              strategy_key, symbol, title, event_interval, rule_type, strike_value,
              start_time, end_time, status, result, prediction_open_time,
              ai_predicted_direction, ai_prediction_correct, ai_high_winrate_rule
            )
            VALUES(?, 'BTCUSDT', 'test', '10m', 'ABOVE', 100.0, '2026-01-01T00:00:00+00:00',
                   '2026-01-01T00:10:00+00:00', 'SETTLED', ?, ?, ?, ?, ?)
            """,
            (
                strategy_key,
                result,
                open_time,
                direction,
                1 if (direction == "up" and result == "YES") or (direction == "down" and result == "NO") else 0,
                rule,
            ),
        )
        conn.execute(
            """
            INSERT INTO orders(event_id, side, price, qty, status, created_at)
            VALUES(?, ?, 0.8, 5.0, 'SETTLED', 'now')
            """,
            (cursor.lastrowid, side),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_slot(path: Path, duration: str, strategy_key: str = HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, *, enabled: int = 1) -> None:
    conn = _connect(path)
    try:
        conn.execute(
            """
            INSERT INTO auto_trade_strategies
            VALUES(?, 'BTCUSDT', ?, ?, 0, ?, 5.0, 'now')
            """,
            (strategy_key, duration, enabled, int(DURATION_TO_MINUTES[duration])),
        )
        conn.commit()
    finally:
        conn.close()


def _slot(path: Path, duration: str, strategy_key: str) -> sqlite3.Row:
    conn = _connect(path)
    try:
        return conn.execute(
            "SELECT * FROM auto_trade_strategies WHERE strategy_key = ? AND symbol = ? AND duration = ?",
            (strategy_key, "BTCUSDT", duration),
        ).fetchone()
    finally:
        conn.close()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn
