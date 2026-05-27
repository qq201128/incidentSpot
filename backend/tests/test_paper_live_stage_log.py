from __future__ import annotations

import sqlite3
from pathlib import Path

from app.services import forward_validation_service, prediction_cache_service


def test_save_prediction_records_generation_stages(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "generation.db"
    _init_prediction_db(db_path)
    monkeypatch.setattr(prediction_cache_service, "get_conn", lambda: _connect(db_path))

    saved = prediction_cache_service.save_prediction(_prediction_result())

    logs = _stage_logs(db_path)
    assert saved is True
    assert [row["stage"] for row in logs] == ["feature_construction", "prediction_generation"]
    assert {row["status"] for row in logs} == {"passed"}
    assert logs[0]["reason"] == "features_available_before_entry"
    row = _prediction_row(db_path)
    assert row["oos_win_rate"] == 0.61
    assert '"stabilityScore": 0.7' in row["walk_forward_result"]


def test_save_prediction_records_missing_feature_failure(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "missing-feature.db"
    _init_prediction_db(db_path)
    monkeypatch.setattr(prediction_cache_service, "get_conn", lambda: _connect(db_path))
    result = {**_prediction_result(), "missing_feature_status": "missing_orderbook_features"}

    prediction_cache_service.save_prediction(result)

    feature = _stage_logs(db_path)[0]
    assert feature["stage"] == "feature_construction"
    assert feature["status"] == "failed"
    assert feature["reason"] == "missing_orderbook_features"


def test_save_prediction_marks_future_source_time_as_data_leakage(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "data-leakage.db"
    _init_prediction_db(db_path)
    monkeypatch.setattr(prediction_cache_service, "get_conn", lambda: _connect(db_path))
    result = {**_prediction_result(), "sourceOpenTime": _prediction_result()["open_time"] + 60_000}

    prediction_cache_service.save_prediction(result)

    row = _prediction_row(db_path)
    logs = _stage_logs(db_path)
    assert row["data_freshness_status"] == "invalid_data_leakage"
    assert logs[0]["stage"] == "feature_construction"
    assert logs[0]["status"] == "failed"
    assert logs[0]["reason"] == "invalid_data_leakage"


def test_settle_due_predictions_records_settlement_stages(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "settled.db"
    _init_settlement_db(db_path)
    _insert_kline(db_path, 0, 100.0)
    _insert_kline(db_path, 600_000, 101.0)
    _insert_prediction(db_path, open_time=0)
    monkeypatch.setattr(forward_validation_service, "get_conn", lambda: _connect(db_path))

    result = forward_validation_service.settle_due_predictions("BTCUSDT", "10m")

    logs = _stage_logs(db_path)
    assert result == {"checked": 1, "settled": 1, "pendingData": 0}
    assert [row["stage"] for row in logs] == ["label_construction", "settlement_update"]
    assert {row["status"] for row in logs} == {"passed"}
    assert logs[1]["reason"] == "settled_with_real_kline"


def test_settle_due_predictions_records_pending_missing_price(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "pending.db"
    _init_settlement_db(db_path)
    _insert_prediction(db_path, open_time=0)
    del monkeypatch

    conn = _connect(db_path)
    try:
        row = dict(conn.execute("SELECT * FROM predictions").fetchone())
        settled = forward_validation_service._settle_prediction(conn, row, "10m")
        conn.commit()
    finally:
        conn.close()

    logs = _stage_logs(db_path)
    assert settled is False
    assert logs[0]["stage"] == "settlement_update"
    assert logs[0]["status"] == "pending"
    assert logs[0]["reason"] == "entry_and_settlement_price_missing"


def _init_prediction_db(path: Path) -> None:
    conn = _connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE predictions(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              signal_key TEXT, strategy_key TEXT, symbol TEXT, duration TEXT,
              open_time INTEGER, direction TEXT, probability_up REAL, confidence REAL,
              certainty_label TEXT, trade_quality_score REAL, trade_quality_passed INTEGER,
              trade_quality_gate TEXT, high_winrate_gate TEXT, high_winrate_rule TEXT,
              high_winrate_gate_passed INTEGER, high_winrate_gate_value REAL,
              high_winrate_gate_min REAL, entry_price REAL, expected_return REAL,
              model_version TEXT, model_family TEXT, validation_win_rate REAL,
              feature_window INTEGER, model_duration TEXT, model_trained_at TEXT,
              oos_win_rate REAL, walk_forward_result TEXT, recent_rolling_result TEXT,
              data_freshness_status TEXT, missing_feature_status TEXT,
              exit_price REAL, actual_return REAL, prediction_correct INTEGER,
              settled_at TEXT, created_at TEXT
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def _init_settlement_db(path: Path) -> None:
    conn = _connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE klines(symbol TEXT, interval TEXT, open_time INTEGER, close REAL);
            CREATE TABLE predictions(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              signal_key TEXT, strategy_key TEXT, symbol TEXT, duration TEXT,
              open_time INTEGER, direction TEXT, entry_price REAL,
              exit_price REAL, actual_return REAL, prediction_correct INTEGER,
              settled_at TEXT, created_at TEXT
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def _prediction_result() -> dict:
    return {
        "signal_key": "factor_alpha",
        "strategy_key": "factor_alpha",
        "symbol": "BTCUSDT",
        "duration": "10m",
        "open_time": 1_778_121_600_000,
        "direction": "up",
        "probability_up": 0.62,
        "confidence": 0.62,
        "certainty_label": "PAPER_LIVE",
        "trade_quality_score": 0.7,
        "trade_quality_passed": True,
        "model_version": "alpha_v1",
        "model_family": "factor",
        "oos_win_rate": 0.61,
        "walk_forward_result": {"stabilityScore": 0.7},
        "recent_rolling_result": {"window": "recent"},
        "data_freshness_status": "fresh",
        "missing_feature_status": "complete",
    }


def _insert_prediction(path: Path, *, open_time: int) -> None:
    conn = _connect(path)
    try:
        conn.execute(
            """
            INSERT INTO predictions(
              signal_key, strategy_key, symbol, duration, open_time, direction, entry_price, created_at
            )
            VALUES('factor_alpha', 'factor_alpha', 'BTCUSDT', '10m', ?, 'up', NULL, 'now')
            """,
            (open_time,),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_kline(path: Path, open_time: int, close: float) -> None:
    conn = _connect(path)
    try:
        conn.execute("INSERT INTO klines VALUES('BTCUSDT', '1m', ?, ?)", (open_time, close))
        conn.commit()
    finally:
        conn.close()


def _stage_logs(path: Path) -> list[sqlite3.Row]:
    conn = _connect(path)
    try:
        rows = conn.execute(
            """
            SELECT stage, status, reason
            FROM paper_live_prediction_stage_log
            ORDER BY id
            """
        ).fetchall()
        return rows
    finally:
        conn.close()


def _prediction_row(path: Path) -> sqlite3.Row:
    conn = _connect(path)
    try:
        return conn.execute("SELECT * FROM predictions LIMIT 1").fetchone()
    finally:
        conn.close()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn
