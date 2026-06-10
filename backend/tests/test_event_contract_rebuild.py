from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services import event_final_decision_service as final_service
from app.services import event_regime_detector as regime_detector
from app.services import market_data_repair_service as repair_service
from app.services.event_regime_detector import EventRegime
from app.services.lstm_feature_builder import build_lstm_training_dataset
from app.services.model_family_config import ModelFamilyTrainingConfig


ENTRY_OPEN_TIME = 1_700_000_000_000


def test_market_data_repair_refetches_gap_without_interpolation(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "repair.db"
    _init_repair_db(db_path)
    _insert_kline(db_path, ENTRY_OPEN_TIME, close=100)
    _insert_kline(db_path, ENTRY_OPEN_TIME + 1_200_000, close=101)
    fetched = []

    def fetcher(symbol, interval, **kwargs):
        fetched.append((symbol, interval, kwargs))
        return [_kline_row(ENTRY_OPEN_TIME + 600_000, close=100.5)]

    monkeypatch.setattr(repair_service, "get_conn", lambda: _connect(db_path))
    result = repair_service.repair_market_klines("BTCUSDT", "10m", fetcher=fetcher, upsert=lambda *_args: None)

    assert result["issues"] == 1
    assert fetched[0][2]["start_time"] == ENTRY_OPEN_TIME + 600_000
    assert _report_statuses(db_path) == ["repaired"]


def test_event_regime_uses_completed_bars_only(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "regime.db"
    _init_regime_db(db_path)
    for index in range(90):
        _insert_kline(db_path, ENTRY_OPEN_TIME - (90 - index) * 600_000, close=100 + index)
    _insert_kline(db_path, ENTRY_OPEN_TIME, close=1)

    monkeypatch.setattr(regime_detector, "get_conn", lambda: _connect(db_path))
    regime = regime_detector.detect_event_regime("BTCUSDT", "10m", ENTRY_OPEN_TIME)

    assert regime.trend_state == "trend_up"
    assert regime.open_time == ENTRY_OPEN_TIME


def test_final_decision_skips_when_candidates_are_insufficient(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "final-skip.db"
    _init_final_db(db_path)
    monkeypatch.setattr(final_service, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(final_service, "detect_event_regime", lambda *_args: _regime())
    monkeypatch.setattr(final_service, "_decision_candidates", lambda *_args: [])

    result = final_service.predict_event_final_decision("BTCUSDT", "10m", entry_open_time=ENTRY_OPEN_TIME)

    assert result is None
    row = _decision_row(db_path)
    assert row["decision"] == "SKIP"
    assert "insufficient_candidates" in row["reason_codes"]


def test_final_decision_emits_prediction_for_model_family_agreement(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "final-up.db"
    _init_final_db(db_path)
    candidates = [
        final_service.DecisionCandidate("a", "factor_lstm_shadow_10m", "up", 0.62, 0.62, 0.40),
        final_service.DecisionCandidate("b", "factor_gru_shadow_10m", "up", 0.60, 0.60, 0.38),
    ]
    monkeypatch.setattr(final_service, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(final_service, "detect_event_regime", lambda *_args: _regime())
    monkeypatch.setattr(final_service, "_decision_candidates", lambda *_args: candidates)

    result = final_service.predict_event_final_decision("BTCUSDT", "10m", entry_open_time=ENTRY_OPEN_TIME)

    assert result is not None
    assert result["strategy_key"] == final_service.EVENT_FINAL_DECISION_STRATEGY_KEY
    assert result["direction"] == "up"
    assert _decision_row(db_path)["decision"] == "UP"


def test_final_decision_skip_keeps_candidate_metrics() -> None:
    candidates = [
        final_service.DecisionCandidate("a", "factor_lstm_shadow_10m", "up", 0.51, 0.51, 0.30),
        final_service.DecisionCandidate("b", "factor_gru_shadow_10m", "down", 0.49, 0.51, 0.30),
    ]

    decision = final_service._final_decision(
        "BTCUSDT",
        "10m",
        ENTRY_OPEN_TIME,
        "trend_up:normal_vol",
        0.8,
        candidates,
    )

    assert decision.decision == "SKIP"
    assert decision.candidate_count == 2
    assert decision.probability_up == 0.5
    assert decision.confidence == 0.5
    assert decision.final_score > 0


def test_model_family_dataset_uses_clean_regime_features(monkeypatch) -> None:
    monkeypatch.setattr("app.services.lstm_market_feature_builder.load_orderbook_features", lambda *_args: pd.DataFrame())
    monkeypatch.setattr("app.services.lstm_market_feature_builder.load_funding_features", lambda *_args: pd.DataFrame())
    monkeypatch.setattr("app.services.lstm_market_feature_builder.load_external_feature_frames", lambda *_args: None)
    monkeypatch.setattr("app.services.lstm_feature_builder.load_factor_learning_memory_for", lambda *_args: None)
    monkeypatch.setattr("app.services.lstm_feature_builder.build_lstm_market_feature_frame", lambda frame, *_args, **_kwargs: frame)
    config = ModelFamilyTrainingConfig(family="lstm", symbol="BTCUSDT", duration="10m", min_samples=20)

    dataset = build_lstm_training_dataset(config, frame_loader=lambda *_args: _raw_frame(360))

    assert any(column.startswith("regime_") for column in dataset.feature_columns)
    assert not any(column.startswith("sim_feedback_") for column in dataset.feature_columns)
    assert not any(column.startswith("factor_combo_") for column in dataset.feature_columns)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _init_repair_db(path: Path) -> None:
    conn = _connect(path)
    conn.executescript(
        """
        CREATE TABLE klines(symbol TEXT, interval TEXT, open_time INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL);
        CREATE TABLE market_data_quality_reports(
          id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, interval TEXT, issue_type TEXT,
          start_open_time INTEGER, end_open_time INTEGER, status TEXT, reason TEXT,
          repair_source TEXT, details_json TEXT, created_at TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def _init_regime_db(path: Path) -> None:
    conn = _connect(path)
    conn.execute("CREATE TABLE klines(symbol TEXT, interval TEXT, open_time INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    conn.commit()
    conn.close()


def _init_final_db(path: Path) -> None:
    conn = _connect(path)
    conn.executescript(
        """
        CREATE TABLE event_market_regimes(symbol TEXT, duration TEXT, open_time INTEGER, trend_state TEXT, volatility_state TEXT, regime_label TEXT, confidence REAL, reason_codes TEXT, metrics_json TEXT, created_at TEXT);
        CREATE TABLE event_final_decisions(symbol TEXT, duration TEXT, open_time INTEGER, decision TEXT, direction TEXT, probability_up REAL, confidence REAL, final_score REAL, regime_label TEXT, candidate_count INTEGER, reason_codes TEXT, settled_at TEXT, decision_correct INTEGER, actual_direction TEXT, exit_price REAL, created_at TEXT);
        """
    )
    conn.commit()
    conn.close()


def _insert_kline(path: Path, open_time: int, *, close: float) -> None:
    conn = _connect(path)
    conn.execute(
        "INSERT INTO klines VALUES('BTCUSDT', '10m', ?, ?, ?, ?, ?, 1)",
        (open_time, close, close + 1, close - 1, close),
    )
    conn.commit()
    conn.close()


def _kline_row(open_time: int, *, close: float) -> dict:
    return {"openTime": open_time, "open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1}


def _report_statuses(path: Path) -> list[str]:
    conn = _connect(path)
    rows = conn.execute("SELECT status FROM market_data_quality_reports ORDER BY id").fetchall()
    conn.close()
    return [str(row["status"]) for row in rows]


def _regime() -> EventRegime:
    return EventRegime("BTCUSDT", "10m", ENTRY_OPEN_TIME, "trend_up", "normal_vol", "trend_up:normal_vol", 0.8, ("trend_up",), {})


def _decision_row(path: Path) -> sqlite3.Row:
    conn = _connect(path)
    row = conn.execute("SELECT * FROM event_final_decisions").fetchone()
    conn.close()
    assert row is not None
    return row


def _raw_frame(rows: int) -> pd.DataFrame:
    open_time = np.arange(rows) * 600_000
    close = 100 + np.sin(np.arange(rows) / 8) + np.arange(rows) * 0.01
    return pd.DataFrame({
        "open_time": open_time,
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(rows, 10.0),
    })
