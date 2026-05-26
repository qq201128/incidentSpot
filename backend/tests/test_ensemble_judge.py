from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.services import auto_trade_service, ensemble_judge_service, ensemble_ranker_prediction_service
from app.services import forward_validation_service
from app.services.auto_trade_types import AutoTradeSettings
from app.services.ensemble_judge_constants import (
    ENSEMBLE_RANKER_STRATEGY_KEY,
    SIGNAL_HIGH_WINRATE_COMBO,
    STAGE_ENSEMBLE_READY,
    STAGE_OBSERVE,
    STAGE_WEIGHT_READY,
)
from app.services.factor_candidate_signal_keys import factor_candidate_signal_key
from app.services.model_family_config import model_family_strategy_key
from app.services.strategy_registry import (
    FACTOR_COMBO_STRATEGY_KEY,
    HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
)


def test_refresh_excludes_retired_strategies_from_candidates(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "retired.db"
    _init_db(db_path)
    _insert_predictions(db_path, FACTOR_COMBO_STRATEGY_KEY, 60, settled=True)
    _insert_predictions(db_path, "orderbook_notional_15m", 120, settled=True)
    _patch_db(monkeypatch, db_path)

    ranking = ensemble_judge_service.refresh_ensemble_judge("btcusdt", "10m")["ranking"]
    signal_keys = {row["signalKey"] for row in ranking}

    assert FACTOR_COMBO_STRATEGY_KEY in signal_keys
    assert "orderbook_notional_15m" not in signal_keys


def test_refresh_reads_only_settled_predictions(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "settled.db"
    _init_db(db_path)
    _insert_predictions(db_path, FACTOR_COMBO_STRATEGY_KEY, 60, settled=True)
    _insert_predictions(db_path, FACTOR_COMBO_STRATEGY_KEY, 40, settled=False)
    _patch_db(monkeypatch, db_path)

    result = ensemble_judge_service.refresh_ensemble_judge("btcusdt", "10m")

    ranking = result["ranking"]
    assert ranking[0]["signalKey"] == FACTOR_COMBO_STRATEGY_KEY
    assert ranking[0]["sampleCount"] == 60
    assert result["status"]["recommendedStage"] == STAGE_OBSERVE


def test_refresh_scores_factor_candidate_signals(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "factor-candidate.db"
    _init_db(db_path)
    candidate_key = factor_candidate_signal_key("rsi_14")
    _insert_predictions(db_path, candidate_key, 60, settled=True, label="rsi_14")
    _patch_db(monkeypatch, db_path)

    result = ensemble_judge_service.refresh_ensemble_judge("btcusdt", "10m")

    ranking = result["ranking"]
    assert ranking[0]["signalKey"] == candidate_key
    assert ranking[0]["signalType"] == "indicator"
    assert ranking[0]["signalLabel"] == "rsi_14"


def test_ranking_includes_unsettled_factor_candidates(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "pending-factor-candidates.db"
    _init_db(db_path)
    candidate_key = factor_candidate_signal_key("macd_hist_12_26_9")
    _insert_predictions(db_path, FACTOR_COMBO_STRATEGY_KEY, 60, settled=True)
    _insert_predictions(db_path, candidate_key, 3, settled=False, label="macd_hist_12_26_9")
    _patch_db(monkeypatch, db_path)
    ensemble_judge_service.refresh_ensemble_judge("btcusdt", "10m")

    ranking = ensemble_judge_service.ensemble_ranking("btcusdt", "10m")["ranking"]
    pending = next(row for row in ranking if row["signalKey"] == candidate_key)

    assert pending["signalType"] == "indicator"
    assert pending["signalLabel"] == "macd_hist_12_26_9"
    assert pending["sampleCount"] == 0
    assert pending["pendingCount"] == 3
    assert pending["pendingSettlement"] is True


def test_refresh_classifies_combo_shadow_signals(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "combo-shadows.db"
    _init_db(db_path)
    _insert_predictions(db_path, f"{FACTOR_COMBO_STRATEGY_KEY}_top2", 60, settled=True)
    _insert_predictions(db_path, f"{HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY}_top2", 60, settled=True)
    _patch_db(monkeypatch, db_path)

    ranking = ensemble_judge_service.refresh_ensemble_judge("btcusdt", "10m")["ranking"]
    by_key = {row["signalKey"]: row for row in ranking}

    assert by_key[f"{FACTOR_COMBO_STRATEGY_KEY}_top2"]["signalType"] == "factor_combo"
    assert by_key[f"{HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY}_top2"]["signalType"] == "high_winrate_combo"


def test_candidate_signals_affect_stage_coverage(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "coverage.db"
    _init_db(db_path)
    candidate_key = factor_candidate_signal_key("rsi_14")
    _insert_predictions(db_path, candidate_key, 200, settled=True, label="rsi_14")
    _insert_predictions(db_path, FACTOR_COMBO_STRATEGY_KEY, 200, settled=True)
    _insert_predictions(db_path, HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, 200, settled=True)
    _insert_predictions(db_path, model_family_strategy_key("lstm", "10m"), 200, settled=True)
    _patch_db(monkeypatch, db_path)

    status = ensemble_judge_service.refresh_ensemble_judge("BTCUSDT", "10m")["status"]

    assert status["sampleCoverage"]["sampleCount"] == 800
    assert status["sampleCoverage"]["readySignalTypeCount"] == 4
    assert status["sampleCoverage"]["requiredSignalTypeCount"] == 4
    assert status["sampleCoverage"]["bySignalType"]["indicator"]["sampleCount"] == 200
    assert status["sampleCoverage"]["byMajorSignalType"]["factor_candidate"]["sampleCount"] == 200


def test_low_sample_does_not_recommend_weight_ready(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "low.db"
    _init_db(db_path)
    for key in _major_keys():
        _insert_predictions(db_path, key, 199, settled=True)
    _patch_db(monkeypatch, db_path)

    status = ensemble_judge_service.refresh_ensemble_judge("BTCUSDT", "10m")["status"]

    assert status["recommendedStage"] == STAGE_OBSERVE


def test_weight_ready_after_major_sources_have_samples_and_days(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "weight.db"
    _init_db(db_path)
    for key in _major_keys():
        _insert_predictions(db_path, key, 200, settled=True)
    _patch_db(monkeypatch, db_path)

    status = ensemble_judge_service.refresh_ensemble_judge("BTCUSDT", "10m")["status"]

    assert status["recommendedStage"] == STAGE_WEIGHT_READY


def test_ensemble_ready_requires_simulated_ensemble_samples(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "ensemble.db"
    _init_db(db_path)
    for key in _major_keys():
        _insert_predictions(db_path, key, 500, settled=True)
    _insert_predictions(db_path, ENSEMBLE_RANKER_STRATEGY_KEY, 100, settled=True)
    _patch_db(monkeypatch, db_path)

    status = ensemble_judge_service.refresh_ensemble_judge("BTCUSDT", "10m")["status"]

    assert status["recommendedStage"] == STAGE_ENSEMBLE_READY


def test_confirm_stage_rejects_non_recommended_stage(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "confirm.db"
    _init_db(db_path)
    _insert_predictions(db_path, FACTOR_COMBO_STRATEGY_KEY, 60, settled=True)
    _patch_db(monkeypatch, db_path)
    ensemble_judge_service.refresh_ensemble_judge("BTCUSDT", "10m")

    with pytest.raises(ValueError, match="cannot confirm weight_ready"):
        ensemble_judge_service.confirm_ensemble_stage("BTCUSDT", "10m", STAGE_WEIGHT_READY)


def test_confirm_ensemble_ready_exposes_all_strategy_slots(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "slots.db"
    _init_db(db_path)
    for key in _major_keys():
        _insert_predictions(db_path, key, 500, settled=True)
    _insert_predictions(db_path, ENSEMBLE_RANKER_STRATEGY_KEY, 100, settled=True)
    _patch_db(monkeypatch, db_path)
    monkeypatch.setattr(auto_trade_service, "get_conn", lambda: _connect(db_path))
    ensemble_judge_service.refresh_ensemble_judge("BTCUSDT", "10m")

    ensemble_judge_service.confirm_ensemble_stage("BTCUSDT", "10m", STAGE_ENSEMBLE_READY)

    slots = [
        item for item in auto_trade_service.list_auto_trade_settings()
        if item.strategy_key == ENSEMBLE_RANKER_STRATEGY_KEY
    ]
    assert [slot.duration for slot in slots] == ["10m", "30m", "60m", "1d"]
    assert [slot.enabled for slot in slots] == [True, False, False, False]
    assert all(slot.live_trading_enabled is False for slot in slots)


def test_ensemble_prediction_fails_when_candidates_are_insufficient(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "insufficient.db"
    _init_db(db_path)
    _insert_stage(db_path, STAGE_ENSEMBLE_READY)
    for key in _major_keys()[:2]:
        _insert_score(db_path, key)
        _insert_prediction(db_path, key, 1, open_time=123, settled=False)
    monkeypatch.setattr(ensemble_ranker_prediction_service, "get_conn", lambda: _connect(db_path))

    with pytest.raises(ValueError, match="insufficient_ensemble_candidates"):
        ensemble_ranker_prediction_service.predict_ensemble_ranker_prediction("BTCUSDT", "10m", entry_open_time=123)


def test_ensemble_live_trading_is_rejected() -> None:
    settings = AutoTradeSettings(
        strategy_key=ENSEMBLE_RANKER_STRATEGY_KEY,
        enabled=True,
        symbol="BTCUSDT",
        duration="10m",
        duration_minutes=10,
        qty=5,
        live_trading_enabled=True,
    )

    with pytest.raises(ValueError, match="simulation only"):
        auto_trade_service.update_auto_trade_settings(settings)


def test_forward_validation_settles_ensemble_predictions(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "settle.db"
    _init_db(db_path)
    _insert_kline(db_path, 0, 100)
    _insert_kline(db_path, 600_000, 101)
    _insert_prediction(db_path, ENSEMBLE_RANKER_STRATEGY_KEY, 1, open_time=0, settled=False)
    monkeypatch.setattr(forward_validation_service, "get_conn", lambda: _connect(db_path))

    result = forward_validation_service.settle_due_predictions("BTCUSDT", "10m")

    row = _prediction_row(db_path, ENSEMBLE_RANKER_STRATEGY_KEY)
    assert result["settled"] == 1
    assert row["prediction_correct"] == 1
    assert row["settled_at"] is not None


def _major_keys() -> tuple[str, str, str]:
    return (
        FACTOR_COMBO_STRATEGY_KEY,
        HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
        model_family_strategy_key("lstm", "10m"),
        factor_candidate_signal_key("rsi_14"),
    )


def _patch_db(monkeypatch, db_path: Path) -> None:
    monkeypatch.setattr(ensemble_judge_service, "get_conn", lambda: _connect(db_path))


def _init_db(path: Path) -> None:
    conn = _connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE klines(symbol TEXT, interval TEXT, open_time INTEGER, close REAL);
            CREATE TABLE predictions(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              signal_key TEXT, strategy_key TEXT, symbol TEXT, duration TEXT, open_time INTEGER,
              direction TEXT, probability_up REAL, confidence REAL, certainty_label TEXT,
              high_winrate_gate TEXT, high_winrate_rule TEXT, model_version TEXT,
              expected_return REAL, entry_price REAL, exit_price REAL, actual_return REAL,
              prediction_correct INTEGER, settled_at TEXT, created_at TEXT
            );
            CREATE TABLE ensemble_stage_status(
              symbol TEXT, duration TEXT, stage TEXT, recommended_stage TEXT,
              recommendation_reason TEXT, confirmed_stage TEXT, confirmed_at TEXT,
              updated_at TEXT, PRIMARY KEY(symbol, duration)
            );
            CREATE TABLE ensemble_signal_scores(
              symbol TEXT, duration TEXT, signal_key TEXT, signal_type TEXT,
              sample_count INTEGER, win_rate REAL, avg_return REAL, profit_factor REAL,
              consecutive_losses INTEGER, stability_score REAL, weight_suggestion REAL,
              score REAL, updated_at TEXT, PRIMARY KEY(symbol, duration, signal_key)
            );
            CREATE TABLE auto_trade_strategies(
              strategy_key TEXT, duration TEXT, enabled INTEGER, live_trading_enabled INTEGER,
              symbol TEXT, duration_minutes INTEGER, qty REAL, updated_at TEXT,
              PRIMARY KEY(strategy_key, duration)
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def _insert_predictions(path: Path, strategy_key: str, count: int, *, settled: bool, label: str | None = None) -> None:
    conn = _connect(path)
    try:
        for index in range(count):
            _insert_prediction_conn(conn, strategy_key, index, index * 86_400_000, settled, label=label)
        conn.commit()
    finally:
        conn.close()


def _insert_prediction(
    path: Path,
    strategy_key: str,
    index: int,
    *,
    open_time: int,
    settled: bool,
    label: str | None = None,
) -> None:
    conn = _connect(path)
    try:
        _insert_prediction_conn(conn, strategy_key, index, open_time, settled, label=label)
        conn.commit()
    finally:
        conn.close()


def _insert_prediction_conn(
    conn: sqlite3.Connection,
    strategy_key: str,
    index: int,
    open_time: int,
    settled: bool,
    *,
    label: str | None = None,
) -> None:
    display = label or strategy_key
    actual_return = 0.01 if settled else None
    conn.execute(
        """
        INSERT INTO predictions
        (signal_key, strategy_key, symbol, duration, open_time, direction, probability_up, confidence,
         certainty_label, high_winrate_gate, high_winrate_rule, model_version, expected_return,
         entry_price, exit_price, actual_return, prediction_correct, settled_at, created_at)
        VALUES(?, ?, 'BTCUSDT', '10m', ?, 'up', 0.62, 0.62, 'test',
               NULL, ?, ?, 0.01, 0.01, NULL, ?, ?, ?, 'now')
        """,
        (
            strategy_key,
            strategy_key,
            open_time,
            display,
            display,
            actual_return,
            1,
            "done" if settled else None,
        ),
    )


def _insert_score(path: Path, strategy_key: str) -> None:
    conn = _connect(path)
    try:
        conn.execute(
            """
            INSERT INTO ensemble_signal_scores
            VALUES('BTCUSDT', '10m', ?, 'factor_combo', 500, 0.6, 0.01, 2, 0, 1, 1, 90, 'now')
            """,
            (strategy_key,),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_stage(path: Path, stage: str) -> None:
    conn = _connect(path)
    try:
        conn.execute(
            """
            INSERT INTO ensemble_stage_status
            VALUES('BTCUSDT', '10m', ?, ?, 'ready', ?, 'now', 'now')
            """,
            (stage, stage, stage),
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


def _prediction_row(path: Path, strategy_key: str) -> sqlite3.Row:
    conn = _connect(path)
    try:
        return conn.execute("SELECT * FROM predictions WHERE signal_key = ?", (strategy_key,)).fetchone()
    finally:
        conn.close()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn
