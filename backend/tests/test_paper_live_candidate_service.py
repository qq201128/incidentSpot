from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from tempfile import gettempdir

import pytest

from app.services import paper_live_candidate_service as service
from app.services import paper_live_failure_store


def test_candidate_report_uses_settled_paper_live_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("paper-live") / "candidates.db"
    _create_db(db_path)
    _insert_predictions(db_path)
    monkeypatch.setattr(service, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(paper_live_failure_store, "get_conn", lambda: _connect(db_path))
    service.log_prediction_failure(
        candidate_key="factor_beta",
        strategy_key="factor_beta",
        symbol="BTCUSDT",
        duration="10m",
        stage="factor_candidate_signal_prediction",
        reason="missing completed 10m source row",
        details={"entryOpenTime": 123},
    )

    report = service.refresh_paper_live_candidate_states("BTCUSDT", "10m")

    candidate = report["candidates"][0]
    failed = report["failed"][0]
    assert report["realTradingEnabled"] is False
    assert report["rankingPolicy"][0] == "paper-live lifecycle stability"
    assert candidate["candidateKey"] == "factor_gamma"
    assert candidate["paperLiveStatus"] == "paper_collecting"
    assert candidate["liveReadiness"]["eligible"] is False
    assert candidate["liveReadiness"]["reason"] == "insufficient_settled_samples"
    assert failed["candidateKey"] == "factor_alpha"
    assert failed["backtestWinRate"] == pytest.approx(0.82)
    assert failed["oosWinRate"] == pytest.approx(0.61)
    assert failed["walkForwardResult"]["stabilityScore"] == pytest.approx(0.72)
    assert failed["recentRollingResult"]["winRate"] == pytest.approx(0.64)
    assert failed["paperLiveWinRate"] == pytest.approx(0.6)
    assert failed["paperLiveSampleCount"] == 30
    assert failed["paperLiveStatus"] == "paper_failed"
    assert failed["liveReadiness"]["eligible"] is False
    assert failed["liveReadiness"]["reason"] == "paper_live_profit_factor_below_target"
    assert failed["reason"] == "paper_live_profit_factor_below_target"
    assert failed["metrics"]["paperLiveWindows"]["recent30"]["winRate"] == pytest.approx(0.6)
    assert failed["metrics"]["maxConsecutiveLosses"] == 2
    assert failed["performanceComparison"]["winRateGap"] == pytest.approx(0.22)
    assert failed["performanceComparison"]["oosWinRate"] == pytest.approx(0.61)
    assert failed["performanceComparison"]["policy"] == "backtest_oos_walk_forward_recent_rolling_are_prefilter_only"
    assert report["avoidNextSearch"][0]["candidateKey"] == "factor_alpha"
    assert report["predictionFailures"][0]["candidateKey"] == "factor_beta"
    assert report["predictionFailures"][0]["reason"] == "missing completed 10m source row"
    assert report["statusChanges"][0]["newStatus"] in {"paper_collecting", "paper_failed"}
    assert report["answers"]["collectingSamples"][0]["candidateKey"] == "factor_gamma"
    assert report["answers"]["failedCandidates"][0]["candidateKey"] == "factor_alpha"
    assert report["answers"]["avoidNextSearch"][0]["candidateKey"] == "factor_alpha"


def test_candidate_status_changes_are_recorded_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("paper-live-history") / "candidates.db"
    _create_db(db_path)
    _insert_predictions(db_path)
    monkeypatch.setattr(service, "get_conn", lambda: _connect(db_path))

    service.refresh_paper_live_candidate_states("BTCUSDT", "10m")
    _insert_gamma_failures(db_path)
    report = service.refresh_paper_live_candidate_states("BTCUSDT", "10m")

    gamma_change = next(row for row in report["statusChanges"] if row["candidateKey"] == "factor_gamma")
    assert gamma_change["oldStatus"] == "paper_collecting"
    assert gamma_change["newStatus"] == "paper_failed"
    assert gamma_change["reason"] == "consecutive_losses"
    assert gamma_change["details"]["paperLiveStatus"] == "paper_failed"


def test_model_candidates_use_model_version_lifecycle_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("paper-live-model") / "candidates.db"
    _create_db(db_path)
    _insert_model_predictions(db_path)
    monkeypatch.setattr(service, "get_conn", lambda: _connect(db_path))

    report = service.refresh_paper_live_candidate_states("BTCUSDT", "10m")
    model = report["failed"][0]

    assert model["candidateType"] == "model"
    assert model["candidateKey"] == "xgboost_v2"
    assert model["strategyKey"] == "factor_xgboost_shadow_10m"
    assert model["modelFamily"] == "xgboost"
    assert model["modelVersion"] == "xgboost_v2"
    assert model["featureWindow"] == 32
    assert model["minConfidence"] == pytest.approx(0.65)
    assert model["validationWinRate"] == pytest.approx(0.64)
    assert model["paperLiveWinRate"] == pytest.approx(0.5)
    assert model["paperLiveSampleCount"] == 30
    assert model["paperLiveStatus"] == "paper_failed"


def test_candidate_with_future_data_leakage_is_marked_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("paper-live-leakage") / "candidates.db"
    _create_db(db_path)
    _insert_leakage_prediction(db_path)
    monkeypatch.setattr(service, "get_conn", lambda: _connect(db_path))

    report = service.refresh_paper_live_candidate_states("BTCUSDT", "10m")

    invalid = report["failed"][0]
    assert invalid["candidateKey"] == "factor_leaky"
    assert invalid["paperLiveStatus"] == "invalid_data_leakage"
    assert invalid["reason"] == "invalid_data_leakage"
    assert invalid["dataFreshnessStatus"] == "invalid_data_leakage"


def test_collecting_candidates_rank_by_oos_after_paper_live_status(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("paper-live-oos-rank") / "candidates.db"
    _create_db(db_path)
    _insert_unsettled_candidate(db_path, "low_oos", 0.55)
    _insert_unsettled_candidate(db_path, "high_oos", 0.68)
    monkeypatch.setattr(service, "get_conn", lambda: _connect(db_path))

    report = service.refresh_paper_live_candidate_states("BTCUSDT", "10m")

    assert [row["candidateKey"] for row in report["candidates"][:2]] == ["high_oos", "low_oos"]
    assert report["candidates"][0]["oosWinRate"] == pytest.approx(0.68)


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
        """
    )
    conn.close()


def _insert_unsettled_candidate(path: Path, factor_name: str, oos_win_rate: float) -> None:
    conn = _connect(path)
    conn.execute(
        """
        INSERT INTO predictions(
          signal_key, strategy_key, symbol, duration, open_time, direction,
          high_winrate_rule, high_winrate_gate_value, oos_win_rate,
          walk_forward_result, recent_rolling_result, data_freshness_status,
          missing_feature_status, created_at
        )
        VALUES(?, ?, 'BTCUSDT', '10m', 1, 'up', ?, 0.7, ?, ?, ?, 'fresh', 'complete', ?)
        """,
        (
            factor_name,
            factor_name,
            factor_name,
            oos_win_rate,
            json.dumps({"stabilityScore": 0.5, "oosWinRate": oos_win_rate}),
            json.dumps({"winRate": 0.6}),
            "2026-05-26T00:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()


def _insert_predictions(path: Path) -> None:
    conn = _connect(path)
    rows = [_prediction_row("factor_alpha", "alpha", 0.82, idx, idx % 5 < 3) for idx in range(30)]
    rows.append(_prediction_row("factor_gamma", "gamma", 0.71, 30, True))
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


def _insert_model_predictions(path: Path) -> None:
    conn = _connect(path)
    rows = [_model_prediction_row(idx, idx % 2 == 0) for idx in range(30)]
    conn.executemany(
        """
        INSERT INTO predictions(
          signal_key, strategy_key, symbol, duration, open_time, direction,
          high_winrate_rule, high_winrate_gate_value, high_winrate_gate_min,
          model_family, model_version, validation_win_rate, feature_window,
          model_duration, data_freshness_status, missing_feature_status,
          actual_return, prediction_correct, settled_at, created_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def _insert_gamma_failures(path: Path) -> None:
    conn = _connect(path)
    rows = [_prediction_row("factor_gamma", "gamma", 0.71, 31 + idx, False) for idx in range(29)]
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


def _insert_leakage_prediction(path: Path) -> None:
    conn = _connect(path)
    conn.execute(
        """
        INSERT INTO predictions(
          signal_key, strategy_key, symbol, duration, open_time, direction,
          high_winrate_rule, high_winrate_gate_value, data_freshness_status,
          missing_feature_status, actual_return, prediction_correct, settled_at, created_at
        )
        VALUES(
          'factor_leaky', 'factor_leaky', 'BTCUSDT', '10m', 1, 'up',
          'leaky', 0.9, 'invalid_data_leakage', 'complete',
          0.01, 1, '2026-05-26T00:00:00+00:00', '2026-05-26T00:00:00+00:00'
        )
        """
    )
    conn.commit()
    conn.close()


def _prediction_row(signal_key: str, factor_name: str, backtest_win_rate: float, index: int, correct: bool) -> tuple:
    actual_return = 0.01 if correct else -0.02
    return (
        signal_key,
        signal_key,
        "BTCUSDT",
        "10m",
        index,
        "up",
        factor_name,
        backtest_win_rate,
        0.61,
        json.dumps({"stabilityScore": 0.72, "oosWinRate": 0.61}),
        json.dumps({"winRate": 0.64}),
        "fresh",
        "complete",
        actual_return,
        int(correct),
        "2026-05-26T00:00:00+00:00",
        "2026-05-26T00:00:00+00:00",
    )


def _model_prediction_row(index: int, correct: bool) -> tuple:
    actual_return = 0.01 if correct else -0.02
    return (
        "factor_xgboost_shadow_10m",
        "factor_xgboost_shadow_10m",
        "BTCUSDT",
        "10m",
        index,
        "up",
        "xgboost_v2",
        0.66,
        0.65,
        "xgboost",
        "xgboost_v2",
        0.64,
        32,
        "10m",
        "fresh",
        "complete",
        actual_return,
        int(correct),
        "2026-05-26T00:00:00+00:00",
        "2026-05-26T00:00:00+00:00",
    )


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _runtime_path(name: str) -> Path:
    path = Path(gettempdir()) / "incidentSpot-pytest-temp" / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
