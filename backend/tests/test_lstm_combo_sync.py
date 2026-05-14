from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from app.services import factor_combination_background as combo_background
from app.services import lstm_combo_snapshot, lstm_prediction_service
from app.services.lstm_artifacts import artifact_paths, write_json
from app.services.lstm_combo_sync_service import (
    LSTM_SYNC_TRAINED,
    LSTM_SYNC_UP_TO_DATE,
    sync_lstm_model_to_combo_ranking,
)
from app.services.lstm_config import LstmTrainingConfig


def test_lstm_model_status_marks_stale_combo_snapshot(monkeypatch) -> None:
    artifact_root = _runtime_path("status-stale")
    _write_trained_artifacts(artifact_root, _snapshot("combo_old", "factor_a"))
    monkeypatch.setattr(
        lstm_combo_snapshot,
        "get_cached_combination_ranking",
        lambda *_args: _ranking("combo_new", "factor_b"),
    )

    status = lstm_prediction_service.lstm_model_status(
        "BTCUSDT",
        "10m",
        artifact_root=artifact_root,
    )

    assert status["comboSnapshotMatches"] is False
    assert status["comboSnapshotReason"] == "combo_snapshot_mismatch"
    assert status["shadowPredictionReady"] is False


def test_lstm_model_status_exposes_torch_blocker(monkeypatch) -> None:
    artifact_root = _runtime_path("status-torch")
    _write_trained_artifacts(artifact_root, _snapshot("combo_current", "factor_a"))
    monkeypatch.setattr(
        lstm_combo_snapshot,
        "get_cached_combination_ranking",
        lambda *_args: _ranking("combo_current", "factor_a"),
    )
    monkeypatch.setattr(
        lstm_prediction_service,
        "torch_availability",
        lambda: {"available": False, "error": "missing torch"},
    )

    status = lstm_prediction_service.lstm_model_status(
        "BTCUSDT",
        "10m",
        artifact_root=artifact_root,
    )

    assert status["torchAvailable"] is False
    assert status["torchStatus"]["error"] == "missing torch"
    assert status["shadowPredictionReady"] is False
    assert status["shadowPredictionBlockedReason"] == "torch_unavailable"


def test_sync_lstm_model_retrains_when_combo_snapshot_changes() -> None:
    artifact_root = _runtime_path("sync-stale")
    _write_trained_artifacts(artifact_root, _snapshot("combo_old", "factor_a"))
    calls = []

    def trainer(config: LstmTrainingConfig) -> dict[str, Any]:
        calls.append(config)
        return {"modelVersion": "lstm_new"}

    result = sync_lstm_model_to_combo_ranking(
        "btcusdt",
        "10m",
        ranking_report=_ranking("combo_new", "factor_b"),
        artifact_root=artifact_root,
        trainer=trainer,
    )

    assert result["status"] == LSTM_SYNC_TRAINED
    assert result["modelVersion"] == "lstm_new"
    assert [(call.symbol, call.duration) for call in calls] == [("BTCUSDT", "10m")]


def test_sync_lstm_model_skips_when_combo_snapshot_matches() -> None:
    artifact_root = _runtime_path("sync-current")
    snapshot = _snapshot("combo_current", "factor_a")
    _write_trained_artifacts(artifact_root, snapshot)

    def trainer(_config: LstmTrainingConfig) -> dict[str, Any]:
        raise AssertionError("trainer should not run for a current LSTM snapshot")

    result = sync_lstm_model_to_combo_ranking(
        "BTCUSDT",
        "10m",
        ranking_report=_ranking("combo_current", "factor_a"),
        artifact_root=artifact_root,
        trainer=trainer,
    )

    assert result["status"] == LSTM_SYNC_UP_TO_DATE


def test_sync_lstm_model_retrains_when_validation_gate_missing() -> None:
    artifact_root = _runtime_path("sync-missing-gate")
    snapshot = _snapshot("combo_current", "factor_a")
    _write_trained_artifacts(artifact_root, snapshot, include_validation_gate=False)
    calls = []

    def trainer(config: LstmTrainingConfig) -> dict[str, Any]:
        calls.append(config)
        return {"modelVersion": "lstm_new"}

    result = sync_lstm_model_to_combo_ranking(
        "BTCUSDT",
        "10m",
        ranking_report=_ranking("combo_current", "factor_a"),
        artifact_root=artifact_root,
        trainer=trainer,
    )

    assert result["status"] == LSTM_SYNC_TRAINED
    assert result["modelVersion"] == "lstm_new"
    assert len(calls) == 1


def test_factor_combo_refresh_syncs_lstm_before_learning(monkeypatch) -> None:
    calls = []

    def fake_run(symbol: str, duration: str, _config: object) -> dict[str, Any]:
        calls.append(("run", symbol, duration))
        return _ranking("combo_current", "factor_a")

    def fake_save(report: dict[str, Any]) -> None:
        calls.append(("save", report["symbol"], report["duration"]))

    def fake_upsert(report: dict[str, Any]) -> dict[str, Any]:
        calls.append(("promote", report["symbol"], report["duration"]))
        return _promotion(report)

    def fake_sync(symbol: str, duration: str, *, ranking_report: dict[str, Any]) -> dict[str, Any]:
        calls.append(("sync", symbol, duration, ranking_report["ranking"][0]["factorName"]))
        return {"status": LSTM_SYNC_UP_TO_DATE}

    def fake_learning(symbol: str, duration: str, ranking_report: dict[str, Any], *, run_llm_agent: bool) -> None:
        assert run_llm_agent is True
        calls.append(("learn", symbol, duration, ranking_report["ranking"][0]["factorName"]))

    monkeypatch.setattr(combo_background, "run_factor_combination_ranking", fake_run)
    monkeypatch.setattr(combo_background, "save_cached_combination_ranking", fake_save)
    monkeypatch.setattr(combo_background, "upsert_good_combinations", fake_upsert)
    monkeypatch.setattr(combo_background, "sync_lstm_model_to_combo_ranking", fake_sync)
    monkeypatch.setattr(combo_background, "refresh_factor_learning_memory", fake_learning)

    combo_background.refresh_combination_ranking_for_symbol_duration("btcusdt", "10m")

    assert [call[0] for call in calls] == ["run", "save", "promote", "sync", "learn"]


def _write_trained_artifacts(
    root: Path,
    combo_snapshot: list[dict[str, Any]],
    *,
    include_validation_gate: bool = True,
) -> None:
    paths = artifact_paths("BTCUSDT", "10m", root)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.model.write_bytes(b"fake")
    write_json(paths.features, {"columns": ["x"], "featureWindow": 4, "comboSnapshot": combo_snapshot})
    write_json(paths.scaler, {"mean": [0.0], "std": [1.0]})
    version = {"modelVersion": "lstm_old", "trainedAt": "2026-05-13T00:00:00+00:00"}
    report = {"status": "trained", "modelVersion": "lstm_old"}
    if include_validation_gate:
        validation_gate = _passed_validation_gate()
        version["validationGate"] = validation_gate
        version["selectedConfidenceThreshold"] = validation_gate["minConfidence"]
        report["validationGate"] = validation_gate
        report["selectedConfidenceThreshold"] = validation_gate["minConfidence"]
    write_json(paths.version, version)
    write_json(paths.status, {"status": "trained", "symbol": "BTCUSDT", "duration": "10m"})
    write_json(paths.report, report)


def _passed_validation_gate() -> dict[str, Any]:
    return {
        "status": "passed",
        "minConfidence": 0.6,
        "winRate": 0.7,
        "profitFactor": 1.2,
        "avgReturn": 0.01,
    }


def _ranking(factor_name: str, member_name: str) -> dict[str, Any]:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "ranking": [
            {"factorName": factor_name, "members": [{"name": member_name}]},
            {"factorName": "combo_top2", "members": [{"name": "factor_c"}]},
            {"factorName": "combo_top3", "members": [{"name": "factor_d"}]},
        ],
    }


def _snapshot(factor_name: str, member_name: str) -> list[dict[str, Any]]:
    return [
        {"rank": 1, "factorName": factor_name, "members": [member_name]},
        {"rank": 2, "factorName": "combo_top2", "members": ["factor_c"]},
        {"rank": 3, "factorName": "combo_top3", "members": ["factor_d"]},
    ]


def _promotion(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": report["symbol"],
        "duration": report["duration"],
        "promoted": 1,
        "libraryTotal": 1,
    }


def _runtime_path(name: str) -> Path:
    base = Path(__file__).resolve().parents[1] / "runtime" / "pytest-temp"
    path = base / f"lstm-combo-{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
