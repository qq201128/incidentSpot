from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services import lstm_combo_snapshot
from app.services import lstm_prediction_service
from app.services.lstm_artifacts import artifact_paths
from app.services.lstm_config import LstmTrainingConfig
from app.services.lstm_feature_builder import LstmDataset
from app.services.lstm_training_service import train_lstm_model

ENTRY_OPEN_TIME = 1778121600000


def test_validation_failed_status_blocks_prediction(monkeypatch) -> None:
    artifact_root = _runtime_path("legacy-status")
    _write_validation_failed_artifacts(artifact_root)
    _patch_combo_ranking(monkeypatch)
    monkeypatch.setattr(lstm_prediction_service, "build_live_feature_window", _live_window)

    status = lstm_prediction_service.lstm_model_status("BTCUSDT", "10m", artifact_root=artifact_root)

    assert status["shadowPredictionReady"] is False
    assert status["shadowPredictionBlockedReason"] == "no_validation_confidence_threshold_met"
    with pytest.raises(ValueError, match="no_validation_confidence_threshold_met"):
        lstm_prediction_service.predict_lstm_signal(
            "BTCUSDT",
            "10m",
            artifact_root=artifact_root,
            backend=_PredictOnlyBackend(),
        )


def test_trained_artifacts_missing_validation_gate_block_prediction(monkeypatch) -> None:
    artifact_root = _runtime_path("missing-gate")
    _write_legacy_trained_artifacts(artifact_root)
    _patch_combo_ranking(monkeypatch)
    monkeypatch.setattr(lstm_prediction_service, "build_live_feature_window", _live_window)

    status = lstm_prediction_service.lstm_model_status("BTCUSDT", "10m", artifact_root=artifact_root)

    assert status["status"] == "trained"
    assert status["comboSnapshotMatches"] is True
    assert status["shadowPredictionReady"] is False
    assert status["shadowPredictionBlockedReason"] == "validation_gate_missing"
    with pytest.raises(ValueError, match="validation_gate_missing"):
        lstm_prediction_service.predict_lstm_signal(
            "BTCUSDT",
            "10m",
            artifact_root=artifact_root,
            backend=_PredictOnlyBackend(),
        )


def test_predict_lstm_signal_reads_validation_gate_from_report(monkeypatch) -> None:
    artifact_root = _runtime_path("report-gate")
    _write_report_gate_artifacts(artifact_root)
    _patch_combo_ranking(monkeypatch)
    monkeypatch.setattr(lstm_prediction_service, "build_live_feature_window", _live_window)

    signal = lstm_prediction_service.predict_lstm_signal(
        "BTCUSDT",
        "10m",
        artifact_root=artifact_root,
        backend=_PredictOnlyBackend(),
    )

    assert signal["selectedConfidenceThreshold"] == pytest.approx(0.6)
    assert signal["validationGatePassed"] is True


def test_train_lstm_model_records_failed_attempt_without_active_model(monkeypatch) -> None:
    artifact_root = _runtime_path("validation-failed")
    report = train_lstm_model(
        LstmTrainingConfig(symbol="BTCUSDT", duration="10m", feature_window=8, min_samples=30, epochs=1),
        artifact_root=artifact_root,
        backend=_LowConfidenceBackend(),
        dataset_builder=_fake_dataset,
    )
    paths = artifact_paths("BTCUSDT", "10m", artifact_root)
    staging_status = _staging_status(artifact_root, report["modelVersion"])
    _patch_combo_ranking(monkeypatch)

    status = lstm_prediction_service.lstm_model_status("BTCUSDT", "10m", artifact_root=artifact_root)

    assert report["status"] == "validation_failed"
    assert report["candidateStatus"] == "rejected_validation"
    assert report["promotionReason"] == "no_validation_confidence_threshold_met"
    assert report["validationGate"]["reason"] == "no_validation_confidence_threshold_met"
    assert report["selectedConfidenceThreshold"] is None
    assert paths.status.exists() is False
    attempt = _read_json(paths.attempt)
    assert attempt["status"] == "validation_failed"
    assert attempt["candidateStatus"] == "rejected_validation"
    assert attempt["promotionReason"] == "no_validation_confidence_threshold_met"
    assert attempt["comboSnapshot"] == _combo_snapshot()
    assert staging_status["status"] == "validation_failed"
    assert staging_status["candidateStatus"] == "rejected_validation"
    assert status["status"] == "untrained"
    assert status["activeModelStatus"] == "untrained"
    assert status["lastAttemptStatus"] == "validation_failed"
    assert status["validationFailureReason"] == "no_validation_confidence_threshold_met"


def test_train_lstm_model_keeps_old_active_model_when_validation_fails(monkeypatch) -> None:
    artifact_root = _runtime_path("old-active-validation-failed")
    _write_report_gate_artifacts(artifact_root)
    report = train_lstm_model(
        LstmTrainingConfig(symbol="BTCUSDT", duration="10m", feature_window=8, min_samples=30, epochs=1),
        artifact_root=artifact_root,
        backend=_LowConfidenceBackend(),
        dataset_builder=_fake_dataset,
    )
    paths = artifact_paths("BTCUSDT", "10m", artifact_root)
    _patch_combo_ranking(monkeypatch)

    status = lstm_prediction_service.lstm_model_status("BTCUSDT", "10m", artifact_root=artifact_root)

    assert report["status"] == "validation_failed"
    assert _read_json(paths.status)["status"] == "trained"
    assert status["activeModelStatus"] == "trained"
    assert status["lastAttemptStatus"] == "validation_failed"
    assert status["validationFailureReason"] == "no_validation_confidence_threshold_met"


def test_train_lstm_model_publishes_active_artifact_when_validation_passes(monkeypatch) -> None:
    artifact_root = _runtime_path("validation-passed")
    report = train_lstm_model(
        LstmTrainingConfig(symbol="BTCUSDT", duration="10m", feature_window=8, min_samples=30, epochs=1),
        artifact_root=artifact_root,
        backend=_HighConfidenceBackend(),
        dataset_builder=_fake_dataset,
    )
    paths = artifact_paths("BTCUSDT", "10m", artifact_root)
    _patch_combo_ranking(monkeypatch)

    status = lstm_prediction_service.lstm_model_status("BTCUSDT", "10m", artifact_root=artifact_root)

    assert report["status"] == "trained"
    assert report["candidateStatus"] == "promoted_active"
    assert report["promotionReason"] == "validation_gate_passed"
    assert paths.model.exists()
    assert status["activeModelStatus"] == "trained"
    assert status["lastAttemptStatus"] == "trained"
    assert status["candidateStatus"] == "promoted_active"
    assert status["selectedConfidenceThreshold"] is not None


def _staging_status(root: Path, model_version: str) -> dict:
    paths = artifact_paths("BTCUSDT", "10m", root)
    return _read_json(paths.root / "_staging" / model_version / "status.json")


class _PredictOnlyBackend:
    def predict(self, _model_path, x):
        return np.asarray([0.7], dtype=np.float32)


class _LowConfidenceBackend:
    def train(self, train_x, train_y, val_x, val_y, *, options, model_path):
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"fake-lstm")
        return {"trainLoss": 0.1, "valLoss": 0.1}

    def predict(self, _model_path, x):
        return np.full(len(x), 0.51, dtype=np.float32)


class _HighConfidenceBackend(_LowConfidenceBackend):
    def predict(self, _model_path, x):
        probs = np.where(x[:, 0, 0] > 0.0, 0.9, 0.1)
        return probs.astype(np.float32)


def _fake_dataset(config: LstmTrainingConfig) -> LstmDataset:
    sample_count = 400
    y = (np.arange(sample_count) % 2 == 0).astype(np.float32)
    x = np.zeros((sample_count, config.feature_window, 2), dtype=np.float32)
    x[:, :, 0] = np.where(y[:, None] > 0, 1.0, -1.0)
    returns = np.where(y > 0, 0.02, -0.02).astype(np.float32)
    times = np.arange(sample_count, dtype=np.int64) * 600_000
    frame = pd.DataFrame({"entry_open_time": times})
    return LstmDataset(x, y, returns, times, ["signal", "noise"], frame, _combo_snapshot())


def _combo_snapshot() -> list[dict]:
    return lstm_combo_snapshot.combo_snapshot_from_ranking(_combo_ranking())


def _combo_ranking() -> dict:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "ranking": [
            {"factorName": "combo_a", "members": [{"name": "factor_a"}]},
            {"factorName": "combo_b", "members": [{"name": "factor_b"}]},
            {"factorName": "combo_c", "members": [{"name": "factor_c"}]},
        ],
    }


def _patch_combo_ranking(monkeypatch) -> None:
    monkeypatch.setattr(lstm_combo_snapshot, "resolve_lstm_combo_ranking", lambda *_args, **_kwargs: _combo_ranking())


def _live_window(*_args, **_kwargs):
    return np.ones((1, 4, 1), dtype=np.float32), {"entryOpenTime": ENTRY_OPEN_TIME, "entryPrice": 100.0}


def _write_validation_failed_artifacts(root: Path) -> None:
    model_dir = root / "BTCUSDT" / "10m"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.pt").write_bytes(b"fake")
    _write_json(model_dir / "features.json", {
        "columns": ["x"],
        "featureWindow": 4,
        "comboSnapshot": _combo_snapshot(),
    })
    _write_json(model_dir / "scaler.json", {"mean": [0.0], "std": [1.0]})
    _write_json(model_dir / "model_version.json", {
        "modelVersion": "lstm_test",
        "trainedAt": "2026-05-13T00:00:00+00:00",
        "returnStats": {"upMean": 0.01, "downMean": -0.01},
    })
    _write_json(model_dir / "status.json", {
        "status": "validation_failed",
        "symbol": "BTCUSDT",
        "duration": "10m",
        "reason": "no_validation_confidence_threshold_met",
    })
    _write_json(model_dir / "training_report.json", {})


def _write_legacy_trained_artifacts(root: Path) -> None:
    model_dir = root / "BTCUSDT" / "10m"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.pt").write_bytes(b"fake")
    _write_json(model_dir / "features.json", {
        "columns": ["x"],
        "featureWindow": 4,
        "comboSnapshot": _combo_snapshot(),
    })
    _write_json(model_dir / "scaler.json", {"mean": [0.0], "std": [1.0]})
    _write_json(model_dir / "model_version.json", {
        "modelVersion": "lstm_legacy",
        "trainedAt": "2026-05-13T00:00:00+00:00",
        "returnStats": {"upMean": 0.01, "downMean": -0.01},
    })
    _write_json(model_dir / "status.json", {
        "status": "trained",
        "symbol": "BTCUSDT",
        "duration": "10m",
    })
    _write_json(model_dir / "training_report.json", {"status": "trained"})


def _write_report_gate_artifacts(root: Path) -> None:
    model_dir = root / "BTCUSDT" / "10m"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.pt").write_bytes(b"fake")
    _write_json(model_dir / "features.json", {
        "columns": ["x"],
        "featureWindow": 4,
        "comboSnapshot": _combo_snapshot(),
    })
    _write_json(model_dir / "scaler.json", {"mean": [0.0], "std": [1.0]})
    _write_json(model_dir / "model_version.json", {
        "modelVersion": "lstm_report_gate",
        "trainedAt": "2026-05-13T00:00:00+00:00",
        "returnStats": {"upMean": 0.01, "downMean": -0.01},
    })
    _write_json(model_dir / "status.json", {
        "status": "trained",
        "symbol": "BTCUSDT",
        "duration": "10m",
    })
    _write_json(model_dir / "training_report.json", {
        "status": "trained",
        "validationGate": {"status": "passed", "minConfidence": 0.6},
        "selectedConfidenceThreshold": 0.6,
    })


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_path(name: str) -> Path:
    path = Path(__file__).resolve().parents[1] / "runtime" / "pytest-temp" / f"lstm-{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
