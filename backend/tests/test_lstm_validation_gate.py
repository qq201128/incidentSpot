from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services import lstm_combo_snapshot
from app.services import lstm_prediction_service
from app.services.lstm_config import LstmTrainingConfig
from app.services.lstm_feature_builder import LstmDataset
from app.services.lstm_training_service import train_lstm_model

ENTRY_OPEN_TIME = 1778121600000


def test_validation_failed_status_blocks_prediction(monkeypatch) -> None:
    artifact_root = _runtime_path("legacy-status")
    _write_validation_failed_artifacts(artifact_root)
    monkeypatch.setattr(lstm_combo_snapshot, "get_cached_combination_ranking", lambda *_args: _combo_ranking())
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


def test_train_lstm_model_marks_validation_failed_when_no_threshold_passes() -> None:
    report = train_lstm_model(
        LstmTrainingConfig(symbol="BTCUSDT", duration="10m", feature_window=8, min_samples=30, epochs=1),
        artifact_root=_runtime_path("validation-failed"),
        backend=_LowConfidenceBackend(),
        dataset_builder=_fake_dataset,
    )

    assert report["status"] == "validation_failed"
    assert report["validationGate"]["reason"] == "no_validation_confidence_threshold_met"
    assert report["selectedConfidenceThreshold"] is None


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


def _fake_dataset(config: LstmTrainingConfig) -> LstmDataset:
    sample_count = 60
    y = (np.arange(sample_count) % 2 == 0).astype(np.float32)
    x = np.zeros((sample_count, config.feature_window, 2), dtype=np.float32)
    x[:, :, 0] = np.where(y[:, None] > 0, 1.0, -1.0)
    returns = np.where(y > 0, 0.02, -0.02).astype(np.float32)
    times = np.arange(sample_count, dtype=np.int64) * 600_000
    frame = pd.DataFrame({"entry_open_time": times})
    return LstmDataset(x, y, returns, times, ["signal", "noise"], frame, _combo_snapshot())


def _combo_snapshot() -> list[dict]:
    return [
        {"rank": 1, "factorName": "combo_a", "members": ["factor_a"]},
        {"rank": 2, "factorName": "combo_b", "members": ["factor_b"]},
        {"rank": 3, "factorName": "combo_c", "members": ["factor_c"]},
    ]


def _combo_ranking() -> dict:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "ranking": [
            {"factorName": row["factorName"], "members": [{"name": name} for name in row["members"]]}
            for row in _combo_snapshot()
        ],
    }


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


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _runtime_path(name: str) -> Path:
    path = Path(__file__).resolve().parents[1] / "runtime" / "pytest-temp" / f"lstm-{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
