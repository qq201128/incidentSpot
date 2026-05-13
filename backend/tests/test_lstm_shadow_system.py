from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services import auto_predict_service
from app.services import lstm_prediction_service
from app.services import lstm_shadow_learning
from app.services.auto_trade_types import AutoTradeSettings
from app.services.factor_combo_simulation_keys import factor_combo_shadow_strategy_key
from app.services.lstm_config import LstmTrainingConfig, lstm_shadow_strategy_key
from app.services.lstm_feature_builder import LstmDataset, duration_labeled_frame
from app.services.lstm_prediction_service import predict_lstm_signal
from app.services.lstm_training_service import train_lstm_model
from app.services.rule_config import DURATION_TO_MINUTES
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY

ENTRY_OPEN_TIME = 1778121600000


def test_duration_labeled_frame_uses_period_specific_future_return() -> None:
    frame = pd.DataFrame({
        "open_time": np.arange(30) * 60_000,
        "close": 100 + np.arange(30),
        "feature_a": np.arange(30, dtype=float),
    })

    labeled = duration_labeled_frame(frame, "10m", 10, 0.0)

    assert labeled["open_time"].iloc[0] == 0
    assert labeled["entry_open_time"].iloc[0] == 60_000
    assert labeled["open_time"].iloc[-1] == 19 * 60_000
    assert labeled.loc[0, "future_return"] == pytest.approx(110 / 100 - 1)


def test_train_lstm_model_writes_separate_artifacts() -> None:
    artifact_root = _runtime_path("artifacts")
    config = LstmTrainingConfig(
        symbol="BTCUSDT",
        duration="10m",
        feature_window=8,
        min_samples=30,
        epochs=1,
    )

    report = train_lstm_model(
        config,
        artifact_root=artifact_root,
        backend=_FakeBackend(),
        dataset_builder=_fake_dataset,
    )

    model_dir = artifact_root / "BTCUSDT" / "10m"
    assert report["status"] == "trained"
    assert (model_dir / "model.pt").exists()
    assert (model_dir / "training_report.json").exists()
    assert (model_dir / "scaler.json").exists()
    assert (model_dir / "features.json").exists()
    assert (model_dir / "model_version.json").exists()


def test_predict_lstm_signal_missing_model_raises() -> None:
    with pytest.raises(ValueError, match="not ready"):
        predict_lstm_signal("BTCUSDT", "10m", artifact_root=_runtime_path("missing"))


def test_legacy_validation_failed_status_is_treated_as_trained(monkeypatch) -> None:
    artifact_root = _runtime_path("legacy-status")
    _write_legacy_validation_artifacts(artifact_root)
    monkeypatch.setattr(lstm_prediction_service, "build_live_feature_window", _live_window)

    signal = lstm_prediction_service.predict_lstm_signal(
        "BTCUSDT",
        "10m",
        artifact_root=artifact_root,
        backend=_PredictOnlyBackend(),
    )
    prediction = lstm_prediction_service.predict_lstm_shadow_prediction.__globals__["_prediction_payload"](signal)

    assert signal["modelStatus"] == "trained"
    assert prediction["trade_quality_passed"] is True
    assert prediction["high_winrate_gate_passed"] is True


def test_lstm_shadow_learning_counts_lstm_and_top_comparison(monkeypatch) -> None:
    db_path = _runtime_path("db") / "predictions.db"
    _create_prediction_db(db_path)
    monkeypatch.setattr(lstm_shadow_learning, "get_conn", lambda: _connect(db_path))

    summary = lstm_shadow_learning.lstm_shadow_learning_summary("BTCUSDT", "10m")

    assert summary["strategyKey"] == lstm_shadow_strategy_key("10m")
    assert summary["sampleCount"] == 2
    assert summary["winRate"] == 0.5
    assert [row["strategyKey"] for row in summary["comparison"]] == [
        FACTOR_COMBO_STRATEGY_KEY,
        factor_combo_shadow_strategy_key(2),
        factor_combo_shadow_strategy_key(3),
        lstm_shadow_strategy_key("10m"),
    ]


def test_auto_predict_saves_lstm_shadow_after_factor_combo(monkeypatch) -> None:
    saved = []

    async def save_prediction(result: dict, _lock, *, allow_existing: bool = False) -> bool:
        saved.append(result["strategy_key"])
        return True

    monkeypatch.setattr(auto_predict_service, "is_lstm_shadow_ready", lambda *_args: True)
    monkeypatch.setattr(auto_predict_service, "predict_lstm_shadow_prediction", _lstm_prediction)
    monkeypatch.setattr(auto_predict_service, "_save_prediction", save_prediction)

    import asyncio

    asyncio.run(auto_predict_service._save_lstm_shadow_prediction(
        _settings(FACTOR_COMBO_STRATEGY_KEY),
        ENTRY_OPEN_TIME,
        asyncio.Lock(),
    ))

    assert saved == [lstm_shadow_strategy_key("10m")]


class _FakeBackend:
    def train(self, train_x, train_y, val_x, val_y, *, options, model_path):
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"fake-lstm")
        return {"trainLoss": 0.1, "valLoss": 0.1}

    def predict(self, _model_path, x):
        return np.where(x[:, -1, 0] > 0, 0.8, 0.2).astype(np.float32)


class _PredictOnlyBackend:
    def predict(self, _model_path, x):
        return np.asarray([0.7], dtype=np.float32)


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
    return [{"rank": 1, "factorName": "combo_a", "members": ["factor_a"]}]


def _create_prediction_db(path: Path) -> None:
    conn = _connect(path)
    conn.executescript(
        """
        CREATE TABLE predictions (
          strategy_key TEXT, symbol TEXT, duration TEXT, open_time INTEGER,
          direction TEXT, actual_return REAL, prediction_correct INTEGER,
          model_version TEXT, feature_window INTEGER, settled_at TEXT
        );
        """
    )
    rows = [
        (FACTOR_COMBO_STRATEGY_KEY, "BTCUSDT", "10m", 1, "up", 0.01, 1, None, None, "x"),
        (factor_combo_shadow_strategy_key(2), "BTCUSDT", "10m", 2, "up", -0.01, 0, None, None, "x"),
        (factor_combo_shadow_strategy_key(3), "BTCUSDT", "10m", 3, "up", 0.01, 1, None, None, "x"),
        (lstm_shadow_strategy_key("10m"), "BTCUSDT", "10m", 4, "up", 0.01, 1, "v1", 64, "x"),
        (lstm_shadow_strategy_key("10m"), "BTCUSDT", "10m", 5, "down", -0.01, 0, "v1", 64, "x"),
    ]
    conn.executemany("INSERT INTO predictions VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _settings(strategy_key: str) -> AutoTradeSettings:
    return AutoTradeSettings(
        strategy_key=strategy_key,
        enabled=True,
        symbol="BTCUSDT",
        duration="10m",
        duration_minutes=DURATION_TO_MINUTES["10m"],
        qty=5.0,
        live_trading_enabled=False,
    )


def _lstm_prediction(*_args, **_kwargs) -> dict:
    return {"strategy_key": lstm_shadow_strategy_key("10m")}


def _live_window(*_args, **_kwargs):
    return np.ones((1, 4, 1), dtype=np.float32), {"entryOpenTime": ENTRY_OPEN_TIME, "entryPrice": 100.0}


def _write_legacy_validation_artifacts(root: Path) -> None:
    model_dir = root / "BTCUSDT" / "10m"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.pt").write_bytes(b"fake")
    _write_json(model_dir / "features.json", {"columns": ["x"], "featureWindow": 4})
    _write_json(model_dir / "scaler.json", {"mean": [0.0], "std": [1.0]})
    _write_json(model_dir / "model_version.json", {
        "modelVersion": "lstm_test",
        "trainedAt": "2026-05-13T00:00:00+00:00",
        "returnStats": {"upMean": 0.01, "downMean": -0.01},
    })
    _write_json(model_dir / "status.json", {"status": "validation_failed", "symbol": "BTCUSDT", "duration": "10m"})
    _write_json(model_dir / "training_report.json", {})


def _write_json(path: Path, payload: dict) -> None:
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")


def _runtime_path(name: str) -> Path:
    path = Path(__file__).resolve().parents[1] / "runtime" / "pytest-temp" / f"lstm-{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
