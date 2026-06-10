from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.services import model_family_training_impl as training_impl
from app.services import model_family_torch_backend as backend
from app.services.lstm_feature_builder import LstmDataset
from app.services.model_family_config import ModelFamilyTrainingConfig
from app.services.model_family_torch_backend import TorchSequenceBackend, TorchSequenceOptions


def test_torch_backend_trains_and_predicts_in_batches(monkeypatch, tmp_path: Path) -> None:
    seen_batch_rows = []
    original_tensor_batch = backend._tensor_batch

    def tracked_tensor_batch(torch, values, idx, scaler=None):
        rows = values[idx].shape[0]
        seen_batch_rows.append(int(rows))
        return original_tensor_batch(torch, values, idx, scaler)

    monkeypatch.setattr(backend, "_tensor_batch", tracked_tensor_batch)
    trainer = TorchSequenceBackend()
    options = TorchSequenceOptions(
        family="cnn",
        input_size=3,
        hidden_size=4,
        num_layers=1,
        learning_rate=0.001,
        batch_size=4,
        epochs=1,
        seed=7,
    )

    trainer.train(
        _features(11),
        _labels(11),
        _features(7),
        _labels(7),
        options=options,
        model_path=tmp_path / "model.pt",
        persist_model=False,
    )
    probabilities = trainer.predict_trained(_features(backend.PREDICT_BATCH_SIZE + 3))

    assert len(probabilities) == backend.PREDICT_BATCH_SIZE + 3
    assert max(seen_batch_rows[:5]) <= options.batch_size
    assert seen_batch_rows[-2:] == [backend.PREDICT_BATCH_SIZE, 3]


def test_torch_training_path_does_not_materialize_scaled_split(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(training_impl, "scaled_split", _raise_scaled_split)
    config = ModelFamilyTrainingConfig(
        family="cnn",
        symbol="BTCUSDT",
        duration="10m",
        feature_window=4,
        min_samples=20,
        epochs=1,
        batch_size=4,
        hidden_size=4,
    )

    report = training_impl.train_model_family(
        config,
        artifact_root=tmp_path,
        dataset_builder=lambda _config: _dataset(30),
        publish_shadow_active=False,
        publish_trade_active=False,
        write_attempt=False,
        persist_artifacts=False,
        evaluate_test=False,
    )

    assert report["modelFamily"] == "cnn"


def _features(rows: int) -> np.ndarray:
    return np.arange(rows * 4 * 3, dtype=np.float32).reshape(rows, 4, 3) / 100.0


def _labels(rows: int) -> np.ndarray:
    return (np.arange(rows) % 2).astype(np.float32)


def _dataset(rows: int) -> LstmDataset:
    return LstmDataset(
        x=_features(rows),
        y=_labels(rows),
        future_returns=np.linspace(-0.01, 0.01, rows, dtype=np.float32),
        entry_open_times=np.arange(rows, dtype=np.int64),
        feature_columns=["a", "b", "c"],
        feature_frame=pd.DataFrame({"a": np.zeros(rows), "b": np.zeros(rows), "c": np.zeros(rows)}),
        combo_snapshot=[],
    )


def _raise_scaled_split(*_args, **_kwargs):
    raise AssertionError("torch training must not materialize scaled_split")
