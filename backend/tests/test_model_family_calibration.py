from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.lstm_feature_builder import LstmDataset
from app.services.model_family_config import ModelFamilyTrainingConfig
from app.services.model_family_training_service import train_model_family


def test_model_family_training_records_probability_calibration(tmp_path) -> None:
    report = train_model_family(
        ModelFamilyTrainingConfig(
            family="knn",
            symbol="BTCUSDT",
            duration="10m",
            feature_window=4,
            min_samples=30,
            epochs=1,
        ),
        artifact_root=tmp_path,
        backend=_CalibrationBackend(),
        dataset_builder=_dataset,
        persist_artifacts=False,
        publish_shadow_active=False,
        publish_trade_active=False,
    )

    calibration = report["probabilityCalibration"]
    assert calibration["calibrator"]["status"] == "fitted"
    assert calibration["validation"]["brierScore"] >= 0.0
    assert calibration["test"]["buckets"]
    assert report["probabilitySource"] == "calibrated_platt"
    assert "rawValidation" in report


class _CalibrationBackend:
    def train(self, *_args, **_kwargs):
        return {"trainLoss": None, "valLoss": None}

    def predict_trained(self, x):
        signal = x[:, -1, 0]
        return np.where(signal > 0, 0.65, 0.35).astype(np.float32)


def _dataset(config: ModelFamilyTrainingConfig) -> LstmDataset:
    sample_count = 180
    y = (np.arange(sample_count) % 2 == 0).astype(np.float32)
    x = np.zeros((sample_count, config.feature_window, 1), dtype=np.float32)
    x[:, :, 0] = np.where(y[:, None] > 0, 1.0, -1.0)
    returns = np.where(y > 0, 0.02, -0.02).astype(np.float32)
    return LstmDataset(
        x=x,
        y=y,
        future_returns=returns,
        entry_open_times=np.arange(sample_count, dtype=np.int64),
        feature_columns=["signal"],
        feature_frame=pd.DataFrame({"signal": np.where(y > 0, 1.0, -1.0)}),
        combo_snapshot=[],
    )
