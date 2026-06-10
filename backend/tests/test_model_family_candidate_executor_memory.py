from __future__ import annotations

import numpy as np
import pandas as pd

from app.services import model_family_candidate_executor as executor
from app.services.lstm_feature_builder import LstmDataset
from app.services.model_family_config import ModelFamilyTrainingConfig


def test_default_dataset_builder_uses_candidate_process_boundary() -> None:
    cache = executor.CandidateDatasetCache()
    config = ModelFamilyTrainingConfig(family="lightgbm", symbol="BTCUSDT", duration="10m")

    assert executor._uses_process_executor([config], cache.build) is True


def test_single_candidate_training_does_not_open_process_pool(monkeypatch) -> None:
    cache = executor.CandidateDatasetCache()
    config = ModelFamilyTrainingConfig(family="lightgbm", symbol="BTCUSDT", duration="10m")
    calls = []

    def fail_process_pool(*_args, **_kwargs):
        raise AssertionError("single-candidate training should stay in the worker process")

    monkeypatch.setattr(executor, "ProcessPoolExecutor", fail_process_pool)
    monkeypatch.setattr(executor, "record_model_candidate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor, "train_model_family", lambda cfg, **_kwargs: calls.append(cfg) or _training_report())

    results = list(executor.train_candidate_reports([config], "full", 1, cache.build, stage="coarse"))

    assert len(results) == 1
    assert calls == [config]
    assert results[0].report["status"] == "validation_failed"


def test_candidate_training_clears_dataset_cache_after_candidate(monkeypatch) -> None:
    cache = executor.CandidateDatasetCache()
    config = ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m")

    monkeypatch.setattr(executor, "build_lstm_training_dataset", lambda _config: _dataset())
    monkeypatch.setattr(executor, "record_model_candidate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor, "train_model_family", _train_with_dataset_builder)

    report = executor.train_candidate(config, "full", cache.build, stage="coarse", record_config=config)

    assert report["status"] == "validation_failed"
    assert cache.datasets == {}


def _train_with_dataset_builder(config, **kwargs):
    kwargs["dataset_builder"](config)
    return {"status": "validation_failed", "candidateStatus": "rejected"}


def _training_report() -> dict:
    return {"status": "validation_failed", "candidateStatus": "rejected"}


def _dataset() -> LstmDataset:
    x = np.zeros((4, 2, 2), dtype=np.float32)
    y = np.zeros(4, dtype=np.float32)
    return LstmDataset(
        x=x,
        y=y,
        future_returns=y.copy(),
        entry_open_times=np.arange(4, dtype=np.int64),
        feature_columns=["a", "b"],
        feature_frame=pd.DataFrame({"a": [0.0], "b": [0.0]}),
        combo_snapshot=[],
    )
