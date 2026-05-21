from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from app.api import models as models_api
from app.main import app
from app.services.lstm_artifacts import artifact_paths
from app.services.lstm_feature_builder import LstmDataset
from app.services.model_family_candidates import (
    attempted_model_search_keys,
    candidate_library_path,
    record_model_candidate,
)
from app.services.model_family_config import (
    ModelFamilyTrainingConfig,
    model_family_strategy_key,
    parse_model_family_strategy,
)
from app.services.model_family_search_rules import model_family_training_rules
from app.services.model_family_status_service import model_family_status
from app.services.model_family_training_service import train_model_family
from app.services import model_family_candidate_search_service as search_service
from app.services import model_family_candidate_publisher as candidate_publisher
from app.services.model_family_joblib_backend import JoblibModelOptions, XGBOOST_TREE_METHOD, _xgboost
from app.services.lstm_training_gate import validation_gate


def test_model_family_strategy_key_round_trips() -> None:
    key = model_family_strategy_key("xgboost", "10m")

    assert key == "factor_xgboost_shadow_10m"
    assert parse_model_family_strategy(key) == ("xgboost", "10m")
    assert parse_model_family_strategy("factor_rl_strategy_shadow_60m") == ("rl_strategy", "60m")


def test_candidate_libraries_are_family_scoped(tmp_path) -> None:
    lstm = ModelFamilyTrainingConfig(family="lstm", symbol="BTCUSDT", duration="10m")
    xgboost = ModelFamilyTrainingConfig(family="xgboost", symbol="BTCUSDT", duration="10m")
    report = {"status": "shadow_active", "validation": {}, "test": {}}

    record_model_candidate(lstm, "fast", report, artifact_root=tmp_path)
    record_model_candidate(xgboost, "fast", report, artifact_root=tmp_path)

    assert candidate_library_path("lstm", "BTCUSDT", "10m", artifact_root=tmp_path).name == "lstm_candidate_library.json"
    assert candidate_library_path("xgboost", "BTCUSDT", "10m", artifact_root=tmp_path).name == "xgboost_candidate_library.json"
    assert candidate_library_path("lstm", "BTCUSDT", "10m", artifact_root=tmp_path).exists()
    assert candidate_library_path("xgboost", "BTCUSDT", "10m", artifact_root=tmp_path).exists()


def test_failed_candidate_records_are_retryable(tmp_path) -> None:
    config = ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 5})
    failed = {"status": "failed", "validation": {}, "test": {}}
    rejected = {"status": "validation_failed", "validation": {}, "test": {}}

    record_model_candidate(config, "fast", failed, artifact_root=tmp_path)

    assert attempted_model_search_keys("knn", "BTCUSDT", "10m", artifact_root=tmp_path) == frozenset()

    record_model_candidate(config, "fast", rejected, artifact_root=tmp_path)

    assert len(attempted_model_search_keys("knn", "BTCUSDT", "10m", artifact_root=tmp_path)) == 1


def test_models_route_is_family_scoped_and_lstm_alias_is_removed(monkeypatch) -> None:
    monkeypatch.setattr(
        models_api,
        "model_family_status",
        lambda family, symbol, duration: {
            "modelFamily": family,
            "symbol": symbol,
            "duration": duration,
            "strategyKey": model_family_strategy_key(family, duration),
        },
    )
    response = models_api.model_status("gru", symbol="BTCUSDT", duration="10m")
    paths = {getattr(route, "path", "") for route in app.routes}

    assert response["strategyKey"] == "factor_gru_shadow_10m"
    assert "/api/models/{family}/status" in paths
    assert "/api/lstm/status" not in paths


def test_validation_gate_requires_win_rate_above_70() -> None:
    metrics = {
        "confidenceThresholds": [
            {"minConfidence": 0.70, "sampleCount": 50, "winRate": 0.70, "profitFactor": 2.0, "avgReturn": 0.01},
        ]
    }

    assert validation_gate(metrics, metrics)["status"] == "failed"


def test_model_family_train_can_publish_initial_baseline(tmp_path) -> None:
    config = ModelFamilyTrainingConfig(
        family="knn",
        symbol="BTCUSDT",
        duration="10m",
        feature_window=8,
        min_samples=30,
        epochs=1,
    )

    report = train_model_family(
        config,
        artifact_root=tmp_path,
        backend=_LowConfidenceBackend(),
        dataset_builder=_fake_dataset,
        publish_initial_baseline=True,
    )
    paths = artifact_paths("BTCUSDT", "10m", tmp_path, family="knn")
    status = model_family_status("knn", "BTCUSDT", "10m", artifact_root=tmp_path)

    assert report["status"] == "initial_baseline"
    assert report["candidateStatus"] == "promoted_initial_baseline"
    assert report["validationGate"]["status"] == "failed"
    assert paths.status.exists()
    assert status["activeModelStatus"] == "initial_baseline"
    assert status["shadowPredictionReady"] is True
    assert status["tradePredictionReady"] is False


def test_each_model_family_has_full_parallel_training_rules() -> None:
    families = ("lstm", "gru", "cnn", "transformer", "random_forest", "xgboost", "svm", "rl_strategy", "bayesian", "knn")

    rules = [model_family_training_rules(family) for family in families]

    assert all(rule["searchMode"] == "full_parallel" for rule in rules)
    assert all(rule["targetWinRateExclusive"] == 0.70 for rule in rules)
    assert all(rule["searchSpaceTotal"] > 0 for rule in rules)
    assert model_family_training_rules("lstm")["searchSpaceTotal"] == 225
    assert "nEstimators" in model_family_training_rules("random_forest")["candidateSearchAxes"]
    assert "stateBins" in model_family_training_rules("rl_strategy")["candidateSearchAxes"]


def test_candidate_search_publishes_best_trade_candidate(monkeypatch) -> None:
    configs = [
        ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 5}),
        ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 9}),
    ]
    reports = [
        _report("trade_active", 0.72, 1),
        _report("trade_active", 0.81, 2),
    ]
    published = []
    training_calls = []

    def fake_train_model_family(config, **_kwargs):
        training_calls.append((config, _kwargs))
        return reports[configs.index(config)]

    monkeypatch.setattr(search_service, "attempted_model_search_keys", lambda *_args: frozenset())
    monkeypatch.setattr(search_service, "next_model_candidate_configs", lambda *_args: configs)
    monkeypatch.setattr(search_service, "start_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "complete_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "finish_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "record_model_candidate", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "train_model_family", fake_train_model_family)
    monkeypatch.setattr(candidate_publisher, "train_model_family", fake_train_model_family)

    result = search_service.run_model_candidate_search(
        search_service.ModelCandidateSearchConfig("knn", "BTCUSDT", "10m", "fast", parallel_workers=1)
    )

    assert result["status"] == "trade_active"
    assert training_calls[-1][0] == configs[1]
    assert training_calls[-1][1] == {}
    assert len(published) == 0


def test_candidate_search_publishes_initial_baseline_when_untrained(monkeypatch) -> None:
    configs = [
        ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 5}),
        ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 9}),
    ]
    reports = [_report("validation_failed", 0.52, 1), _report("validation_failed", 0.54, 2)]
    training_calls = []

    def fake_train_model_family(config, **kwargs):
        training_calls.append((config, kwargs))
        report = reports[configs.index(config)]
        if kwargs.get("publish_initial_baseline"):
            return {**report, "status": "initial_baseline", "candidateStatus": "promoted_initial_baseline"}
        return report

    monkeypatch.setattr(search_service, "attempted_model_search_keys", lambda *_args: frozenset())
    monkeypatch.setattr(search_service, "next_model_candidate_configs", lambda *_args: configs)
    monkeypatch.setattr(search_service, "start_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "complete_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "finish_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "record_model_candidate", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "train_model_family", fake_train_model_family)
    monkeypatch.setattr(candidate_publisher, "model_family_status", lambda *_args: {"activeModelStatus": "untrained"})
    monkeypatch.setattr(candidate_publisher, "train_model_family", fake_train_model_family)

    result = search_service.run_model_candidate_search(
        search_service.ModelCandidateSearchConfig("knn", "BTCUSDT", "10m", "fast", parallel_workers=1)
    )

    assert result["status"] == "initial_baseline"
    assert training_calls[-1][0] == configs[1]
    assert training_calls[-1][1]["publish_initial_baseline"] is True


def test_exhausted_candidate_search_uses_library_status(monkeypatch) -> None:
    captured = {}
    library = {"records": [_report("validation_failed", 0.52, 1), _report("validation_failed", 0.54, 2)]}
    base = ModelFamilyTrainingConfig(family="random_forest", symbol="BTCUSDT", duration="60m")

    monkeypatch.setattr(search_service, "model_training_config_for_profile", lambda *_args: base)
    monkeypatch.setattr(search_service, "attempted_model_search_keys", lambda *_args: frozenset({"a", "b"}))
    monkeypatch.setattr(search_service, "next_model_candidate_configs", lambda *_args: [])
    monkeypatch.setattr(search_service, "read_model_candidate_library", lambda *_args: library)
    monkeypatch.setattr(search_service, "read_model_candidate_progress", lambda *_args: {"status": "failed", "completed": 1})
    monkeypatch.setattr(search_service, "finish_model_candidate_progress_from_library", lambda *args, **kwargs: captured.update(kwargs))

    result = search_service.run_model_candidate_search(
        search_service.ModelCandidateSearchConfig("random_forest", "BTCUSDT", "60m", "full")
    )

    assert result["status"] == "validation_failed"
    assert captured["status"] == "validation_failed"


def test_xgboost_process_worker_override_is_explicit(monkeypatch) -> None:
    monkeypatch.setenv(search_service.XGBOOST_PROCESS_WORKERS_ENV, "3")

    assert search_service._process_worker_count(10) == 3


def test_xgboost_process_worker_override_rejects_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv(search_service.XGBOOST_PROCESS_WORKERS_ENV, "0")

    with pytest.raises(ValueError):
        search_service._process_worker_count(10)


def test_xgboost_uses_hist_tree_method() -> None:
    model = _xgboost(JoblibModelOptions("xgboost", 20260513, {}))

    assert model.get_params()["tree_method"] == XGBOOST_TREE_METHOD


def _report(status: str, win_rate: float, version: int) -> dict:
    return {
        "status": status,
        "modelVersion": f"v{version}",
        "validation": {"winRate": win_rate, "profitFactor": 2.0},
        "test": {"winRate": win_rate, "profitFactor": 2.0},
        "sampleCounts": {"test": 60},
    }


class _LowConfidenceBackend:
    def train(self, train_x, train_y, val_x, val_y, *, options, model_path, persist_model=True):
        if persist_model:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"fake-model")
        return {"trainLoss": None, "valLoss": None}

    def predict_trained(self, x):
        return np.full(len(x), 0.51, dtype=np.float32)


def _fake_dataset(config: ModelFamilyTrainingConfig) -> LstmDataset:
    sample_count = 400
    y = (np.arange(sample_count) % 2 == 0).astype(np.float32)
    x = np.zeros((sample_count, config.feature_window, 2), dtype=np.float32)
    returns = np.where(y > 0, 0.01, -0.01).astype(np.float32)
    return LstmDataset(
        x=x,
        y=y,
        future_returns=returns,
        entry_open_times=np.arange(sample_count, dtype=np.int64),
        feature_columns=["a", "b"],
        feature_frame=pd.DataFrame({"a": np.zeros(sample_count), "b": np.zeros(sample_count)}),
        combo_snapshot=[{"rank": 1, "key": "combo"}],
    )
