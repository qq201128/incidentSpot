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
    model_search_space_size,
    record_model_candidate,
)
from app.services.model_family_config import (
    ModelFamilyTrainingConfig,
    model_family_strategy_key,
    parse_model_family_strategy,
)
from app.services.model_family_search_rules import model_family_training_rules
from app.services.model_family_status_service import model_family_status
from app.services import model_family_status_service
from app.services import model_family_status_progress
from app.services.model_family_training_service import train_model_family
from app.services.model_family_training_payloads import backend_options
from app.services.model_family_prediction_service import _prediction_payload, _signal_payload
from app.services import model_family_candidate_search_service as search_service
from app.services import model_family_candidate_executor as candidate_executor
from app.services import model_family_candidate_halving as candidate_halving
from app.services import model_family_candidate_publisher as candidate_publisher
from app.services import model_family_walk_forward
from app.services.model_family_joblib_backend import JoblibModelOptions, XGBOOST_TREE_METHOD, _xgboost
from app.services.model_family_joblib_extra_estimators import (
    catboost_estimator,
    extra_trees_estimator,
    lightgbm_estimator,
    logistic_elasticnet_estimator,
)
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
    app_paths = {getattr(route, "path", "") for route in app.routes}
    model_paths = {getattr(route, "path", "") for route in models_api.router.routes}

    assert response["strategyKey"] == "factor_gru_shadow_10m"
    assert "/api/models/{family}/status" in model_paths
    assert "/api/lstm/status" not in app_paths


def test_model_train_route_enqueues_instead_of_training(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        models_api,
        "model_candidate_search",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"status": "queued", "duration": kwargs["duration"]},
    )

    response = models_api.model_train("knn", symbol="btcusdt")

    assert response["status"] == "queued"
    assert response["duration"] == "10m"
    assert calls[0][0] == ("knn",)
    assert calls[0][1]["symbol"] == "btcusdt"
    assert calls[0][1]["reset_history"] is False


def test_validation_gate_requires_win_rate_above_62() -> None:
    metrics = {
        "confidenceThresholds": [
            {"minConfidence": 0.70, "sampleCount": 50, "winRate": 0.62, "profitFactor": 2.0, "avgReturn": 0.01},
        ]
    }

    assert validation_gate(metrics, metrics)["status"] == "failed"


def test_validation_gate_passes_when_win_rate_is_above_62() -> None:
    metrics = {
        "confidenceThresholds": [
            {"minConfidence": 0.70, "sampleCount": 50, "winRate": 0.6201, "profitFactor": 2.0, "avgReturn": 0.01},
        ]
    }

    assert validation_gate(metrics, metrics)["status"] == "passed"


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
    assert status["paperLiveAdmission"]["allowed"] is True
    assert status["paperLiveStatus"] == "paper_collecting"
    assert status["validationRole"] == "validation_gate_or_relative_shadow_observation"
    assert status["realTradingEnabled"] is False


def test_model_family_status_reflects_queued_search_job(monkeypatch) -> None:
    _patch_untrained_model_status(monkeypatch)
    monkeypatch.setattr(
        model_family_status_progress,
        "list_model_search_jobs",
        lambda _filters: [
            {
                "job_id": "job-1",
                "status": "pending",
                "profile": "full",
                "created_at": "2026-05-31T00:00:00+00:00",
                "parallel_workers": 4,
                "internal_threads": 2,
                "xgboost_process_workers": 1,
            }
        ],
    )

    status = model_family_status("knn", "BTCUSDT", "10m")

    progress = status["candidateSearchProgress"]
    assert progress["status"] == "queued"
    assert progress["modelSearchJob"]["job_id"] == "job-1"
    assert progress["searchSpaceTotal"] == model_family_training_rules("knn")["searchSpaceTotal"]
    assert progress["total"] == model_family_training_rules("knn")["searchSpaceTotal"]
    assert progress["parallelWorkers"] == 4
    assert progress["internalThreads"] == 2


def test_model_family_status_prefers_running_job_over_newer_pending_job(monkeypatch) -> None:
    _patch_untrained_model_status(monkeypatch)
    monkeypatch.setattr(
        model_family_status_progress,
        "list_model_search_jobs",
        lambda _filters: [
            {
                "job_id": "job-pending",
                "status": "pending",
                "profile": "full",
                "created_at": "2026-05-31T00:05:00+00:00",
                "parallel_workers": 4,
                "internal_threads": 2,
                "xgboost_process_workers": 1,
            },
            {
                "job_id": "job-running",
                "status": "running",
                "profile": "full",
                "created_at": "2026-05-31T00:00:00+00:00",
                "started_at": "2026-05-31T00:01:00+00:00",
                "parallel_workers": 8,
                "internal_threads": 3,
                "xgboost_process_workers": 1,
            },
        ],
    )

    status = model_family_status("knn", "BTCUSDT", "10m")

    progress = status["candidateSearchProgress"]
    assert progress["status"] == "running"
    assert progress["modelSearchJob"]["job_id"] == "job-running"
    assert progress["parallelWorkers"] == 8
    assert progress["internalThreads"] == 3


def test_model_family_status_marks_stale_runtime_progress_paused(monkeypatch) -> None:
    _patch_untrained_model_status(monkeypatch)
    monkeypatch.setattr(
        model_family_status_progress,
        "read_model_candidate_progress_view",
        lambda *_args, **_kwargs: {"status": "running", "completed": 12, "total": 100},
    )
    monkeypatch.setattr(model_family_status_progress, "list_model_search_jobs", lambda _filters: [])

    status = model_family_status("knn", "BTCUSDT", "10m")

    progress = status["candidateSearchProgress"]
    assert progress["status"] == "paused"
    assert progress["staleRuntimeStatus"] == "running"


def test_model_family_status_recovers_zero_progress_from_candidate_library(tmp_path) -> None:
    config = ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 5})
    record_model_candidate(config, "fast", _report("trade_active", 0.74, 1), artifact_root=tmp_path)

    status = model_family_status(
        "knn",
        "BTCUSDT",
        "10m",
        artifact_root=tmp_path,
        current_combo_snapshot=[],
    )

    progress = status["candidateSearchProgress"]
    assert progress["source"] == "candidate_library"
    assert progress["status"] == "trade_active"
    assert progress["completed"] == 1
    assert progress["total"] == model_search_space_size("knn")
    assert progress["counts"]["tradeActive"] == 1
    assert progress["latestCompleted"]["status"] == "trade_active"


def test_model_family_status_hides_stale_validation_reason_after_successful_candidate(tmp_path) -> None:
    config = ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 5})
    paths = artifact_paths("BTCUSDT", "10m", tmp_path, family="knn")
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.status.write_text(
        '{"status":"initial_baseline","reason":"no_validation_confidence_threshold_met"}',
        encoding="utf-8",
    )
    record_model_candidate(config, "fast", _report("trade_active", 0.74, 1), artifact_root=tmp_path)

    status = model_family_status(
        "knn",
        "BTCUSDT",
        "10m",
        artifact_root=tmp_path,
        current_combo_snapshot=[],
    )

    assert status["activeValidationFailureReason"] == "no_validation_confidence_threshold_met"
    assert status["validationFailureReason"] is None


def _patch_untrained_model_status(monkeypatch) -> None:
    monkeypatch.setattr(model_family_status_service, "read_json", lambda *_args: None)
    monkeypatch.setattr(model_family_status_service, "required_artifacts_exist", lambda _paths: False)
    monkeypatch.setattr(model_family_status_progress, "read_model_candidate_progress_view", _empty_candidate_progress)
    monkeypatch.setattr(
        model_family_status_service,
        "model_candidate_library_summary",
        lambda *_args, **_kwargs: {"total": 0},
    )
    monkeypatch.setattr(model_family_status_service, "current_combo_snapshot", lambda *_args: [])


def _empty_candidate_progress(*_args, **_kwargs) -> dict:
    return {"status": "idle", "total": 0}


def test_each_model_family_has_successive_halving_training_rules() -> None:
    families = (
        "lstm",
        "gru",
        "cnn",
        "transformer",
        "random_forest",
        "extra_trees",
        "xgboost",
        "lightgbm",
        "catboost",
        "logistic_elasticnet",
        "svm",
        "rl_strategy",
        "bayesian",
        "knn",
    )

    rules = [model_family_training_rules(family) for family in families]

    assert all(rule["searchMode"] == "successive_halving" for rule in rules)
    assert all([stage["stage"] for stage in rule["successiveHalving"]] == ["coarse", "full", "walk_forward"] for rule in rules)
    assert all(rule["targetWinRateExclusive"] == 0.62 for rule in rules)
    assert all(rule["searchSpaceTotal"] > 0 for rule in rules)
    assert all(rule["internalThreads"] == 1 for rule in rules)
    assert all(rule["parallelWorkers"] == 1 for rule in rules)
    assert all(rule["xgboostProcessWorkers"] == 1 for rule in rules)
    lstm_rules = model_family_training_rules("lstm")
    transformer_rules = model_family_training_rules("transformer")
    assert lstm_rules["searchSpaceTotal"] == 900
    assert transformer_rules["searchSpaceTotal"] == 1800
    assert "dropout" in lstm_rules["candidateSearchAxes"]
    assert "usePositionalEncoding" in transformer_rules["candidateSearchAxes"]
    assert "nEstimators" in model_family_training_rules("random_forest")["candidateSearchAxes"]
    assert "nEstimators" in model_family_training_rules("extra_trees")["candidateSearchAxes"]
    assert "numLeaves" in model_family_training_rules("lightgbm")["candidateSearchAxes"]
    assert "l2LeafReg" in model_family_training_rules("catboost")["candidateSearchAxes"]
    assert "l1Ratio" in model_family_training_rules("logistic_elasticnet")["candidateSearchAxes"]
    assert "stateBins" in model_family_training_rules("rl_strategy")["candidateSearchAxes"]


def test_torch_backend_options_include_regularization_params() -> None:
    config = ModelFamilyTrainingConfig(
        family="transformer",
        symbol="BTCUSDT",
        duration="10m",
        params={
            "dropout": 0.15,
            "weight_decay": 1e-4,
            "early_stopping_patience": 3,
            "class_weight_mode": "balanced",
            "return_weight_mode": "abs_return",
            "transformer_nhead": 8,
            "use_positional_encoding": True,
        },
    )

    options = backend_options(config, input_size=5)

    assert options.dropout == 0.15
    assert options.weight_decay == 1e-4
    assert options.early_stopping_patience == 3
    assert options.class_weight_mode == "balanced"
    assert options.return_weight_mode == "abs_return"
    assert options.transformer_nhead == 8
    assert options.use_positional_encoding is True


def test_model_family_prediction_payload_uses_live_feature_status_metadata() -> None:
    signal = _signal_payload(
        "xgboost",
        "BTCUSDT",
        "10m",
        0.72,
        {
            "entryOpenTime": 123,
            "entryPrice": 100.5,
            "dataFreshnessStatus": "latest_available",
            "missingFeatureStatus": "missing_factor_combo_features",
        },
        {"featureWindow": 8},
        {
            "modelVersion": "xgb_v1",
            "trainedAt": "2026-05-31T00:00:00+00:00",
            "selectedConfidenceThreshold": 0.7,
            "validationGate": {"status": "passed", "validation": {"winRate": 0.71}},
            "returnStats": {"upMean": 0.01, "downMean": -0.01},
        },
        {"status": "trade_active"},
    )

    payload = _prediction_payload(signal)

    assert payload["data_freshness_status"] == "latest_available"
    assert payload["missing_feature_status"] == "missing_factor_combo_features"


def test_candidate_search_publishes_best_trade_candidate(monkeypatch) -> None:
    configs = [
        ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 5}),
        ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 9}),
    ]
    reports = [
        _report("trade_active", 0.72, 1),
        _report("trade_active", 0.81, 2),
    ]
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
    monkeypatch.setattr(search_service, "train_candidate_reports", _candidate_report_iterator(configs, reports))
    monkeypatch.setattr(search_service, "run_walk_forward_stage", _walk_forward_passthrough)
    monkeypatch.setattr(candidate_publisher, "train_model_family", fake_train_model_family)

    result = search_service.run_model_candidate_search(
        search_service.ModelCandidateSearchConfig("knn", "BTCUSDT", "10m", "fast", parallel_workers=1)
    )

    assert result["status"] == "trade_active"
    assert training_calls[-1][0] == configs[1]
    assert training_calls[-1][1] == {}
    assert [stage["stage"] for stage in result["successiveHalvingStages"]] == ["coarse", "full", "walk_forward"]


def test_validation_failed_candidate_stops_after_full_stage() -> None:
    config = ModelFamilyTrainingConfig(family="catboost", symbol="BTCUSDT", duration="30m")
    coarse = candidate_executor.CandidateTrainingResult(config, _report("validation_failed", 0.52, 1))
    coarse_closed = candidate_halving.close_halving_stage([coarse], "coarse")
    full = candidate_executor.CandidateTrainingResult(config, _report("validation_failed", 0.52, 1))
    full_closed = candidate_halving.close_halving_stage([full], "full")

    assert coarse_closed.survivors == [coarse]
    assert full_closed.survivors == []
    assert full_closed.reports[0].report["advancedToNextStage"] is False
    assert full_closed.reports[0].report["eliminationReason"] == "full_training_failed"


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
    monkeypatch.setattr(search_service, "train_candidate_reports", _candidate_report_iterator(configs, reports))
    monkeypatch.setattr(search_service, "run_walk_forward_stage", _walk_forward_passthrough)
    monkeypatch.setattr(candidate_publisher, "model_family_status", lambda *_args: {"activeModelStatus": "untrained"})
    monkeypatch.setattr(candidate_publisher, "train_model_family", fake_train_model_family)

    result = search_service.run_model_candidate_search(
        search_service.ModelCandidateSearchConfig("knn", "BTCUSDT", "10m", "fast", parallel_workers=1)
    )

    assert result["status"] == "initial_baseline"
    assert training_calls[-1][0] == configs[1]
    assert training_calls[-1][1]["publish_initial_baseline"] is True


def test_candidate_search_publishes_relative_shadow_when_candidate_beats_active(monkeypatch) -> None:
    configs = [
        ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 5}),
        ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 9}),
    ]
    reports = [_report("validation_failed", 0.58, 1), _report("validation_failed", 0.61, 2)]
    training_calls = []

    def fake_status(*_args):
        return {
            "activeModelStatus": "shadow_active",
            "validationWinRate": 0.56,
            "validationGate": {"validation": {"winRate": 0.56, "profitFactor": 1.1, "sampleCount": 60}},
        }

    def fake_train_model_family(config, **kwargs):
        training_calls.append((config, kwargs))
        return {"status": "shadow_active", "relativePromotion": {"promoted": True}}

    monkeypatch.setattr(search_service, "attempted_model_search_keys", lambda *_args: frozenset())
    monkeypatch.setattr(search_service, "next_model_candidate_configs", lambda *_args: configs)
    monkeypatch.setattr(search_service, "start_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "complete_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "finish_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "record_model_candidate", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "train_candidate_reports", _candidate_report_iterator(configs, reports))
    monkeypatch.setattr(search_service, "run_walk_forward_stage", _walk_forward_passthrough)
    monkeypatch.setattr(candidate_publisher, "model_family_status", fake_status)
    monkeypatch.setattr(candidate_publisher, "train_model_family", fake_train_model_family)

    result = search_service.run_model_candidate_search(
        search_service.ModelCandidateSearchConfig("knn", "BTCUSDT", "10m", "fast", parallel_workers=1)
    )

    assert result["status"] == "shadow_active"
    assert training_calls[-1][0] == configs[1]
    assert training_calls[-1][1]["active_status_loader"] is candidate_publisher.model_family_status


def test_candidate_search_publishes_relative_shadow_on_win_rate_improvement(monkeypatch) -> None:
    configs = [
        ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 5}),
        ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 9}),
    ]
    weak_profit_factor = _report("validation_failed", 0.61, 2)
    weak_profit_factor["validation"]["profitFactor"] = 0.8
    reports = [_report("validation_failed", 0.58, 1), weak_profit_factor]
    training_calls = []

    def fake_status(*_args):
        return {
            "activeModelStatus": "shadow_active",
            "validationGate": {"validation": {"winRate": 0.60, "profitFactor": 1.2, "sampleCount": 80}},
        }

    def fake_train_model_family(config, **kwargs):
        training_calls.append((config, kwargs))
        return {"status": "shadow_active", "relativePromotion": {"promoted": True}}

    monkeypatch.setattr(search_service, "attempted_model_search_keys", lambda *_args: frozenset())
    monkeypatch.setattr(search_service, "next_model_candidate_configs", lambda *_args: configs)
    monkeypatch.setattr(search_service, "start_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "complete_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "finish_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "record_model_candidate", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "train_candidate_reports", _candidate_report_iterator(configs, reports))
    monkeypatch.setattr(search_service, "run_walk_forward_stage", _walk_forward_passthrough)
    monkeypatch.setattr(candidate_publisher, "model_family_status", fake_status)
    monkeypatch.setattr(candidate_publisher, "train_model_family", fake_train_model_family)

    result = search_service.run_model_candidate_search(
        search_service.ModelCandidateSearchConfig("knn", "BTCUSDT", "10m", "fast", parallel_workers=1)
    )

    assert result["status"] == "shadow_active"
    assert training_calls[-1][0] == configs[1]


def test_candidate_search_updates_shadow_candidate_when_it_beats_active(monkeypatch) -> None:
    configs = [
        ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 5}),
        ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 9}),
    ]
    reports = [_report("shadow_active", 0.59, 1), _report("shadow_active", 0.63, 2)]
    training_calls = []

    def fake_status(*_args):
        return {
            "activeModelStatus": "shadow_active",
            "validationWinRate": 0.60,
            "validationGate": {"validation": {"winRate": 0.60, "profitFactor": 1.2, "sampleCount": 80}},
        }

    def fake_train_model_family(config, **kwargs):
        training_calls.append((config, kwargs))
        return {"status": "shadow_active", "relativePromotion": {"promoted": True}}

    monkeypatch.setattr(search_service, "attempted_model_search_keys", lambda *_args: frozenset())
    monkeypatch.setattr(search_service, "next_model_candidate_configs", lambda *_args: configs)
    monkeypatch.setattr(search_service, "start_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "complete_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "finish_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "record_model_candidate", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "train_candidate_reports", _candidate_report_iterator(configs, reports))
    monkeypatch.setattr(search_service, "run_walk_forward_stage", _walk_forward_passthrough)
    monkeypatch.setattr(candidate_publisher, "model_family_status", fake_status)
    monkeypatch.setattr(candidate_publisher, "train_model_family", fake_train_model_family)

    result = search_service.run_model_candidate_search(
        search_service.ModelCandidateSearchConfig("knn", "BTCUSDT", "10m", "fast", parallel_workers=1)
    )

    assert result["status"] == "shadow_active"
    assert training_calls[-1][0] == configs[1]
    assert training_calls[-1][1]["active_status_loader"] is candidate_publisher.model_family_status


def test_candidate_search_keeps_current_shadow_when_candidate_is_not_better(monkeypatch) -> None:
    configs = [ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 5})]
    reports = [_report("shadow_active", 0.59, 1)]
    training_calls = []

    monkeypatch.setattr(search_service, "attempted_model_search_keys", lambda *_args: frozenset())
    monkeypatch.setattr(search_service, "next_model_candidate_configs", lambda *_args: configs)
    monkeypatch.setattr(search_service, "start_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "complete_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "finish_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "record_model_candidate", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "train_candidate_reports", _candidate_report_iterator(configs, reports))
    monkeypatch.setattr(search_service, "run_walk_forward_stage", _walk_forward_passthrough)
    monkeypatch.setattr(
        candidate_publisher,
        "model_family_status",
        lambda *_args: {"activeModelStatus": "shadow_active", "validationWinRate": 0.60},
    )
    monkeypatch.setattr(candidate_publisher, "train_model_family", lambda *args, **kwargs: training_calls.append((args, kwargs)))

    result = search_service.run_model_candidate_search(
        search_service.ModelCandidateSearchConfig("knn", "BTCUSDT", "10m", "fast", parallel_workers=1)
    )

    assert result["status"] == "shadow_active"
    assert training_calls == []


def test_candidate_search_publishes_relative_shadow_from_full_stage(monkeypatch) -> None:
    configs = [
        ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 5}),
        ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 9}),
    ]
    reports = [_report("validation_failed", 0.58, 1), _report("validation_failed", 0.61, 2)]
    training_calls = []

    def fake_status(*_args):
        return {
            "activeModelStatus": "shadow_active",
            "validationGate": {"validation": {"winRate": 0.60, "profitFactor": 1.2, "sampleCount": 80}},
        }

    def fake_train_model_family(config, **kwargs):
        training_calls.append((config, kwargs))
        return {"status": "shadow_active", "relativePromotion": {"promoted": True}}

    def fail_walk_forward(finalists, _dataset_builder):
        return [], {"stage": "walk_forward", "evaluated": len(finalists), "advanced": 0, "candidates": []}

    monkeypatch.setattr(search_service, "attempted_model_search_keys", lambda *_args: frozenset())
    monkeypatch.setattr(search_service, "next_model_candidate_configs", lambda *_args: configs)
    monkeypatch.setattr(search_service, "start_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "complete_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "finish_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "record_model_candidate", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "train_candidate_reports", _candidate_report_iterator(configs, reports))
    monkeypatch.setattr(search_service, "run_walk_forward_stage", fail_walk_forward)
    monkeypatch.setattr(candidate_publisher, "model_family_status", fake_status)
    monkeypatch.setattr(candidate_publisher, "train_model_family", fake_train_model_family)

    result = search_service.run_model_candidate_search(
        search_service.ModelCandidateSearchConfig("knn", "BTCUSDT", "10m", "fast", parallel_workers=1)
    )

    assert result["status"] == "shadow_active"
    assert training_calls[-1][0] == configs[1]
    assert training_calls[-1][1]["active_status_loader"] is candidate_publisher.model_family_status


def test_candidate_search_reset_history_ignores_attempted_keys(monkeypatch) -> None:
    config = search_service.ModelCandidateSearchConfig("knn", "BTCUSDT", "10m", "fast", parallel_workers=1, reset_history=True)
    base = ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 5})
    requested = []

    monkeypatch.setattr(search_service, "model_training_config_for_profile", lambda *_args, **_kwargs: base)
    monkeypatch.setattr(search_service, "attempted_model_search_keys", lambda *_args: frozenset({"already_tried"}))
    monkeypatch.setattr(search_service, "next_model_candidate_configs", lambda _base, _profile, attempted: requested.append(attempted) or [base])
    monkeypatch.setattr(search_service, "start_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "complete_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "finish_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "record_model_candidate", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        search_service,
        "train_candidate_reports",
        _candidate_report_iterator([base], [_report("validation_failed", 0.52, 1)]),
    )
    monkeypatch.setattr(search_service, "run_walk_forward_stage", _walk_forward_passthrough)
    monkeypatch.setattr(candidate_publisher, "model_family_status", lambda *_args: {"activeModelStatus": "trade_active"})

    result = search_service.run_model_candidate_search(config)

    assert result["status"] == "validation_failed"
    assert requested[0] == frozenset()


def test_model_candidate_search_finishes_progress(monkeypatch) -> None:
    calls = []
    config = search_service.ModelCandidateSearchConfig("knn", "BTCUSDT", "10m", "fast", parallel_workers=1)
    base = ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 5})

    monkeypatch.setattr(search_service, "model_training_config_for_profile", lambda *_args, **_kwargs: base)
    monkeypatch.setattr(search_service, "attempted_model_search_keys", lambda *_args: frozenset())
    monkeypatch.setattr(search_service, "next_model_candidate_configs", lambda *_args: [base])
    monkeypatch.setattr(search_service, "start_model_candidate_progress", lambda *args, **kwargs: calls.append(("progress_start", kwargs["total"])) or {})
    monkeypatch.setattr(
        search_service,
        "complete_model_candidate_progress",
        lambda *args, **kwargs: calls.append(("progress_complete", kwargs["completed"])) or {},
    )
    monkeypatch.setattr(search_service, "finish_model_candidate_progress", lambda *args, **kwargs: calls.append(("progress_finish", kwargs["status"])) or {})
    monkeypatch.setattr(search_service, "record_model_candidate", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        search_service,
        "train_candidate_reports",
        _candidate_report_iterator([base], [_report("validation_failed", 0.52, 1)]),
    )
    monkeypatch.setattr(search_service, "run_walk_forward_stage", _walk_forward_passthrough)
    monkeypatch.setattr(candidate_publisher, "model_family_status", lambda *_args: {"activeModelStatus": "trade_active"})

    result = search_service.run_model_candidate_search(config)

    assert result["status"] == "validation_failed"
    assert calls[-1] == ("progress_finish", "validation_failed")


def test_model_candidate_search_records_failure_details(monkeypatch) -> None:
    calls = []
    config = search_service.ModelCandidateSearchConfig("knn", "BTCUSDT", "10m", "fast", parallel_workers=1)
    base = ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 5})

    monkeypatch.setattr(search_service, "model_training_config_for_profile", lambda *_args, **_kwargs: base)
    monkeypatch.setattr(search_service, "attempted_model_search_keys", lambda *_args: frozenset())
    monkeypatch.setattr(search_service, "next_model_candidate_configs", lambda *_args: [base])
    monkeypatch.setattr(search_service, "start_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "complete_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "finish_model_candidate_progress", lambda *args, **kwargs: calls.append(kwargs) or {})
    monkeypatch.setattr(search_service, "record_model_candidate", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        search_service,
        "train_candidate_reports",
        _candidate_report_iterator([base], [_report("shadow_active", 0.62, 1)]),
    )
    monkeypatch.setattr(search_service, "run_walk_forward_stage", _raise_walk_forward_failure)

    with pytest.raises(RuntimeError, match="walk-forward exploded"):
        search_service.run_model_candidate_search(config)

    assert calls[-1]["status"] == "failed"
    assert calls[-1]["failure"] == {
        "stage": "candidate_search",
        "error": "walk-forward exploded",
        "exceptionType": "RuntimeError",
    }


def test_exhausted_candidate_search_uses_library_status(monkeypatch) -> None:
    captured = {}
    library = {"records": [_report("validation_failed", 0.52, 1), _report("validation_failed", 0.54, 2)]}
    base = ModelFamilyTrainingConfig(family="random_forest", symbol="BTCUSDT", duration="60m")

    monkeypatch.setattr(search_service, "model_training_config_for_profile", lambda *_args, **_kwargs: base)
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
    monkeypatch.setenv(candidate_executor.XGBOOST_PROCESS_WORKERS_ENV, "3")

    assert candidate_executor._process_worker_count(10) == 3


def test_xgboost_process_worker_override_rejects_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv(candidate_executor.XGBOOST_PROCESS_WORKERS_ENV, "0")

    with pytest.raises(ValueError):
        candidate_executor._process_worker_count(10)


def test_torch_worker_override_limits_thread_workers(monkeypatch) -> None:
    config = ModelFamilyTrainingConfig(family="lstm", symbol="BTCUSDT", duration="10m")
    monkeypatch.setenv(candidate_executor.TORCH_JOBS_ENV, "2")

    assert candidate_executor._thread_worker_count([config], 10) == 2


def test_torch_worker_override_rejects_invalid_value(monkeypatch) -> None:
    config = ModelFamilyTrainingConfig(family="lstm", symbol="BTCUSDT", duration="10m")
    monkeypatch.setenv(candidate_executor.TORCH_JOBS_ENV, "0")

    with pytest.raises(ValueError):
        candidate_executor._thread_worker_count([config], 10)


def test_xgboost_uses_hist_tree_method() -> None:
    model = _xgboost(JoblibModelOptions("xgboost", 20260513, {}))

    assert model.get_params()["tree_method"] == XGBOOST_TREE_METHOD


def test_lightgbm_model_family_uses_lightgbm_classifier() -> None:
    model = lightgbm_estimator({}, 20260513)

    assert model.__class__.__name__ == "LGBMClassifier"


def test_extra_trees_model_family_uses_extra_trees_classifier() -> None:
    model = extra_trees_estimator({}, 20260513)

    assert model.__class__.__name__ == "ExtraTreesClassifier"


def test_logistic_elasticnet_model_family_uses_elasticnet_penalty() -> None:
    model = logistic_elasticnet_estimator({}, 20260513)

    assert model.__class__.__name__ == "SGDClassifier"
    assert model.get_params()["loss"] == "log_loss"
    assert model.get_params()["penalty"] == "elasticnet"


def test_catboost_model_family_uses_catboost_classifier() -> None:
    model = catboost_estimator({}, 20260513)

    assert model.__class__.__name__ == "CatBoostClassifier"


def test_candidate_training_withholds_test_set_during_search(monkeypatch) -> None:
    config = ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m")
    captured = {}

    def fake_train_model_family(_config, **kwargs):
        captured.update(kwargs)
        return {
            "status": "validation_failed",
            "validation": {"winRate": 0.55, "profitFactor": 1.2, "sampleCount": 60},
            "test": {"status": "withheld"},
        }

    monkeypatch.setattr(candidate_executor, "train_model_family", fake_train_model_family)
    report = candidate_executor.train_candidate(config, "fast", lambda _cfg: _fake_dataset(_cfg), stage="coarse", record_config=config)

    assert captured["evaluate_test"] is False
    assert report["test"]["status"] == "withheld"
    assert report["searchStage"] == "coarse"


def test_candidate_training_failure_records_exception_details(monkeypatch) -> None:
    config = ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m")
    captured = {}

    def fail_train(*_args, **_kwargs):
        raise RuntimeError("candidate exploded")

    monkeypatch.setattr(candidate_executor, "train_model_family", fail_train)
    monkeypatch.setattr(
        candidate_executor,
        "record_model_candidate",
        lambda _cfg, _profile, report: captured.update(report=report),
    )

    report = candidate_executor.train_candidate(
        config,
        "fast",
        lambda _cfg: _fake_dataset(_cfg),
        stage="coarse",
        record_config=config,
    )

    assert report["status"] == "failed"
    assert report["failure"] == {
        "stage": "candidate_training",
        "exceptionType": "RuntimeError",
        "error": "candidate exploded",
    }
    assert captured["report"]["failure"]["exceptionType"] == "RuntimeError"


def test_model_family_report_includes_training_input_observability(tmp_path) -> None:
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
        dataset_builder=_observable_dataset,
        publish_initial_baseline=True,
    )

    assert report["featureColumns"] == ["a", "factor_combo_top1_score", "sim_feedback_win_rate"]
    assert report["explicitFactorComboFeatures"]["included"] is True
    assert report["explicitFactorComboFeatures"]["source"] == "historical_replay"
    assert report["simFeedback"]["settledCount"] == 0
    assert report["dataQuality"]["status"] == "passed"
    assert report["tradingCosts"]["roundtripCostRate"] >= 0.0


def test_candidate_score_uses_validation_not_test() -> None:
    strong_validation_bad_test = _report("trade_active", 0.80, 1)
    weak_validation_good_test = _report("trade_active", 0.60, 2)
    strong_validation_bad_test["test"] = {"winRate": 0.10, "profitFactor": 0.1}
    weak_validation_good_test["test"] = {"winRate": 0.99, "profitFactor": 9.0}

    selected = max(
        [strong_validation_bad_test, weak_validation_good_test],
        key=candidate_publisher._candidate_score,
    )

    assert selected is strong_validation_bad_test


def test_model_family_walk_forward_outputs_fold_boundaries(monkeypatch) -> None:
    config = ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", feature_window=8)
    finalist = candidate_executor.CandidateTrainingResult(config, _report("trade_active", 0.80, 1))

    def fake_train_model_family(_config, **kwargs):
        dataset = kwargs["dataset_builder"](_config)
        return {
            "validationGate": {"status": "passed"},
            "validation": {"sampleCount": len(dataset.y) // 5, "winRate": 0.8},
            "test": {"sampleCount": len(dataset.y) // 5, "winRate": 0.78},
            "sampleCounts": {"train": 120, "validation": 60, "test": 60},
            "validationFailureReason": None,
        }

    monkeypatch.setattr(model_family_walk_forward, "train_model_family", fake_train_model_family)

    survivors, payload = model_family_walk_forward.run_walk_forward_stage([finalist], _fake_dataset)

    assert len(survivors) == 1
    assert payload["stage"] == "walk_forward"
    assert payload["evaluated"] == 1
    assert payload["advanced"] == 1
    assert payload["candidates"][0]["foldCount"] == 3
    assert payload["candidates"][0]["folds"][0]["trainEnd"] > payload["candidates"][0]["folds"][0]["start"]


def _report(status: str, win_rate: float, version: int) -> dict:
    return {
        "status": status,
        "modelVersion": f"v{version}",
        "validation": {"winRate": win_rate, "profitFactor": 2.0, "sampleCount": 60},
        "test": {"winRate": win_rate, "profitFactor": 2.0},
        "sampleCounts": {"validation": 60, "test": 60},
    }


def _candidate_report_iterator(configs: list[ModelFamilyTrainingConfig], reports: list[dict]):
    def _iter(train_configs, profile, workers, dataset_builder, *, stage, record_configs=None):
        del profile, workers, dataset_builder
        records = record_configs or train_configs
        for record_config in records:
            report = {**reports[configs.index(record_config)], "searchStage": stage}
            yield candidate_executor.CandidateTrainingResult(record_config, report)

    return _iter


def _walk_forward_passthrough(finalists, _dataset_builder):
    return finalists, {
        "stage": "walk_forward",
        "evaluated": len(finalists),
        "advanced": len(finalists),
        "candidateKeys": [item.report.get("searchKey") for item in finalists],
        "advancedKeys": [item.report.get("searchKey") for item in finalists],
        "candidates": [],
    }


def _raise_walk_forward_failure(*_args):
    raise RuntimeError("walk-forward exploded")


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


def _observable_dataset(config: ModelFamilyTrainingConfig) -> LstmDataset:
    base = _fake_dataset(config)
    return LstmDataset(
        x=base.x,
        y=base.y,
        future_returns=base.future_returns,
        entry_open_times=base.entry_open_times,
        feature_columns=["a", "factor_combo_top1_score", "sim_feedback_win_rate"],
        feature_frame=pd.DataFrame({"a": np.zeros(len(base.y))}),
        combo_snapshot=base.combo_snapshot,
        learning_context={},
        data_quality_report={"status": "passed", "features": {"featureColumnCount": 3}},
        sim_feedback_metadata={"enabled": True, "settledCount": 0, "neutralFeaturesUsed": True},
        factor_combo_metadata={
            "enabled": True,
            "source": "historical_replay",
            "snapshotCount": 400,
            "missingRate": 0.0,
        },
    )
