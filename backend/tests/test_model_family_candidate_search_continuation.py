from __future__ import annotations

import pytest

from app.services import model_family_candidate_search_service as search_service
from app.services.model_family_config import ModelFamilyTrainingConfig


def test_batched_candidate_search_counts_failed_records_as_attempted(monkeypatch: pytest.MonkeyPatch) -> None:
    base = ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m")
    candidate = ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 7})
    requested = []
    config = search_service.ModelCandidateSearchConfig(
        "knn",
        "BTCUSDT",
        "10m",
        "fast",
        parallel_workers=1,
        candidates_per_job=1,
    )

    monkeypatch.setattr(search_service, "model_training_config_for_profile", lambda *_args, **_kwargs: base)
    monkeypatch.setattr(search_service, "recorded_model_search_keys", lambda *_args: frozenset({"failed_key"}))
    monkeypatch.setattr(search_service, "attempted_model_search_keys", _unexpected_attempted_keys)
    monkeypatch.setattr(
        search_service,
        "next_model_candidate_configs",
        lambda _base, _profile, attempted: requested.append(attempted) or [candidate],
    )
    monkeypatch.setattr(search_service, "start_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "complete_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "finish_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "record_model_candidate", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "train_candidate_reports", _candidate_reports(candidate))
    monkeypatch.setattr(search_service, "run_walk_forward_stage", lambda *_args: ([], {"candidates": []}))
    monkeypatch.setattr(search_service, "publish_best_model_candidate", lambda *_args, **_kwargs: None)

    result = search_service.run_model_candidate_search(config)

    assert requested[0] == frozenset({"failed_key"})
    assert result["jobBatch"]["hasMoreCandidates"] is False


def test_candidate_budget_stops_requeue_after_budget_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    base = ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m")
    candidates = [
        ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": value})
        for value in (3, 5, 7, 9, 11)
    ]
    config = search_service.ModelCandidateSearchConfig(
        "knn",
        "BTCUSDT",
        "10m",
        "fast",
        parallel_workers=1,
        candidates_per_job=2,
        candidate_budget=2,
    )

    monkeypatch.setattr(search_service, "model_training_config_for_profile", lambda *_args, **_kwargs: base)
    monkeypatch.setattr(search_service, "recorded_model_search_keys", lambda *_args: frozenset())
    monkeypatch.setattr(search_service, "next_model_candidate_configs", lambda *_args: candidates)
    monkeypatch.setattr(search_service, "start_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "complete_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "finish_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "record_model_candidate", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(search_service, "train_candidate_reports", _candidate_reports(*candidates[:2]))
    monkeypatch.setattr(search_service, "run_walk_forward_stage", lambda *_args: ([], {"candidates": []}))
    monkeypatch.setattr(search_service, "publish_best_model_candidate", lambda *_args, **_kwargs: None)

    result = search_service.run_model_candidate_search(config)

    assert result["jobBatch"]["selectedCandidates"] == 2
    assert result["jobBatch"]["hasMoreCandidates"] is False
    assert result["jobBatch"]["budgetExhausted"] is True
    assert result["jobBatch"]["unsearchedCandidatesAfterBudget"] == 3


def test_candidate_budget_exhausted_returns_explicit_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    base = ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m")
    candidate = ModelFamilyTrainingConfig(family="knn", symbol="BTCUSDT", duration="10m", params={"n_neighbors": 3})
    config = search_service.ModelCandidateSearchConfig(
        "knn",
        "BTCUSDT",
        "10m",
        "fast",
        parallel_workers=1,
        candidates_per_job=2,
        candidate_budget=1,
    )

    monkeypatch.setattr(search_service, "model_training_config_for_profile", lambda *_args, **_kwargs: base)
    monkeypatch.setattr(search_service, "recorded_model_search_keys", lambda *_args: frozenset({"already_done"}))
    monkeypatch.setattr(search_service, "next_model_candidate_configs", lambda *_args: [candidate])
    monkeypatch.setattr(search_service, "read_model_candidate_library", lambda *_args, **_kwargs: {"records": [{"status": "validation_failed"}]})
    monkeypatch.setattr(search_service, "finish_model_candidate_progress_from_library", lambda *_args, **_kwargs: {})

    result = search_service.run_model_candidate_search(config)

    assert result["reason"] == "candidate_budget_exhausted"
    assert result["jobBatch"]["hasMoreCandidates"] is False
    assert result["jobBatch"]["budgetExhausted"] is True


def _candidate_reports(*candidates: ModelFamilyTrainingConfig):
    def _iterator(*_args, **_kwargs):
        for candidate in candidates:
            report = {"status": "validation_failed", "validation": {}, "test": {}}
            yield search_service.CandidateTrainingResult(candidate, report)

    return _iterator


def _unexpected_attempted_keys(*_args):
    pytest.fail("batched candidate search must use recorded keys, including failed records")
