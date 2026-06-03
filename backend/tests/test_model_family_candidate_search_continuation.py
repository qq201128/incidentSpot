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


def _candidate_reports(candidate: ModelFamilyTrainingConfig):
    def _iterator(*_args, **_kwargs):
        report = {"status": "validation_failed", "validation": {}, "test": {}}
        yield search_service.CandidateTrainingResult(candidate, report)

    return _iterator


def _unexpected_attempted_keys(*_args):
    pytest.fail("batched candidate search must use recorded keys, including failed records")
