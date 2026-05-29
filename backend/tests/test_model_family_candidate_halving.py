from __future__ import annotations

from app.services import model_family_candidate_executor as executor
from app.services import model_family_candidate_search_service as service
from app.services.model_family_candidate_search_service import ModelCandidateSearchConfig
from app.services.model_family_config import ModelFamilyTrainingConfig


def test_candidate_search_runs_successive_halving(monkeypatch) -> None:
    base = ModelFamilyTrainingConfig(family="lstm", symbol="BTCUSDT", duration="10m", epochs=12)
    candidates = [
        ModelFamilyTrainingConfig(**{**base.__dict__, "params": {"rank": rank, "score": score}})
        for rank, score in enumerate([0.91, 0.82, 0.44, 0.31], start=1)
    ]
    calls = []
    recorded = []

    monkeypatch.setattr(service, "model_training_config_for_profile", lambda *_args: base)
    monkeypatch.setattr(service, "attempted_model_search_keys", lambda *_args: frozenset())
    monkeypatch.setattr(service, "next_model_candidate_configs", lambda *_args: candidates)
    monkeypatch.setattr(service, "start_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(service, "complete_model_candidate_progress", lambda *_args: {})
    monkeypatch.setattr(service, "finish_model_candidate_progress", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(service, "record_model_candidate", lambda config, profile, report: recorded.append(report))
    monkeypatch.setattr(executor, "record_model_candidate", lambda config, profile, report: recorded.append(report))
    monkeypatch.setattr(executor, "train_model_family", _training_stub(calls))
    monkeypatch.setattr(service, "run_walk_forward_stage", _walk_forward_stub)
    monkeypatch.setattr(service, "publish_best_model_candidate", _publisher_stub(calls))

    result = service.run_model_candidate_search(
        ModelCandidateSearchConfig(
            family="lstm",
            symbol="BTCUSDT",
            duration="10m",
            profile="fast",
            parallel_workers=1,
        )
    )

    assert result["trainingRules"]["searchMode"] == "successive_halving"
    assert [stage["stage"] for stage in result["successiveHalvingStages"]] == ["coarse", "full", "walk_forward"]
    assert result["successiveHalvingStages"][0]["evaluated"] == 4
    assert result["successiveHalvingStages"][0]["advanced"] == 2
    assert result["successiveHalvingStages"][1]["evaluated"] == 2
    assert [item for item in calls if item[0] == "train"] == [
        ("train", 1, 6),
        ("train", 2, 6),
        ("train", 3, 6),
        ("train", 4, 6),
        ("train", 1, 12),
        ("train", 2, 12),
    ]
    assert calls[-1] == ("publish", [1])
    assert any(item["searchStage"] == "coarse" and item.get("advancedToNextStage") is False for item in recorded)
    assert any(item.get("eliminationReason") == "coarse_rank_below_survival_cutoff" for item in recorded)


def _training_stub(calls):
    def _train(config: ModelFamilyTrainingConfig, **_kwargs):
        calls.append(("train", config.params["rank"], config.epochs))
        score = config.params["score"]
        return {
            "status": "trade_active",
            "candidateStatus": "trade_active",
            "modelFamily": config.family,
            "symbol": config.symbol,
            "duration": config.duration,
            "modelVersion": f"candidate-{config.params['rank']}",
            "validation": {"winRate": score, "profitFactor": score + 1.0, "sampleCount": 20},
            "test": {"winRate": score, "profitFactor": score + 1.0, "sampleCount": 20},
            "sampleCounts": {"test": 20},
        }

    return _train


def _publisher_stub(calls):
    def _publish(trainings):
        calls.append(("publish", [item.config.params["rank"] for item in trainings]))
        return {"status": "trade_active"}

    return _publish


def _walk_forward_stub(finalists, _dataset_builder):
    payload = {
        "stage": "walk_forward",
        "evaluated": len(finalists),
        "advanced": len(finalists),
        "candidateKeys": [item.report.get("searchKey") for item in finalists],
        "advancedKeys": [item.report.get("searchKey") for item in finalists],
        "candidates": [],
    }
    return finalists, payload
