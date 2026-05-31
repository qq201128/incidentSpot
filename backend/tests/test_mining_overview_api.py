from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api import mining as mining_api
from app.services import factor_learning_memory_store


@pytest.fixture()
def isolated_memory_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(factor_learning_memory_store, "FACTOR_LEARNING_DIR", tmp_path)
    return tmp_path


def test_mining_overview_returns_404_without_memory(isolated_memory_dir: Path) -> None:
    with pytest.raises(HTTPException) as exc_info:
        mining_api.get_mining_overview(symbol="BTCUSDT", duration="10m", fresh=True)
    assert exc_info.value.status_code == 404


def test_mining_overview_shape(isolated_memory_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_memory(symbol: str, duration: str) -> dict:
        return {
            "symbol": symbol,
            "duration": duration,
            "updatedAt": "2026-05-21T06:32:08Z",
            "source": {"rankingRefreshSource": "cache", "minedFrameFailureCount": 0},
            "adaptiveLearning": {"overallAccuracy": 0.64, "sampleCount": 218},
            "lossMemory": {"status": "learned", "sampleCount": 218, "lossCount": 146, "patterns": []},
            "factorMining": {"successPatterns": [], "forbiddenRegions": []},
            "weights": {"trend": 0.28, "volatility": 0.21},
            "llmAgent": {"review": {"factorMiningPlan": {"candidateFactorIdeas": [{"nameHint": "f1"}]}}},
            "agentCandidatePromotion": {"candidateCount": 1, "promoted": 0, "records": []},
            "agentMinedFactorLibrary": {"total": 0},
            "minedFactorLibrary": {"total": 0},
            "monitoring": {"issues": []},
        }

    monkeypatch.setattr("app.services.mining_overview_service.get_factor_learning_memory", fake_memory)
    monkeypatch.setattr(
        "app.services.mining_overview_service.model_search_queue_status",
        lambda _filters: {
            "counts": {"pending": 2, "running": 0},
            "runningJobs": [],
            "latestLogPath": None,
        },
    )
    monkeypatch.setattr(
        "app.services.mining_overview_service.model_family_status",
        lambda family, symbol, duration, **_kwargs: {
            "modelFamily": family,
            "strategyKey": f"{family}_shadow",
            "status": "untrained",
            "shadowPredictionReady": False,
            "candidateSearchProgress": {"status": "idle"},
            "candidateLibrary": {"total": 0},
            "trainingRules": {"searchSpaceTotal": 10},
        },
    )
    payload = mining_api.get_mining_overview(symbol="BTCUSDT", duration="10m", fresh=True)
    assert payload["summary"]["overallAccuracy"] == 0.64
    assert payload["trainingRules"]["targetWinRateExclusive"] == 0.62
    assert payload["trainingRules"]["internalThreads"] == 4
    assert payload["trainingRules"]["parallelWorkers"] == 10
    assert payload["trainingRules"]["xgboostProcessWorkers"] == 1
    assert "> 62%" in payload["trainingRules"]["text"]
    assert payload["trainingRules"]["workerStatus"]["state"] == "worker_required"
    assert payload["summary"]["searchPendingCount"] == 2
    assert len(payload["models"]) == 14
    assert payload["agentCandidates"][0]["factorName"]


def test_mining_overview_model_card_treats_combo_mismatch_as_ready() -> None:
    from app.services.mining_overview_service import _model_card

    card = _model_card(
        {
            "modelFamily": "knn",
            "strategyKey": "factor_knn_shadow_10m",
            "status": "shadow_active",
            "shadowPredictionReady": False,
            "shadowPredictionBlockedReason": "combo_snapshot_mismatch",
            "candidateSearchProgress": {"status": "idle"},
            "candidateLibrary": {"total": 0},
            "trainingRules": {"searchSpaceTotal": 24},
        }
    )

    assert card["cardState"] == "ready"
    assert card["predictionReadyLabel"] == "就绪"
