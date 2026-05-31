from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.mining import router as mining_router
from app.main import app
from app.services import factor_learning_memory_store


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(factor_learning_memory_store, "FACTOR_LEARNING_DIR", tmp_path)
    if "/api/mining/overview" not in {getattr(route, "path", "") for route in app.routes}:
        app.include_router(mining_router)
    return TestClient(app)


def test_mining_overview_returns_404_without_memory(client: TestClient) -> None:
    response = client.get("/api/mining/overview", params={"symbol": "BTCUSDT", "duration": "10m"})
    assert response.status_code == 404


def test_mining_overview_shape(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
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
    response = client.get("/api/mining/overview", params={"symbol": "BTCUSDT", "duration": "10m", "fresh": "true"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["overallAccuracy"] == 0.64
    assert payload["trainingRules"]["targetWinRateExclusive"] == 0.62
    assert "> 62%" in payload["trainingRules"]["text"]
    assert len(payload["models"]) == 14
    assert payload["agentCandidates"][0]["factorName"]
