from __future__ import annotations

from app.services.model_family_candidates import model_candidate_library_summary, record_model_candidate
from app.services.model_family_config import ModelFamilyTrainingConfig


def test_candidate_library_summarizes_best_validation_and_test_candidates(tmp_path) -> None:
    strong_validation = ModelFamilyTrainingConfig(
        family="knn",
        symbol="BTCUSDT",
        duration="10m",
        feature_window=24,
        params={"n_neighbors": 5},
    )
    strong_test = ModelFamilyTrainingConfig(
        family="knn",
        symbol="BTCUSDT",
        duration="10m",
        feature_window=32,
        params={"n_neighbors": 9},
    )

    record_model_candidate(
        strong_validation,
        "fast",
        _report("shadow_active", "knn_validation", validation=0.72, test=0.58),
        artifact_root=tmp_path,
    )
    record_model_candidate(
        strong_test,
        "fast",
        _report("trade_active", "knn_test", validation=0.63, test=0.77),
        artifact_root=tmp_path,
    )

    summary = model_candidate_library_summary("knn", "BTCUSDT", "10m", artifact_root=tmp_path)

    assert summary["total"] == 2
    assert summary["bestValidationCandidate"]["modelVersion"] == "knn_validation"
    assert summary["bestTestCandidate"]["modelVersion"] == "knn_test"
    assert summary["bestShadowCandidate"]["modelVersion"] == "knn_validation"
    assert summary["bestTradeCandidate"]["modelVersion"] == "knn_test"


def _report(status: str, model_version: str, *, validation: float, test: float) -> dict:
    return {
        "status": status,
        "modelVersion": model_version,
        "validation": _metrics(validation),
        "test": _metrics(test),
    }


def _metrics(win_rate: float) -> dict:
    return {"winRate": win_rate, "profitFactor": 1.2, "sampleCount": 60}
