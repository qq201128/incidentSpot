from __future__ import annotations

import pytest

from app.services.model_family_daily_candidates import model_family_daily_candidate_report


def test_daily_model_candidates_merge_settled_paper_live_lifecycle() -> None:
    report = model_family_daily_candidate_report(
        "btcusdt",
        "10m",
        families=("xgboost", "knn"),
        status_loader=_status,
        lifecycle_loader=_lifecycle,
    )

    xgboost = report["models"][0]
    knn = report["models"][1]
    assert report["familyCount"] == 2
    assert report["paperLiveReadyCount"] == 2
    assert xgboost["modelFamily"] == "xgboost"
    assert xgboost["paperLiveStatus"] == "paper_failed"
    assert xgboost["paperLiveWinRate"] == pytest.approx(0.5)
    assert xgboost["paperLiveSampleCount"] == 30
    assert xgboost["paperLiveReason"] == "paper_live_win_rate_below_target"
    assert xgboost["candidateLibrary"]["bestValidationCandidate"]["modelVersion"] == "xgboost_best_val"
    assert xgboost["candidateLibrary"]["bestTestCandidate"]["modelVersion"] == "xgboost_best_test"
    assert knn["paperLiveStatus"] == "paper_collecting"
    assert knn["paperLiveSampleCount"] == 0


def test_daily_model_candidates_surface_status_loader_failure() -> None:
    report = model_family_daily_candidate_report(
        "BTCUSDT",
        "10m",
        families=("bayesian",),
        status_loader=lambda *_args: (_ for _ in ()).throw(RuntimeError("status failed")),
        lifecycle_loader=lambda *_args: {"allCandidates": []},
    )

    failure = report["failures"][0]
    assert report["status"] if "status" in report else True
    assert failure["modelFamily"] == "bayesian"
    assert failure["status"] == "failed"
    assert failure["reason"] == "status failed"


def _status(family: str, _symbol: str, _duration: str) -> dict:
    return {
        "modelVersion": f"{family}_v1",
        "featureWindow": 32,
        "selectedConfidenceThreshold": 0.65,
        "validationWinRate": 0.64,
        "activeModelStatus": "trade_active",
        "candidateLibrary": {
            "bestValidationCandidate": {"modelVersion": f"{family}_best_val"},
            "bestTestCandidate": {"modelVersion": f"{family}_best_test"},
        },
        "paperLiveAdmission": {
            "allowed": True,
            "status": "paper_collecting",
            "reason": "validation_gate_passed",
            "minConfidence": 0.65,
            "validationWinRate": 0.64,
        },
    }


def _lifecycle(_symbol: str, _duration: str) -> dict:
    return {
        "allCandidates": [
            {
                "candidateType": "model",
                "modelFamily": "xgboost",
                "paperLiveStatus": "paper_failed",
                "paperLiveWinRate": 0.5,
                "paperLiveSampleCount": 30,
                "reason": "paper_live_win_rate_below_target",
            }
        ]
    }
