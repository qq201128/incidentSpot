from __future__ import annotations

from app.services.model_family_paper_live_policy import (
    model_status_policy_payload,
    paper_live_admission_payload,
)


def test_validation_gate_pass_only_allows_paper_live_collection() -> None:
    gate = {
        "status": "passed",
        "minConfidence": 0.65,
        "validation": {"winRate": 0.6401},
    }

    admission = paper_live_admission_payload("trade_active", gate)

    assert admission["allowed"] is True
    assert admission["status"] == "paper_collecting"
    assert admission["validationWinRate"] == 0.6401
    assert admission["paperLiveWinRate"] is None
    assert admission["paperLiveSampleCount"] == 0
    assert admission["realTradingEnabled"] is False
    assert admission["credibilitySource"] == "settled_paper_live_predictions"


def test_validation_gate_failure_is_not_paper_live_stable() -> None:
    gate = {"status": "failed", "reason": "no_validation_confidence_threshold_met"}

    payload = model_status_policy_payload("validation_failed", gate)

    assert payload["paperLiveAdmission"]["allowed"] is False
    assert payload["paperLiveStatus"] == "backtest_candidate"
    assert payload["validationRole"] == "validation_gate_or_relative_shadow_observation"
    assert payload["realTradingEnabled"] is False


def test_shadow_active_collects_paper_live_without_trade_gate() -> None:
    gate = {"status": "failed", "reason": "no_validation_confidence_threshold_met"}

    payload = model_status_policy_payload("shadow_active", gate)

    assert payload["paperLiveAdmission"]["allowed"] is True
    assert payload["paperLiveStatus"] == "paper_collecting"
    assert payload["paperLiveAdmission"]["reason"] == "shadow_observation_allowed_without_trade_gate"
    assert payload["realTradingEnabled"] is False
