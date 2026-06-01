from __future__ import annotations

from typing import Any

PAPER_LIVE_PREFILTER_POLICY = "validation_gate_allows_paper_live_only"
PAPER_LIVE_RELATIVE_POLICY = "validation_gate_or_relative_shadow_observation"
PAPER_LIVE_CREDIBILITY_SOURCE = "settled_paper_live_predictions"


def paper_live_admission_payload(status: str | None, gate: dict[str, Any]) -> dict[str, Any]:
    passed = gate.get("status") == "passed"
    observing = str(status or "") in {"shadow_active", "initial_baseline", "trade_active", "trained"}
    return {
        "policy": PAPER_LIVE_RELATIVE_POLICY,
        "allowed": observing,
        "status": "paper_collecting" if observing else "backtest_candidate",
        "reason": _admission_reason(status, gate, passed),
        "validationWinRate": _validation_win_rate(gate),
        "minConfidence": gate.get("minConfidence"),
        "paperLiveWinRate": None,
        "paperLiveSampleCount": 0,
        "credibilitySource": PAPER_LIVE_CREDIBILITY_SOURCE,
        "realTradingEnabled": False,
    }


def model_status_policy_payload(active_status: str | None, gate: dict[str, Any]) -> dict[str, Any]:
    admission = paper_live_admission_payload(active_status, gate)
    return {
        "paperLiveAdmission": admission,
        "paperLiveStatus": admission["status"],
        "paperLiveWinRate": None,
        "paperLiveSampleCount": 0,
        "modelCredibilitySource": PAPER_LIVE_CREDIBILITY_SOURCE,
        "validationRole": PAPER_LIVE_RELATIVE_POLICY,
        "realTradingEnabled": False,
    }


def _admission_reason(status: str | None, gate: dict[str, Any], passed: bool) -> str:
    if passed:
        return "validation_gate_passed"
    if str(status or "") in {"shadow_active", "initial_baseline"}:
        return "shadow_observation_allowed_without_trade_gate"
    return _gate_failure_reason(status, gate)


def _gate_failure_reason(status: str | None, gate: dict[str, Any]) -> str:
    if gate.get("reason"):
        return str(gate["reason"])
    return f"model_status_{status or 'unknown'}"


def _validation_win_rate(gate: dict[str, Any]) -> float | None:
    validation = gate.get("validation")
    if not isinstance(validation, dict):
        return None
    value = validation.get("winRate")
    return None if value is None else float(value)
