from __future__ import annotations

from typing import Any

from app.services.lstm_lifecycle import (
    LSTM_STATUS_SHADOW_ACTIVE,
    LSTM_STATUS_VALIDATION_FAILED,
    shadow_predictable_status,
)

RELATIVE_PROMOTION_POLICY = "relative_validation_win_rate_improvement_enables_shadow_observation"
RELATIVE_OBSERVATION_STATUSES = frozenset({LSTM_STATUS_SHADOW_ACTIVE, LSTM_STATUS_VALIDATION_FAILED})
MIN_VALIDATION_SAMPLE_COUNT = 30
MIN_WIN_RATE_EDGE = 0.0


def relative_shadow_report(report: dict[str, Any], active_status: dict[str, Any]) -> dict[str, Any]:
    decision = relative_shadow_decision(report, active_status)
    if not decision["promoted"]:
        return report
    return {
        **report,
        "status": "shadow_active",
        "candidateStatus": "promoted_shadow_active",
        "promotionReason": decision["reason"],
        "relativePromotion": decision,
    }


def relative_shadow_decision(report: dict[str, Any], active_status: dict[str, Any]) -> dict[str, Any]:
    candidate = _metrics(report)
    active = _active_metrics(active_status)
    if not _eligible_report(report, candidate):
        return _decision(False, "candidate_not_relative_observation_eligible", candidate, active)
    if not shadow_predictable_status(active_status.get("activeModelStatus") or active_status.get("status")):
        return _decision(False, "no_active_shadow_model", candidate, active)
    if active["winRate"] is None:
        return _decision(False, "active_model_win_rate_missing", candidate, active)
    if candidate["winRate"] <= active["winRate"] + MIN_WIN_RATE_EDGE:
        return _decision(False, "candidate_win_rate_not_improved", candidate, active)
    return _decision(True, "candidate_win_rate_beats_active_model", candidate, active)


def _eligible_report(report: dict[str, Any], metrics: dict[str, Any]) -> bool:
    return (
        str(report.get("status") or "") in RELATIVE_OBSERVATION_STATUSES
        and metrics["sampleCount"] >= MIN_VALIDATION_SAMPLE_COUNT
        and metrics["winRate"] is not None
    )


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("validation") or {}
    sample_counts = report.get("sampleCounts") or {}
    return {
        "winRate": _float_or_none(metrics.get("winRate")),
        "profitFactor": _float_or_none(metrics.get("profitFactor")),
        "sampleCount": int(sample_counts.get("validation") or metrics.get("sampleCount") or 0),
    }


def _active_metrics(status: dict[str, Any]) -> dict[str, Any]:
    gate = status.get("validationGate") or {}
    metrics = gate.get("validation") if isinstance(gate.get("validation"), dict) else {}
    win_rate = _first_float(metrics.get("winRate"), status.get("validationWinRate"))
    return {
        "winRate": win_rate,
        "profitFactor": _float_or_none(metrics.get("profitFactor")),
        "sampleCount": int(metrics.get("sampleCount") or (status.get("sampleCounts") or {}).get("validation") or 0),
    }


def _decision(promoted: bool, reason: str, candidate: dict[str, Any], active: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": RELATIVE_PROMOTION_POLICY,
        "promoted": promoted,
        "reason": reason,
        "candidate": candidate,
        "active": active,
        "realTradingEnabled": False,
    }


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is not None:
            return float(value)
    return None
