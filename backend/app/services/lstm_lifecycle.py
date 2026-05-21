from __future__ import annotations

from typing import Any

LSTM_STATUS_TRAINING = "training"
LSTM_STATUS_TRADE_ACTIVE = "trade_active"
LSTM_STATUS_SHADOW_ACTIVE = "shadow_active"
LSTM_STATUS_LEGACY_TRAINED = "trained"
LSTM_STATUS_VALIDATION_FAILED = "validation_failed"
LSTM_STATUS_INSUFFICIENT_SAMPLES = "insufficient_samples"
LSTM_STATUS_FAILED = "failed"

CANDIDATE_TRAINING = "training"
CANDIDATE_PROMOTED_TRADE_ACTIVE = "promoted_trade_active"
CANDIDATE_PROMOTED_SHADOW_ACTIVE = "promoted_shadow_active"
CANDIDATE_REJECTED_VALIDATION = "rejected_validation"
CANDIDATE_REJECTED_INSUFFICIENT_SAMPLES = "rejected_insufficient_samples"
CANDIDATE_FAILED = "failed"

SHADOW_MIN_SAMPLE_COUNT = 50
SHADOW_MIN_WIN_RATE = 0.70
SHADOW_MIN_PROFIT_FACTOR = 1.0
SHADOW_MIN_AVG_RETURN = 0.0


def lifecycle_status(gate: dict[str, Any], validation: dict[str, Any], test: dict[str, Any]) -> str:
    if gate.get("status") == "passed":
        return LSTM_STATUS_TRADE_ACTIVE
    if shadow_quality_passed(validation, test):
        return LSTM_STATUS_SHADOW_ACTIVE
    return LSTM_STATUS_VALIDATION_FAILED


def shadow_quality_passed(validation: dict[str, Any], test: dict[str, Any]) -> bool:
    return any(
        _split_shadow_quality_passed(val_row) and _split_shadow_quality_passed(test_row)
        for val_row, test_row in _matching_quality_rows(validation, test)
    )


def candidate_status(status: str) -> str:
    if status == LSTM_STATUS_TRAINING:
        return CANDIDATE_TRAINING
    if status in {LSTM_STATUS_TRADE_ACTIVE, LSTM_STATUS_LEGACY_TRAINED}:
        return CANDIDATE_PROMOTED_TRADE_ACTIVE
    if status == LSTM_STATUS_SHADOW_ACTIVE:
        return CANDIDATE_PROMOTED_SHADOW_ACTIVE
    if status == LSTM_STATUS_VALIDATION_FAILED:
        return CANDIDATE_REJECTED_VALIDATION
    if status == LSTM_STATUS_INSUFFICIENT_SAMPLES:
        return CANDIDATE_REJECTED_INSUFFICIENT_SAMPLES
    return CANDIDATE_FAILED


def promotion_reason(status: str, gate: dict[str, Any]) -> str:
    if status == LSTM_STATUS_TRADE_ACTIVE:
        return "validation_gate_passed"
    if status == LSTM_STATUS_SHADOW_ACTIVE:
        return "shadow_quality_passed"
    return str(gate.get("reason") or "validation_gate_failed")


def publishes_active_artifacts(status: str) -> bool:
    return status in {LSTM_STATUS_SHADOW_ACTIVE, LSTM_STATUS_TRADE_ACTIVE, LSTM_STATUS_LEGACY_TRAINED}


def shadow_predictable_status(status: str | None) -> bool:
    return status in {LSTM_STATUS_SHADOW_ACTIVE, LSTM_STATUS_TRADE_ACTIVE, LSTM_STATUS_LEGACY_TRAINED}


def trade_active_status(status: str | None) -> bool:
    return status in {LSTM_STATUS_TRADE_ACTIVE, LSTM_STATUS_LEGACY_TRAINED}


def _split_shadow_quality_passed(metrics: dict[str, Any]) -> bool:
    return (
        int(metrics.get("sampleCount") or 0) >= SHADOW_MIN_SAMPLE_COUNT
        and _float(metrics.get("winRate")) > SHADOW_MIN_WIN_RATE
        and _float(metrics.get("profitFactor")) > SHADOW_MIN_PROFIT_FACTOR
        and _float(metrics.get("avgReturn")) > SHADOW_MIN_AVG_RETURN
    )


def _float(value: Any) -> float:
    if value is None:
        return float("-inf")
    return float(value)


def _matching_quality_rows(validation: dict[str, Any], test: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    val_rows = _quality_rows(validation)
    test_rows = {float(row["minConfidence"]): row for row in _quality_rows(test)}
    return [
        (val_row, test_rows[float(val_row["minConfidence"])])
        for val_row in val_rows
        if float(val_row["minConfidence"]) in test_rows
    ]


def _quality_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = metrics.get("confidenceThresholds")
    if isinstance(rows, list) and rows:
        return [dict(row) for row in rows if row.get("minConfidence") is not None]
    return [{**metrics, "minConfidence": 0.5}]
