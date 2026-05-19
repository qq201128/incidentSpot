from __future__ import annotations

from typing import Any

TRADE_GATE_THRESHOLDS = (0.55, 0.60, 0.65, 0.70)
TARGET_VALIDATION_WIN_RATE = 0.70
MIN_VALIDATION_PROFIT_FACTOR = 1.0
MIN_VALIDATION_AVG_RETURN = 0.0
MIN_THRESHOLD_SAMPLE_COUNT = 50


def validation_gate(validation: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    criteria = validation_gate_criteria()
    candidates = _threshold_candidates(validation, test)
    for row in candidates:
        if _threshold_passes(row["validation"]) and _threshold_passes(row["test"]):
            return {
                "status": "passed",
                "criteria": criteria,
                "minConfidence": row["minConfidence"],
                "validation": row["validation"],
                "test": row["test"],
            }
    return {
        "status": "failed",
        "reason": "no_validation_confidence_threshold_met",
        "criteria": criteria,
        "candidates": candidates,
    }


def validation_failure_reason(gate: dict[str, Any]) -> str | None:
    if gate["status"] == "passed":
        return None
    return str(gate["reason"])


def validation_gate_criteria() -> dict[str, Any]:
    return {
        "thresholds": list(TRADE_GATE_THRESHOLDS),
        "targetWinRateInclusive": TARGET_VALIDATION_WIN_RATE,
        "minProfitFactorExclusive": MIN_VALIDATION_PROFIT_FACTOR,
        "minAvgReturnExclusive": MIN_VALIDATION_AVG_RETURN,
        "minThresholdSampleCount": MIN_THRESHOLD_SAMPLE_COUNT,
        "requiresValidationAndTest": True,
    }


def _threshold_candidates(validation: dict[str, Any], test: dict[str, Any]) -> list[dict[str, Any]]:
    val_by_threshold = _threshold_map(validation)
    test_by_threshold = _threshold_map(test)
    return [
        {
            "minConfidence": threshold,
            "validation": val_by_threshold.get(threshold) or _missing_threshold(threshold),
            "test": test_by_threshold.get(threshold) or _missing_threshold(threshold),
        }
        for threshold in TRADE_GATE_THRESHOLDS
    ]


def _threshold_map(metrics: dict[str, Any]) -> dict[float, dict[str, Any]]:
    rows = metrics.get("confidenceThresholds") or []
    return {
        float(row.get("minConfidence")): dict(row)
        for row in rows
        if row.get("minConfidence") is not None
    }


def _missing_threshold(threshold: float) -> dict[str, Any]:
    return {
        "minConfidence": threshold,
        "sampleCount": 0,
        "winRate": None,
        "profitFactor": None,
        "avgReturn": None,
    }


def _threshold_passes(row: dict[str, Any]) -> bool:
    return (
        int(row.get("sampleCount") or 0) >= MIN_THRESHOLD_SAMPLE_COUNT
        and row.get("winRate") is not None
        and float(row["winRate"]) >= TARGET_VALIDATION_WIN_RATE
        and row.get("profitFactor") is not None
        and float(row["profitFactor"]) > MIN_VALIDATION_PROFIT_FACTOR
        and row.get("avgReturn") is not None
        and float(row["avgReturn"]) > MIN_VALIDATION_AVG_RETURN
    )
