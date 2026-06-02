from __future__ import annotations

from typing import Any

ACTIVE_STATUSES = frozenset({"trade_active", "trained"})
SHADOW_STATUS = "shadow_active"


def model_candidate_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(records),
        "latest": records[-1] if records else None,
        "bestTradeCandidate": _best_record(records, ACTIVE_STATUSES),
        "bestShadowCandidate": _best_record(records, {SHADOW_STATUS}),
        "bestValidationCandidate": _best_metric_record(records, "validation"),
        "bestTestCandidate": _best_metric_record(records, "test"),
    }


def _best_record(records: list[dict[str, Any]], statuses: set[str] | frozenset[str]) -> dict[str, Any] | None:
    selected = [row for row in records if str(row.get("status") or "") in statuses]
    return max(selected, key=_validation_score) if selected else None


def _best_metric_record(records: list[dict[str, Any]], metric_key: str) -> dict[str, Any] | None:
    selected = [
        row
        for row in records
        if isinstance(row.get(metric_key), dict) and _has_metric_evidence(row[metric_key])
    ]
    return max(selected, key=lambda row: _metric_score(row, metric_key)) if selected else None


def _validation_score(record: dict[str, Any]) -> tuple[float, float, int]:
    return _metric_score(record, "validation")


def _metric_score(record: dict[str, Any], metric_key: str) -> tuple[float, float, int]:
    metrics = record.get(metric_key) or {}
    return (
        _num(metrics.get("winRate")),
        _num(metrics.get("profitFactor")),
        int(metrics.get("sampleCount") or 0),
    )


def _num(value: Any) -> float:
    return float(value) if value is not None else float("-inf")


def _has_metric_evidence(metrics: dict[str, Any]) -> bool:
    return any(metrics.get(key) is not None for key in ("winRate", "profitFactor", "sampleCount"))
