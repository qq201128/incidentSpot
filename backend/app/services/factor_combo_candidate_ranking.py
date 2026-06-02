from __future__ import annotations

from math import isfinite
from typing import Any

OFFLINE_PREFILTER_POLICY = "offline_cross_period_stability_sample_size_profit_factor_prefilter_only"
OFFLINE_RANKING_POLICY = ["cross_period_stability", "sample_count", "profit_factor"]


def offline_candidate_rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    walk_forward = _walk_forward(row)
    return (
        _num(_first_present(walk_forward.get("stabilityScore"), walk_forward.get("score"))),
        _recent_rolling_score(row.get("recentRollingResult")),
        _num(walk_forward.get("oosWinRate")),
        _sample_count(row),
        _num(row.get("profitFactor")),
        _passed_score(row.get("walkForwardPassed")),
        _num(_first_present(row.get("winRate"), row.get("backtestWinRate"))),
        _num(row.get("factorScore")),
    )


def _recent_rolling_score(value: Any) -> float:
    if isinstance(value, list):
        return _min_metric(value, "winRate")
    if not isinstance(value, dict):
        return float("-inf")
    stability = value.get("stability")
    if isinstance(stability, dict):
        score = _finite_float(stability.get("worstRecentRollingWinRate"))
        if score is not None:
            return score
    for key in ("recentRolling", "rollingWindows", "windows"):
        rows = value.get(key)
        if isinstance(rows, list):
            return _min_metric(rows, "winRate")
    return _num(_first_present(value.get("winRate"), value.get("score")))


def _sample_count(row: dict[str, Any]) -> float:
    for key in ("totalPeriods", "trades", "sampleCount"):
        value = _finite_float(row.get(key))
        if value is not None:
            return value
    return float("-inf")


def _min_metric(rows: list[Any], key: str) -> float:
    values = [_finite_float(row.get(key)) for row in rows if isinstance(row, dict)]
    selected = [value for value in values if value is not None]
    return min(selected) if selected else float("-inf")


def _walk_forward(row: dict[str, Any]) -> dict[str, Any]:
    walk_forward = row.get("walkForward")
    return walk_forward if isinstance(walk_forward, dict) else {}


def _passed_score(value: Any) -> float:
    if value is True or value == 1:
        return 1.0
    if value is False or value == 0:
        return 0.0
    return float("-inf")


def _num(value: Any) -> float:
    number = _finite_float(value)
    return number if number is not None else float("-inf")


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
