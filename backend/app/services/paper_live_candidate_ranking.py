from __future__ import annotations

import json
from typing import Any

STATUS_STABLE = "paper_stable"
STATUS_COLLECTING = "paper_collecting"
STATUS_BACKTEST = "backtest_candidate"


def candidate_rank_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    metrics = candidate["metrics"]
    stability = metrics.get("paperStability") or {}
    return (
        _status_priority(str(candidate.get("status") or "")),
        _stability_score(stability),
        _num(candidate.get("oosWinRate")),
        _walk_forward_score(candidate.get("walkForwardResult")),
        _recent_rolling_score(candidate.get("recentRollingResult")),
        _num(metrics.get("profitFactor")),
        _num(metrics.get("avgReturn")),
        float(metrics.get("sampleCount") or 0),
    )


def performance_comparison(candidate: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    backtest = candidate.get("high_winrate_gate_value")
    paper = metrics.get("winRate")
    return {
        "policy": "backtest_oos_walk_forward_recent_rolling_are_prefilter_only",
        "backtestWinRate": backtest,
        "oosWinRate": candidate.get("oos_win_rate"),
        "walkForwardResult": _json_value(candidate.get("walk_forward_result")),
        "recentRollingResult": _json_value(candidate.get("recent_rolling_result")),
        "validationWinRate": candidate.get("validation_win_rate"),
        "paperLiveWinRate": paper,
        "paperLiveSampleCount": metrics.get("sampleCount"),
        "winRateGap": _gap(backtest, paper),
        "paperLiveStatus": None,
    }


def focus_pool(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    eligible = [row for row in candidates if row.get("status") not in {"paper_failed", "invalid_data_leakage"}]
    return eligible[:limit]


def _status_priority(status: str) -> float:
    if status == STATUS_STABLE:
        return 3.0
    if status == STATUS_COLLECTING:
        return 2.0
    if status == STATUS_BACKTEST:
        return 1.0
    return 0.0


def _stability_score(stability: dict[str, Any]) -> float:
    rolling = stability.get("rollingWindows") if isinstance(stability, dict) else None
    if not isinstance(rolling, list) or not rolling:
        return float("-inf")
    rates = [_num(row.get("winRate")) for row in rolling if isinstance(row, dict)]
    return min(rates) if rates else float("-inf")


def _walk_forward_score(value: Any) -> float:
    if not isinstance(value, dict):
        return float("-inf")
    if value.get("status") == "passed" or value.get("passed") is True:
        return max(_num(value.get("score")), 0.0)
    return _num(value.get("score"))


def _recent_rolling_score(value: Any) -> float:
    if isinstance(value, dict):
        stability = value.get("paperStability") or value
        return _stability_score(stability if isinstance(stability, dict) else {})
    return float("-inf")


def _gap(first: Any, second: Any) -> float | None:
    if first is None or second is None:
        return None
    return round(float(first) - float(second), 4)


def _num(value: Any) -> float:
    return float(value) if value is not None else float("-inf")


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
