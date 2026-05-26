from __future__ import annotations

import os
from dataclasses import dataclass
from math import isfinite
from typing import Any

from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS
from app.services.factor_combo_simulation_keys import is_high_winrate_combo_name
from app.services.factor_learning_common import SUCCESS_PROFIT_FACTOR_MIN, SUCCESS_WIN_RATE_MIN
from app.services.factor_signal_timing import FactorSignalTiming
from app.services.kline_timing import is_within_entry_grace

LIVE_MIN_WIN_RATE = SUCCESS_WIN_RATE_MIN
LIVE_MIN_PROFIT_FACTOR = SUCCESS_PROFIT_FACTOR_MIN
LIVE_MIN_TOTAL_PERIODS = BACKTEST_MIN_PERIODS
DEFAULT_SIGNAL_THRESHOLD = 0.0


@dataclass(frozen=True)
class EntryWindow:
    open_time: int | None
    grace_ms: int | None


def resolve_apply_quality_gate(requested: bool) -> bool:
    env = os.getenv("FACTOR_COMBO_LIVE_QUALITY_GATE", "").strip().lower()
    if env in {"1", "true", "yes"}:
        return True
    if env in {"0", "false", "no"}:
        return False
    return requested


def backtest_aligned_quality() -> dict[str, Any]:
    return {
        "passed": True,
        "metricsPassed": True,
        "thresholdPassed": True,
        "entryWindowPassed": True,
        "factorTimingPassed": True,
        "reason": "backtest_aligned",
    }


def quality_gate(
    row: dict[str, Any],
    confidence: float,
    window: EntryWindow,
    *,
    timing: FactorSignalTiming,
    score: float,
) -> dict[str, Any]:
    metric_reason = quality_metric_reason(row, confidence)
    threshold_reason = threshold_reason_for(row, score)
    entry_passed = entry_window_passed(window)
    metrics_passed = metric_reason == "passed"
    threshold_passed = threshold_reason == "passed"
    passed = metrics_passed and threshold_passed and entry_passed is not False and timing.passed
    return {
        "passed": passed,
        "metricsPassed": metrics_passed,
        "thresholdPassed": threshold_passed,
        "entryWindowPassed": entry_passed,
        "factorTimingPassed": timing.passed,
        "reason": quality_reason(metric_reason, threshold_reason, entry_passed, timing),
    }


def quality_metric_reason(row: dict[str, Any], confidence: float) -> str:
    if confidence < LIVE_MIN_WIN_RATE:
        return "win_rate_below_min"
    profit_factor = finite_float(row.get("profitFactor"))
    if profit_factor is None:
        return "profit_factor_missing"
    if profit_factor < LIVE_MIN_PROFIT_FACTOR:
        return "profit_factor_below_min"
    return period_and_walk_forward_reason(row)


def period_and_walk_forward_reason(row: dict[str, Any]) -> str:
    min_periods = min_total_periods(row)
    total_periods = finite_float(row.get("totalPeriods"))
    if total_periods is None:
        return "total_periods_missing"
    if total_periods < min_periods:
        return "total_periods_below_min"
    reason = walk_forward_reason(row)
    return reason or "passed"


def walk_forward_reason(row: dict[str, Any]) -> str | None:
    if is_high_winrate_combo_name(str(row.get("factorName") or "")):
        return None
    passed = row.get("walkForwardPassed")
    if passed is True or passed == 1:
        return None
    if passed is False or passed == 0:
        return str(row.get("walkForwardFailureReason") or "walk_forward_failed")
    return "walk_forward_missing"


def min_total_periods(row: dict[str, Any]) -> float:
    value = finite_float(row.get("minTrades"))
    if value is None:
        return float(LIVE_MIN_TOTAL_PERIODS)
    if value <= 0:
        raise ValueError(f"combination row has invalid minTrades: {row.get('factorName')}")
    return value


def threshold_reason_for(row: dict[str, Any], score: float) -> str:
    if abs(score) < signal_threshold(row):
        return "signal_threshold_not_met"
    return "passed"


def signal_threshold(row: dict[str, Any]) -> float:
    threshold = finite_float(row.get("threshold"))
    if threshold is None:
        return DEFAULT_SIGNAL_THRESHOLD
    if threshold < 0:
        raise ValueError(f"combination row has negative threshold: {row.get('factorName')}")
    return threshold


def entry_window_passed(window: EntryWindow) -> bool | None:
    if window.open_time is None or window.grace_ms is None:
        return None
    return is_within_entry_grace(int(window.open_time), grace_ms=int(window.grace_ms))


def quality_reason(
    metric_reason: str,
    threshold_reason: str,
    entry_passed: bool | None,
    timing: FactorSignalTiming,
) -> str:
    if metric_reason != "passed":
        return metric_reason
    if threshold_reason != "passed":
        return threshold_reason
    if not timing.passed:
        return timing.reason
    if entry_passed is False:
        return "entry_window_closed"
    return "passed"


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None
