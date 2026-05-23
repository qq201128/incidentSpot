from __future__ import annotations

import os
from dataclasses import dataclass
from math import isfinite
from typing import Any

import pandas as pd

from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS
from app.services.factor_combo_scoring import combination_score
from app.services.factor_combo_simulation_keys import is_high_winrate_combo_name
from app.services.factor_combination_service import COMBINATION_METHOD
from app.services.factor_duration_alignment import live_duration_entry_index
from app.services.factor_learning_common import SUCCESS_PROFIT_FACTOR_MIN, SUCCESS_WIN_RATE_MIN
from app.services.factor_learning_signal_filter import MEMORY_NOT_PROVIDED, enrich_signal_with_factor_learning
from app.services.factor_signal_timing import FactorSignalTiming, combination_kline_close_timing
from app.services.kline_timing import is_within_entry_grace

STRICT_LIVE_MIN_WIN_RATE = 0.60
LIVE_MIN_WIN_RATE = max(SUCCESS_WIN_RATE_MIN, STRICT_LIVE_MIN_WIN_RATE)
LIVE_MIN_PROFIT_FACTOR = SUCCESS_PROFIT_FACTOR_MIN
LIVE_MIN_TOTAL_PERIODS = BACKTEST_MIN_PERIODS
PROBABILITY_DECIMALS = 4
SCORE_DECIMALS = 6
DEFAULT_SIGNAL_THRESHOLD = 0.0
COMBO_MEDIAN_DECIMALS = 6


@dataclass(frozen=True)
class _EntryWindow:
    open_time: int | None
    grace_ms: int | None


@dataclass(frozen=True)
class _SignalContext:
    row: dict[str, Any]
    symbol: str
    duration: str
    score: float
    historical_median: float
    index: Any
    direction: str
    confidence: float
    quality: dict[str, Any]
    timing: FactorSignalTiming


@dataclass(frozen=True)
class SignalBuildContext:
    learning_memory: dict[str, Any] | None
    zscore_cache: dict[tuple[str, int], pd.Series]
    mined_by_name: dict[str, dict[str, Any]]


def build_live_signal_from_ranking(
    frame: pd.DataFrame,
    row: dict[str, Any],
    *,
    symbol: str,
    duration: str,
    entry_open_time: int | None = None,
    entry_grace_ms: int | None = None,
    context: SignalBuildContext | None = None,
    apply_quality_gate: bool = True,
) -> dict[str, Any]:
    signal = _combo_signal_at_duration_entry(
        frame,
        row,
        duration=duration,
        entry_open_time=entry_open_time,
        zscore_cache=None if context is None else context.zscore_cache,
    )
    confidence = _live_confidence(row)
    window = _EntryWindow(entry_open_time, entry_grace_ms)
    timing = combination_kline_close_timing(
        row,
        symbol=symbol,
        duration=duration,
        mined_by_name=None if context is None else context.mined_by_name,
    )
    use_quality_gate = _resolve_apply_quality_gate(apply_quality_gate)
    quality = (
        _quality_gate(row, confidence, window, timing=timing, score=signal.score)
        if use_quality_gate
        else _backtest_aligned_quality()
    )
    payload = _live_signal_payload(
        _SignalContext(
            row=row,
            symbol=symbol,
            duration=duration,
            score=signal.score,
            historical_median=signal.historical_median,
            index=signal.index,
            direction=signal.direction,
            confidence=confidence,
            quality=quality,
            timing=timing,
        )
    )
    priced = {
        **payload,
        "entryPrice": _frame_value(frame, signal.index, "close"),
        "sourceOpenTime": _frame_value(frame, signal.index, "open_time"),
    }
    return enrich_signal_with_factor_learning(
        priced,
        frame,
        signal.index,
        symbol=symbol,
        duration=duration,
        memory=MEMORY_NOT_PROVIDED if context is None else context.learning_memory,
        zscore_cache=None if context is None else context.zscore_cache,
        enforce_quality_gate=use_quality_gate,
    )


@dataclass(frozen=True)
class _ComboSignal:
    score: float
    historical_median: float
    direction: str
    index: Any


def _combo_signal_at_duration_entry(
    frame: pd.DataFrame,
    row: dict[str, Any],
    *,
    duration: str,
    entry_open_time: int | None,
    zscore_cache: dict[tuple[str, int], pd.Series] | None,
) -> _ComboSignal:
    members = _row_members(row)
    _require_member_columns(frame, members)
    index = live_duration_entry_index(frame, duration, entry_open_time)
    score_series = _combo_score_series(frame, members, zscore_cache)
    return _combo_direction_from_row_rule(score_series, index, row, duration)


def _combo_score_series(
    frame: pd.DataFrame,
    members: list[dict[str, Any]],
    zscore_cache: dict[tuple[str, int], pd.Series] | None,
) -> pd.Series:
    if zscore_cache is None:
        return combination_score(frame, members).replace([float("inf"), float("-inf")], float("nan"))
    scores = [_member_score_series(frame, member, zscore_cache) for member in members]
    return pd.concat(scores, axis=1).mean(axis=1).replace([float("inf"), float("-inf")], float("nan"))


def _combo_direction_from_row_rule(
    score_series: pd.Series,
    index: Any,
    row: dict[str, Any],
    duration: str,
) -> _ComboSignal:
    factor_name = row.get("factorName")
    score = _finite_float(_series_value_at(score_series, index))
    if score is None:
        raise ValueError(f"combination signal has no finite score at {duration} entry: {factor_name}")
    median_series = score_series.expanding(min_periods=BACKTEST_MIN_PERIODS).median().shift(1)
    historical_median = _finite_float(_series_value_at(median_series, index))
    if historical_median is None:
        raise ValueError(f"combination signal has insufficient historical median at {duration} entry: {factor_name}")
    direction = _live_direction(row, score, historical_median)
    return _ComboSignal(score=score, historical_median=historical_median, direction=direction, index=index)


def _live_direction(row: dict[str, Any], score: float, historical_median: float) -> str:
    threshold = _signal_threshold(row)
    if is_high_winrate_combo_name(str(row.get("factorName") or "")) and threshold > 0:
        if score >= threshold:
            return "up"
        if score <= -threshold:
            return "down"
    return "up" if score >= historical_median else "down"


def _combo_score_at_index(
    frame: pd.DataFrame,
    members: list[dict[str, Any]],
    index: Any,
    zscore_cache: dict[tuple[str, int], pd.Series] | None,
) -> float | None:
    if zscore_cache is None:
        return _finite_float(_series_value_at(combination_score(frame, members), index))
    values = [_member_score_at_index(frame, member, index, zscore_cache) for member in members]
    finite = [value for value in values if value is not None]
    return None if not finite else sum(finite) / len(finite)


def _member_score_series(
    frame: pd.DataFrame,
    member: dict[str, Any],
    zscore_cache: dict[tuple[str, int], pd.Series],
) -> pd.Series:
    name = str(member["name"])
    orientation = int(member.get("orientation") or 1)
    key = (name, orientation)
    if key not in zscore_cache:
        zscore_cache[key] = combination_score(frame, [member])
    return zscore_cache[key]


def _member_score_at_index(
    frame: pd.DataFrame,
    member: dict[str, Any],
    index: Any,
    zscore_cache: dict[tuple[str, int], pd.Series],
) -> float | None:
    name = str(member["name"])
    orientation = int(member.get("orientation") or 1)
    key = (name, orientation)
    if key not in zscore_cache:
        zscore_cache[key] = combination_score(frame, [member])
    return _finite_float(_series_value_at(zscore_cache[key], index))


def _live_signal_payload(ctx: _SignalContext) -> dict[str, Any]:
    probability_up = ctx.confidence if ctx.direction == "up" else 1.0 - ctx.confidence
    return {
        "symbol": ctx.symbol.upper(),
        "duration": ctx.duration,
        "factorName": ctx.row.get("factorName"),
        "factorDisplayName": ctx.row.get("factorDisplayName"),
        "comboRank": ctx.row.get("comboRank"),
        "members": _row_members(ctx.row),
        "direction": ctx.direction,
        "probabilityUp": round(probability_up, PROBABILITY_DECIMALS),
        "confidence": round(ctx.confidence, PROBABILITY_DECIMALS),
        "score": round(ctx.score, SCORE_DECIMALS),
        "historicalMedianScore": round(ctx.historical_median, COMBO_MEDIAN_DECIMALS),
        "source": "factor_combination_ranking",
        "method": ctx.row.get("method") or COMBINATION_METHOD,
        "historicalWinRate": ctx.row.get("winRate"),
        "historicalProfitFactor": ctx.row.get("profitFactor"),
        "historicalSharpe": ctx.row.get("sharpe"),
        "historicalIr": ctx.row.get("ir"),
        "historicalTotalPeriods": ctx.row.get("totalPeriods"),
        "walkForwardPassed": ctx.row.get("walkForwardPassed"),
        "walkForwardFailureReason": ctx.row.get("walkForwardFailureReason"),
        "qualityPassed": ctx.quality["passed"],
        "qualityMetricsPassed": ctx.quality["metricsPassed"],
        "qualityThresholdPassed": ctx.quality["thresholdPassed"],
        "signalThreshold": _signal_threshold(ctx.row),
        "qualityEntryWindowPassed": ctx.quality["entryWindowPassed"],
        "factorTimingMode": ctx.timing.mode,
        "factorTimingPassed": ctx.quality["factorTimingPassed"],
        "factorTimingReason": ctx.timing.reason,
        "factorTimingEligibleMembers": list(ctx.timing.eligible_members),
        "factorTimingBlockedMembers": list(ctx.timing.blocked_members),
        "qualityGateReason": ctx.quality["reason"],
        "qualityMinWinRate": LIVE_MIN_WIN_RATE,
        "qualityMinProfitFactor": LIVE_MIN_PROFIT_FACTOR,
        "qualityMinPeriods": _min_total_periods(ctx.row),
        "frameIndex": str(ctx.index),
    }


def _resolve_apply_quality_gate(requested: bool) -> bool:
    env = os.getenv("FACTOR_COMBO_LIVE_QUALITY_GATE", "").strip().lower()
    if env in {"1", "true", "yes"}:
        return True
    if env in {"0", "false", "no"}:
        return False
    return requested


def _backtest_aligned_quality() -> dict[str, Any]:
    """Combo backtests do not apply live trade-quality gates; keep SIM aligned with that path."""
    return {
        "passed": True,
        "metricsPassed": True,
        "thresholdPassed": True,
        "entryWindowPassed": True,
        "factorTimingPassed": True,
        "reason": "backtest_aligned",
    }


def _quality_gate(
    row: dict[str, Any],
    confidence: float,
    window: _EntryWindow,
    *,
    timing: FactorSignalTiming,
    score: float,
) -> dict[str, Any]:
    metric_reason = _quality_metric_reason(row, confidence)
    threshold_reason = _threshold_reason(row, score)
    entry_passed = _entry_window_passed(window)
    metrics_passed = metric_reason == "passed"
    threshold_passed = threshold_reason == "passed"
    passed = metrics_passed and threshold_passed and entry_passed is not False and timing.passed
    return {
        "passed": passed,
        "metricsPassed": metrics_passed,
        "thresholdPassed": threshold_passed,
        "entryWindowPassed": entry_passed,
        "factorTimingPassed": timing.passed,
        "reason": _quality_reason(metric_reason, threshold_reason, entry_passed, timing),
    }


def _quality_metric_reason(row: dict[str, Any], confidence: float) -> str:
    if confidence < LIVE_MIN_WIN_RATE:
        return "win_rate_below_min"
    profit_factor = _finite_float(row.get("profitFactor"))
    if profit_factor is None:
        return "profit_factor_missing"
    if profit_factor < LIVE_MIN_PROFIT_FACTOR:
        return "profit_factor_below_min"
    min_periods = _min_total_periods(row)
    total_periods = _finite_float(row.get("totalPeriods"))
    if total_periods is None:
        return "total_periods_missing"
    if total_periods < min_periods:
        return "total_periods_below_min"
    walk_forward_reason = _walk_forward_reason(row)
    if walk_forward_reason is not None:
        return walk_forward_reason
    return "passed"


def _walk_forward_reason(row: dict[str, Any]) -> str | None:
    if is_high_winrate_combo_name(str(row.get("factorName") or "")):
        return None
    passed = row.get("walkForwardPassed")
    if passed is True or passed == 1:
        return None
    if passed is False or passed == 0:
        return str(row.get("walkForwardFailureReason") or "walk_forward_failed")
    return "walk_forward_missing"


def _min_total_periods(row: dict[str, Any]) -> float:
    value = _finite_float(row.get("minTrades"))
    if value is None:
        return float(LIVE_MIN_TOTAL_PERIODS)
    if value <= 0:
        raise ValueError(f"combination row has invalid minTrades: {row.get('factorName')}")
    return value


def _threshold_reason(row: dict[str, Any], score: float) -> str:
    threshold = _signal_threshold(row)
    if abs(score) < threshold:
        return "signal_threshold_not_met"
    return "passed"


def _signal_threshold(row: dict[str, Any]) -> float:
    threshold = _finite_float(row.get("threshold"))
    if threshold is None:
        return DEFAULT_SIGNAL_THRESHOLD
    if threshold < 0:
        raise ValueError(f"combination row has negative threshold: {row.get('factorName')}")
    return threshold


def _entry_window_passed(window: _EntryWindow) -> bool | None:
    if window.open_time is None or window.grace_ms is None:
        return None
    return is_within_entry_grace(int(window.open_time), grace_ms=int(window.grace_ms))


def _quality_reason(
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


def _row_members(row: dict[str, Any]) -> list[dict[str, Any]]:
    members = row.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"combination row missing members: {row.get('factorName')}")
    return [dict(member) for member in members]


def _require_member_columns(frame: pd.DataFrame, members: list[dict[str, Any]]) -> None:
    missing = [member["name"] for member in members if member["name"] not in frame.columns]
    if missing:
        raise ValueError(f"combination signal missing factors: {', '.join(missing)}")


def _series_value_at(series: pd.Series, index: Any) -> Any:
    value = series.loc[index]
    if isinstance(value, pd.Series):
        return value.iloc[-1]
    return value


def _live_confidence(row: dict[str, Any]) -> float:
    value = _finite_float(row.get("winRate"))
    if value is None:
        raise ValueError(f"combination row missing winRate: {row.get('factorName')}")
    return max(0.0, min(float(value), 0.99))


def _frame_value(frame: pd.DataFrame, index: Any, column: str) -> float | int | None:
    if column not in frame.columns:
        return None
    value = frame.at[index, column]
    if value is None:
        return None
    number = float(value)
    if not isfinite(number):
        return None
    return int(number) if column == "open_time" else number


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None
