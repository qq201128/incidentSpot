from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import pandas as pd
from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS
from app.services.factor_combo_scoring import combination_score
from app.services.factor_combination_quality_gate import (
    EntryWindow,
    LIVE_MIN_PROFIT_FACTOR,
    LIVE_MIN_WIN_RATE,
    RegimeGate,
    backtest_aligned_quality,
    live_evidence,
    quality_gate,
    resolve_apply_quality_gate,
)
from app.services.factor_combination_signal_payloads import (
    LiveSignalPayloadContext,
    live_direction,
    live_signal_payload,
)
from app.services.factor_duration_alignment import live_duration_entry_index
from app.services.factor_learning_signal_filter import MEMORY_NOT_PROVIDED, enrich_signal_with_factor_learning
from app.services.factor_regime_analysis import current_factor_regime, factor_regime_gate
from app.services.factor_signal_timing import FactorSignalTiming, combination_kline_close_timing

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
    timing = _signal_timing(row, symbol, duration, context)
    regime_gate = _regime_gate(frame, signal.index, duration, signal.direction, confidence)
    quality = _signal_quality(
        row,
        signal,
        confidence=confidence,
        timing=timing,
        regime_gate=regime_gate,
        entry_open_time=entry_open_time,
        entry_grace_ms=entry_grace_ms,
        apply_quality_gate=apply_quality_gate,
    )
    priced = _priced_payload(frame, signal, live_signal_payload(
        _payload_context(row, symbol, duration, signal, confidence, quality, timing, regime_gate)
    ))
    use_quality_gate = resolve_apply_quality_gate(apply_quality_gate)
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


def _signal_timing(row: dict[str, Any], symbol: str, duration: str, context: SignalBuildContext | None):
    return combination_kline_close_timing(
        row,
        symbol=symbol,
        duration=duration,
        mined_by_name=None if context is None else context.mined_by_name,
    )


def _signal_quality(
    row: dict[str, Any],
    signal: "_ComboSignal",
    *,
    confidence: float,
    timing: Any,
    regime_gate: RegimeGate,
    entry_open_time: int | None,
    entry_grace_ms: int | None,
    apply_quality_gate: bool,
) -> dict[str, Any]:
    if not resolve_apply_quality_gate(apply_quality_gate):
        return backtest_aligned_quality()
    window = EntryWindow(entry_open_time, entry_grace_ms)
    return quality_gate(row, confidence, window, timing=timing, score=signal.score, regime=regime_gate)


def _payload_context(
    row: dict[str, Any],
    symbol: str,
    duration: str,
    signal: "_ComboSignal",
    confidence: float,
    quality: dict[str, Any],
    timing: Any,
    regime_gate: RegimeGate,
) -> LiveSignalPayloadContext:
    return LiveSignalPayloadContext(
        row, symbol, duration, signal.score, signal.historical_median, signal.index,
        signal.direction, confidence, quality, timing, regime_gate.regime or {},
    )


def _priced_payload(frame: pd.DataFrame, signal: "_ComboSignal", payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "entryPrice": _frame_value(frame, signal.index, "close"),
        "sourceOpenTime": _frame_value(frame, signal.index, "open_time"),
    }


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
    direction = live_direction(row, score, historical_median)
    return _ComboSignal(score=score, historical_median=historical_median, direction=direction, index=index)


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
    evidence = live_evidence(row)
    value = _finite_float(evidence.get("winRate"))
    if value is None:
        value = _finite_float(row.get("winRate"))
    if value is None:
        raise ValueError(f"combination row missing live or historical winRate: {row.get('factorName')}")
    return max(0.0, min(float(value), 0.99))


def _regime_gate(
    frame: pd.DataFrame,
    index: Any,
    duration: str,
    direction: str,
    confidence: float,
) -> RegimeGate:
    gate = factor_regime_gate(direction, confidence, current_factor_regime(frame, index, duration))
    return RegimeGate(
        passed=gate.passed,
        reason=gate.reason,
        regime=gate.regime,
        min_win_rate=gate.min_win_rate,
    )


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
