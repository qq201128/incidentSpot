from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import pandas as pd

from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS
from app.services.factor_combo_scoring import combination_score
from app.services.factor_combination_service import COMBINATION_METHOD
from app.services.factor_learning_common import SUCCESS_PROFIT_FACTOR_MIN, SUCCESS_WIN_RATE_MIN
from app.services.factor_learning_signal_filter import enrich_signal_with_factor_learning
from app.services.kline_timing import is_within_entry_grace

LIVE_MIN_WIN_RATE = SUCCESS_WIN_RATE_MIN
LIVE_MIN_PROFIT_FACTOR = SUCCESS_PROFIT_FACTOR_MIN
LIVE_MIN_TOTAL_PERIODS = BACKTEST_MIN_PERIODS
PROBABILITY_DECIMALS = 4
SCORE_DECIMALS = 6


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
    index: Any
    direction: str
    confidence: float
    quality: dict[str, Any]


def build_live_signal_from_ranking(
    frame: pd.DataFrame,
    row: dict[str, Any],
    *,
    symbol: str,
    duration: str,
    entry_open_time: int | None = None,
    entry_grace_ms: int | None = None,
) -> dict[str, Any]:
    score, index = _latest_combo_score(frame, row)
    direction = "up" if score >= 0 else "down"
    confidence = _live_confidence(row)
    window = _EntryWindow(entry_open_time, entry_grace_ms)
    quality = _quality_gate(row, confidence, window)
    payload = _live_signal_payload(
        _SignalContext(
            row=row,
            symbol=symbol,
            duration=duration,
            score=score,
            index=index,
            direction=direction,
            confidence=confidence,
            quality=quality,
        )
    )
    priced = {
        **payload,
        "entryPrice": _frame_value(frame, index, "close"),
        "sourceOpenTime": _frame_value(frame, index, "open_time"),
    }
    return enrich_signal_with_factor_learning(priced, frame, index, symbol=symbol, duration=duration)


def _latest_combo_score(frame: pd.DataFrame, row: dict[str, Any]) -> tuple[float, Any]:
    members = _row_members(row)
    missing = [member["name"] for member in members if member["name"] not in frame.columns]
    if missing:
        raise ValueError(f"combination signal missing factors: {', '.join(missing)}")
    scores = combination_score(frame, members).dropna()
    if scores.empty:
        raise ValueError(f"combination signal has no finite score: {row.get('factorName')}")
    return float(scores.iloc[-1]), scores.index[-1]


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
        "source": "factor_combination_ranking",
        "method": ctx.row.get("method") or COMBINATION_METHOD,
        "historicalWinRate": ctx.row.get("winRate"),
        "historicalProfitFactor": ctx.row.get("profitFactor"),
        "historicalSharpe": ctx.row.get("sharpe"),
        "historicalIr": ctx.row.get("ir"),
        "historicalTotalPeriods": ctx.row.get("totalPeriods"),
        "qualityPassed": ctx.quality["passed"],
        "qualityMetricsPassed": ctx.quality["metricsPassed"],
        "qualityEntryWindowPassed": ctx.quality["entryWindowPassed"],
        "qualityGateReason": ctx.quality["reason"],
        "qualityMinWinRate": LIVE_MIN_WIN_RATE,
        "qualityMinProfitFactor": LIVE_MIN_PROFIT_FACTOR,
        "qualityMinPeriods": LIVE_MIN_TOTAL_PERIODS,
        "frameIndex": str(ctx.index),
    }


def _quality_gate(row: dict[str, Any], confidence: float, window: _EntryWindow) -> dict[str, Any]:
    metric_reason = _quality_metric_reason(row, confidence)
    entry_passed = _entry_window_passed(window)
    metrics_passed = metric_reason == "passed"
    passed = metrics_passed and entry_passed is not False
    return {
        "passed": passed,
        "metricsPassed": metrics_passed,
        "entryWindowPassed": entry_passed,
        "reason": _quality_reason(metric_reason, entry_passed),
    }


def _quality_metric_reason(row: dict[str, Any], confidence: float) -> str:
    if confidence < LIVE_MIN_WIN_RATE:
        return "win_rate_below_min"
    profit_factor = _finite_float(row.get("profitFactor"))
    if profit_factor is None:
        return "profit_factor_missing"
    if profit_factor < LIVE_MIN_PROFIT_FACTOR:
        return "profit_factor_below_min"
    total_periods = _finite_float(row.get("totalPeriods"))
    if total_periods is None:
        return "total_periods_missing"
    if total_periods < LIVE_MIN_TOTAL_PERIODS:
        return "total_periods_below_min"
    return "passed"


def _entry_window_passed(window: _EntryWindow) -> bool | None:
    if window.open_time is None or window.grace_ms is None:
        return None
    return is_within_entry_grace(int(window.open_time), grace_ms=int(window.grace_ms))


def _quality_reason(metric_reason: str, entry_passed: bool | None) -> str:
    if metric_reason != "passed":
        return metric_reason
    if entry_passed is False:
        return "entry_window_closed"
    return "passed"


def _row_members(row: dict[str, Any]) -> list[dict[str, Any]]:
    members = row.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"combination row missing members: {row.get('factorName')}")
    return [dict(member) for member in members]


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
