from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import pandas as pd

from app.services.factor_duration_alignment import backtest_duration_frame
from app.services.factor_performance_metrics import signal_returns
from app.services.factor_registry import FactorDefinition

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
VALIDATION_MIN_SAMPLE_COUNT = 50
VALIDATION_MIN_WIN_RATE = 0.62
VALIDATION_MIN_PROFIT_FACTOR = 1.05
VALIDATION_MIN_AVG_RETURN = 0.0
RECENT_SAMPLE_COUNT = 50
RECENT_MIN_WIN_RATE = 0.50
METRIC_DECIMALS = 6


@dataclass(frozen=True)
class WalkForwardResult:
    payload: dict[str, Any]
    passed: bool
    failure_reason: str | None


def walk_forward_validation(frame: pd.DataFrame, factor_def: FactorDefinition, duration: str) -> WalkForwardResult:
    metric_frame = _metric_frame(frame, factor_def, duration)
    returns = signal_returns(metric_frame, factor_def)
    windows = _window_returns(returns)
    payload = {name: _window_metrics(values) for name, values in windows.items()}
    failure_reason = _failure_reason(payload)
    return WalkForwardResult(payload, failure_reason is None, failure_reason)


def _metric_frame(frame: pd.DataFrame, factor_def: FactorDefinition, duration: str) -> pd.DataFrame:
    if "fwd_ret" in frame.columns:
        return frame
    return backtest_duration_frame(frame, factor_def.name, duration).dropna(subset=[factor_def.name, "fwd_ret"])


def _window_returns(returns: pd.Series) -> dict[str, pd.Series]:
    train_end = int(len(returns) * TRAIN_RATIO)
    validation_end = train_end + int(len(returns) * VALIDATION_RATIO)
    return {
        "train": returns.iloc[:train_end],
        "validation": returns.iloc[train_end:validation_end],
        "test": returns.iloc[validation_end:],
        "recent": returns.iloc[-RECENT_SAMPLE_COUNT:],
    }


def _window_metrics(returns: pd.Series) -> dict[str, Any]:
    if returns.empty:
        return {"sampleCount": 0, "winRate": None, "profitFactor": None, "avgReturn": None}
    return {
        "sampleCount": int(len(returns)),
        "winRate": round(float((returns > 0).mean()), METRIC_DECIMALS),
        "profitFactor": _round_or_none(_profit_factor(returns)),
        "avgReturn": _round_or_none(float(returns.mean())),
    }


def _failure_reason(payload: dict[str, dict[str, Any]]) -> str | None:
    for name in ("validation", "test"):
        reason = _window_failure_reason(name, payload[name])
        if reason is not None:
            return reason
    recent_win_rate = _finite_float(payload["recent"].get("winRate"))
    if recent_win_rate is not None and recent_win_rate < RECENT_MIN_WIN_RATE:
        return "recent_win_rate_weak"
    return None


def _window_failure_reason(name: str, metrics: dict[str, Any]) -> str | None:
    if int(metrics["sampleCount"]) < VALIDATION_MIN_SAMPLE_COUNT:
        return f"{name}_sample_count_below_min"
    win_rate = _finite_float(metrics.get("winRate"))
    if win_rate is None or win_rate < VALIDATION_MIN_WIN_RATE:
        return f"{name}_win_rate_below_min"
    profit_factor = _finite_float(metrics.get("profitFactor"))
    if profit_factor is None or profit_factor <= VALIDATION_MIN_PROFIT_FACTOR:
        return f"{name}_profit_factor_below_min"
    avg_return = _finite_float(metrics.get("avgReturn"))
    if avg_return is None or avg_return <= VALIDATION_MIN_AVG_RETURN:
        return f"{name}_avg_return_below_min"
    return None


def _profit_factor(returns: pd.Series) -> float:
    gains = float(returns[returns > 0].sum())
    losses = abs(float(returns[returns < 0].sum()))
    if gains <= 0:
        return 0.0
    return float("inf") if losses == 0 else gains / losses


def _round_or_none(value: float | None) -> float | None:
    if value is None or not isfinite(float(value)):
        return value
    return round(float(value), METRIC_DECIMALS)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None
