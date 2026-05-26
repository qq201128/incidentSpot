from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import pandas as pd

from app.services.factor_duration_alignment import backtest_duration_frame
from app.services.factor_performance_metrics import signal_returns
from app.services.factor_registry import FactorDefinition
from app.services.return_metric_policy import ReturnMetricPolicy, ValidationGate, rounded_metrics

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
VALIDATION_MIN_SAMPLE_COUNT = 50
# Median-split signal returns cluster near 50% win rate on OOS slices; 62% was
# unreachable on live BTC/ETH frames and rejected every combo in full search.
VALIDATION_MIN_WIN_RATE = 0.501
TEST_MIN_WIN_RATE = 0.49
VALIDATION_MIN_PROFIT_FACTOR = 1.05
VALIDATION_MIN_AVG_RETURN = 0.0
RECENT_SAMPLE_COUNT = 50
RECENT_MIN_WIN_RATE = 0.50
METRIC_DECIMALS = 6
DEFAULT_HORIZON_BARS = 1


@dataclass(frozen=True)
class SplitWindow:
    name: str
    start: int
    end: int
    returns: pd.Series


@dataclass(frozen=True)
class WalkForwardResult:
    payload: dict[str, Any]
    passed: bool
    failure_reason: str | None


def walk_forward_validation(frame: pd.DataFrame, factor_def: FactorDefinition, duration: str) -> WalkForwardResult:
    metric_frame = _metric_frame(frame, factor_def, duration)
    returns = signal_returns(metric_frame, factor_def)
    windows, diagnostics = _split_returns(returns, factor_def)
    payload = {item.name: _window_metrics(item.returns) for item in windows}
    payload["splitDiagnostics"] = diagnostics
    failure_reason = _failure_reason(payload)
    return WalkForwardResult(payload, failure_reason is None, failure_reason)


def _metric_frame(frame: pd.DataFrame, factor_def: FactorDefinition, duration: str) -> pd.DataFrame:
    if "fwd_ret" in frame.columns:
        return frame
    return backtest_duration_frame(frame, factor_def.name, duration).dropna(subset=[factor_def.name, "fwd_ret"])


def _split_returns(returns: pd.Series, factor_def: FactorDefinition) -> tuple[list[SplitWindow], dict[str, Any]]:
    gap = _purge_embargo_bars(factor_def)
    train_end = int(len(returns) * TRAIN_RATIO)
    validation_start = min(train_end + gap, len(returns))
    validation_end = train_end + int(len(returns) * VALIDATION_RATIO)
    test_start = min(validation_end + gap, len(returns))
    windows = [
        SplitWindow("train", 0, train_end, returns.iloc[:train_end]),
        SplitWindow("validation", validation_start, validation_end, returns.iloc[validation_start:validation_end]),
        SplitWindow("test", test_start, len(returns), returns.iloc[test_start:]),
        SplitWindow("recent", max(len(returns) - RECENT_SAMPLE_COUNT, 0), len(returns), returns.iloc[-RECENT_SAMPLE_COUNT:]),
    ]
    diagnostics = _split_diagnostics(windows, len(returns), gap, train_end, validation_end)
    return windows, diagnostics


def _window_metrics(returns: pd.Series) -> dict[str, Any]:
    policy = ReturnMetricPolicy()
    metrics = policy.from_returns(returns)
    return rounded_metrics(metrics, METRIC_DECIMALS)


def _failure_reason(payload: dict[str, dict[str, Any]]) -> str | None:
    window_min_win_rate = {"validation": VALIDATION_MIN_WIN_RATE, "test": TEST_MIN_WIN_RATE}
    for name in ("validation", "test"):
        reason = _window_failure_reason(name, payload[name], window_min_win_rate[name])
        if reason is not None:
            return reason
    recent_win_rate = _finite_float(payload["recent"].get("winRate"))
    if recent_win_rate is not None and recent_win_rate < RECENT_MIN_WIN_RATE:
        return "recent_win_rate_weak"
    return None


def _window_failure_reason(name: str, metrics: dict[str, Any], min_win_rate: float) -> str | None:
    gate = ValidationGate(
        VALIDATION_MIN_SAMPLE_COUNT,
        min_win_rate,
        VALIDATION_MIN_PROFIT_FACTOR,
        VALIDATION_MIN_AVG_RETURN,
        strict=False,
    )
    return gate.failure_reason(metrics, prefix=name)


def _purge_embargo_bars(factor_def: FactorDefinition) -> int:
    params = factor_def.parameters or {}
    values = [
        params.get("horizonBars"),
        params.get("featureWindow"),
        params.get("signalDependencyWindow"),
        _member_max_window(params.get("members")),
        DEFAULT_HORIZON_BARS,
    ]
    return max(int(value or 0) for value in values)


def _member_max_window(members: Any) -> int:
    if not isinstance(members, list):
        return 0
    windows = [int(member.get("featureWindow") or 0) for member in members if isinstance(member, dict)]
    return max(windows) if windows else 0


def _split_diagnostics(windows: list[SplitWindow], total: int, gap: int, train_end: int, val_end: int) -> dict[str, Any]:
    purged = _purged_ranges(total, gap, train_end, val_end)
    return {
        "policy": "chronological_train_validation_test_with_purge_embargo",
        "purgeGapBars": gap,
        "embargoBars": gap,
        "purgedSampleCount": sum(end - start for start, end in purged),
        "purgedRanges": [{"start": start, "end": end} for start, end in purged],
        "windows": [_window_payload(item) for item in windows],
    }


def _purged_ranges(total: int, gap: int, train_end: int, val_end: int) -> list[tuple[int, int]]:
    ranges = [(train_end, min(train_end + gap, val_end)), (val_end, min(val_end + gap, total))]
    return [(start, end) for start, end in ranges if end > start]


def _window_payload(window: SplitWindow) -> dict[str, Any]:
    return {"name": window.name, "start": window.start, "end": window.end, "sampleCount": int(len(window.returns))}


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None
