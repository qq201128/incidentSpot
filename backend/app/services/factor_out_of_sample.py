from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.services.factor_performance_metrics import signal_returns
from app.services.factor_registry import FactorDefinition
from app.services.factor_research_metrics import (
    compute_quintile_returns,
    compute_rolling_ic,
    compute_turnover,
    ic_metrics,
)
from app.services.return_metric_policy import ReturnMetricPolicy, ValidationGate, rounded_metrics
from app.services.trading_costs import roundtrip_cost_rate

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
MIN_OOS_SAMPLE_COUNT = 30
MIN_OOS_WIN_RATE = 0.50
MIN_OOS_PROFIT_FACTOR = 1.0
MIN_OOS_AVG_RETURN = 0.0


@dataclass(frozen=True)
class FactorOosWindow:
    name: str
    start: int
    end: int
    frame: pd.DataFrame


def factor_out_of_sample_report(df: pd.DataFrame, factor_def: FactorDefinition) -> dict[str, Any]:
    windows = _split_windows(df)
    payload = {window.name: _window_payload(window, factor_def) for window in windows}
    gate = _selection_gate(payload)
    return {
        **payload,
        "selectionGate": gate,
        "splitPolicy": "chronological_train_validation_test_no_shuffle",
        "selectionMetricSource": "validation_and_test_only",
    }


def _split_windows(df: pd.DataFrame) -> list[FactorOosWindow]:
    train_end = int(len(df) * TRAIN_RATIO)
    validation_end = train_end + int(len(df) * VALIDATION_RATIO)
    return [
        FactorOosWindow("train", 0, train_end, df.iloc[:train_end]),
        FactorOosWindow("validation", train_end, validation_end, df.iloc[train_end:validation_end]),
        FactorOosWindow("test", validation_end, len(df), df.iloc[validation_end:]),
    ]


def _window_payload(window: FactorOosWindow, factor_def: FactorDefinition) -> dict[str, Any]:
    metrics = _return_metrics(window.frame, factor_def)
    research = _research_payload(window.frame, factor_def.name)
    return {
        "start": window.start,
        "end": window.end,
        "sampleCount": int(len(window.frame)),
        "returnMetrics": metrics,
        "researchMetrics": research,
    }


def _return_metrics(df: pd.DataFrame, factor_def: FactorDefinition) -> dict[str, Any]:
    policy = ReturnMetricPolicy(cost_rate=roundtrip_cost_rate())
    metrics = policy.from_returns(signal_returns(df, factor_def))
    return rounded_metrics(metrics)


def _research_payload(df: pd.DataFrame, factor_name: str) -> dict[str, Any]:
    ic = ic_metrics(compute_rolling_ic(df[factor_name], df["fwd_ret"]))
    return {
        "icMean": _round_or_none(ic["mean"], 6),
        "icStd": _round_or_none(ic["std"], 6),
        "ir": _round_or_none(ic["ir"], 4),
        "icPositiveRate": _round_or_none(ic["positive_rate"], 4),
        "quintileReturns": [round(value, 6) for value in compute_quintile_returns(df, factor_name)],
        "turnover": _round_or_none(compute_turnover(df, factor_name), 4),
    }


def _selection_gate(payload: dict[str, Any]) -> dict[str, Any]:
    gate = ValidationGate(MIN_OOS_SAMPLE_COUNT, MIN_OOS_WIN_RATE, MIN_OOS_PROFIT_FACTOR, MIN_OOS_AVG_RETURN)
    failures = [
        reason
        for name in ("validation", "test")
        if (reason := gate.failure_reason(payload[name]["returnMetrics"], prefix=name)) is not None
    ]
    return {
        "status": "passed" if not failures else "failed",
        "reason": None if not failures else failures[0],
        "criteria": {
            "minSampleCount": MIN_OOS_SAMPLE_COUNT,
            "minWinRateExclusive": MIN_OOS_WIN_RATE,
            "minProfitFactorExclusive": MIN_OOS_PROFIT_FACTOR,
            "minAvgReturnExclusive": MIN_OOS_AVG_RETURN,
        },
    }


def _round_or_none(value: float | None, decimals: int) -> float | None:
    return None if value is None else round(float(value), decimals)
