from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd

from app.services.factor_learning_common import SUCCESS_PROFIT_FACTOR_MIN
from app.services.factor_performance_metrics import BACKTEST_MIN_PERIODS
from app.services.high_winrate_combo_goal_search import ComboHit, TARGET_WIN_RATE
from app.services.trading_costs import roundtrip_cost_rate

RECOMPUTED_MIN_WIN_RATE = TARGET_WIN_RATE
OOS_MIN_WIN_RATE = 0.70
MIN_OOS_TRADES = 20
WINDOW_COUNT = 3
OOS_RATIO = 0.20
TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
METRIC_DECIMALS = 4
AVG_RETURN_DECIMALS = 8


@dataclass(frozen=True)
class GoalComboValidation:
    passed: list[ComboHit]
    payload: dict[str, Any]


def validate_goal_combo_hits(frame: pd.DataFrame, hits: list[ComboHit], duration: str) -> GoalComboValidation:
    records = [_validation_record(frame, hit, duration) for hit in hits]
    passed = [hit for hit, record in zip(hits, records) if record["status"] == "passed"]
    return GoalComboValidation(passed, _gate_payload(records))


def _validation_record(frame: pd.DataFrame, hit: ComboHit, duration: str) -> dict[str, Any]:
    returns = _threshold_returns(frame, hit)
    recomputed = _recomputed_payload(returns)
    reasons = _rejection_reasons(recomputed)
    return {
        "duration": duration,
        "factorName": _factor_name(hit),
        "members": list(hit.members),
        "threshold": hit.threshold,
        "status": "passed" if not reasons else "rejected",
        "reasons": reasons,
        "reported": _reported_metrics(hit),
        "recomputed": recomputed,
    }


def _gate_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    rejected = [record for record in records if record["status"] == "rejected"]
    passed = [record for record in records if record["status"] == "passed"]
    return {
        "version": "high_winrate_combo_validation_gate_v1",
        "status": _gate_status(records, rejected),
        "failureReason": _gate_failure_reason(records, rejected),
        "thresholds": _threshold_payload(),
        "evaluatedCount": len(records),
        "passedCount": len(passed),
        "rejectedCount": len(rejected),
        "passed": [_summary_record(record) for record in passed],
        "rejections": [_summary_record(record) for record in rejected],
    }


def _threshold_returns(frame: pd.DataFrame, hit: ComboHit) -> pd.Series:
    score = hit.score.reindex(frame.index)
    signal = pd.Series(np.where(score >= hit.threshold, 1.0, np.where(score <= -hit.threshold, -1.0, np.nan)))
    returns = signal.set_axis(frame.index) * frame["fwd_ret"].astype(float) - roundtrip_cost_rate()
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def _recomputed_payload(returns: pd.Series) -> dict[str, Any]:
    segments = _nested_segments(returns)
    return {
        "full": _metrics(returns),
        "oos": _metrics(_oos_returns(returns)),
        "nested": {name: _metrics(values) for name, values in segments.items()},
        "windows": [
            {**_metrics(window), "window": index}
            for index, window in enumerate(_window_returns(returns), start=1)
        ],
    }


def _rejection_reasons(recomputed: dict[str, Any]) -> list[str]:
    reasons = _full_reasons(recomputed["full"])
    reasons.extend(_scoped_reasons("oos", recomputed["oos"], require_trades=True))
    reasons.extend(_nested_reasons(recomputed["nested"]))
    for row in recomputed["windows"]:
        reasons.extend(_scoped_reasons(f"window {row['window']}", row, require_trades=False))
    return reasons


def _full_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons = []
    if int(metrics["trades"]) < BACKTEST_MIN_PERIODS:
        reasons.append(f"full recompute: trades {metrics['trades']} < {BACKTEST_MIN_PERIODS}")
    reasons.extend(_scoped_reasons("full recompute", metrics, require_trades=False))
    return reasons


def _scoped_reasons(scope: str, metrics: dict[str, Any], *, require_trades: bool) -> list[str]:
    reasons = []
    if require_trades and int(metrics["trades"]) < MIN_OOS_TRADES:
        reasons.append(f"{scope}: trades {metrics['trades']} < {MIN_OOS_TRADES}")
    if float(metrics["winRate"]) < _min_win_rate_for_scope(scope):
        reasons.append(f"{scope}: winRate {metrics['winRate']} < {_min_win_rate_for_scope(scope)}")
    if float(metrics["profitFactor"]) < SUCCESS_PROFIT_FACTOR_MIN:
        reasons.append(f"{scope}: profitFactor {metrics['profitFactor']} < {SUCCESS_PROFIT_FACTOR_MIN}")
    if float(metrics["avgReturn"]) <= 0.0:
        reasons.append(f"{scope}: avgReturn {metrics['avgReturn']} <= 0")
    return reasons


def _nested_reasons(nested: dict[str, dict[str, Any]]) -> list[str]:
    reasons = []
    reasons.extend(_scoped_reasons("train", nested["train"], require_trades=True))
    reasons.extend(_scoped_reasons("validation", nested["validation"], require_trades=True))
    reasons.extend(_scoped_reasons("test", nested["test"], require_trades=True))
    return reasons


def _metrics(returns: pd.Series) -> dict[str, Any]:
    if returns.empty:
        return {"trades": 0, "winRate": 0.0, "profitFactor": 0.0, "avgReturn": 0.0}
    return {
        "trades": int(len(returns)),
        "winRate": round(float((returns > 0).mean()), METRIC_DECIMALS),
        "profitFactor": _round_metric(_profit_factor(returns), METRIC_DECIMALS),
        "avgReturn": _round_metric(float(returns.mean()), AVG_RETURN_DECIMALS),
    }


def _window_returns(returns: pd.Series) -> list[pd.Series]:
    return [pd.Series(chunk) for chunk in np.array_split(returns.to_numpy(), WINDOW_COUNT)]


def _oos_returns(returns: pd.Series) -> pd.Series:
    count = max(1, int(len(returns) * OOS_RATIO))
    return returns.iloc[-count:]


def _nested_segments(returns: pd.Series) -> dict[str, pd.Series]:
    total = len(returns)
    train_end = int(total * TRAIN_RATIO)
    validation_end = train_end + int(total * VALIDATION_RATIO)
    return {
        "train": returns.iloc[:train_end],
        "validation": returns.iloc[train_end:validation_end],
        "test": returns.iloc[validation_end:],
    }


def _min_win_rate_for_scope(scope: str) -> float:
    if scope in {"oos", "validation", "test"} or scope.startswith("window "):
        return OOS_MIN_WIN_RATE
    return RECOMPUTED_MIN_WIN_RATE


def _profit_factor(returns: pd.Series) -> float:
    gains = float(returns[returns > 0].sum())
    losses = abs(float(returns[returns < 0].sum()))
    if gains <= 0.0:
        return 0.0
    return float("inf") if losses == 0.0 else gains / losses


def _round_metric(value: float, decimals: int) -> float:
    return value if not isfinite(value) else round(float(value), decimals)


def _reported_metrics(hit: ComboHit) -> dict[str, Any]:
    return {
        "winRate": round(hit.win_rate, METRIC_DECIMALS),
        "profitFactor": _round_metric(hit.profit_factor, METRIC_DECIMALS),
        "trades": hit.trades,
        "avgReturn": _round_metric(hit.avg_return, AVG_RETURN_DECIMALS),
    }


def _summary_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "factorName": record["factorName"],
        "duration": record["duration"],
        "members": record["members"],
        "threshold": record["threshold"],
        "status": record["status"],
        "reasons": record["reasons"],
        "reported": record["reported"],
        "recomputed": record["recomputed"],
    }


def _threshold_payload() -> dict[str, Any]:
    return {
        "reportedMinTrades": BACKTEST_MIN_PERIODS,
        "recomputedMinWinRate": RECOMPUTED_MIN_WIN_RATE,
        "oosMinWinRate": OOS_MIN_WIN_RATE,
        "minProfitFactor": SUCCESS_PROFIT_FACTOR_MIN,
        "minOosTrades": MIN_OOS_TRADES,
        "windowCount": WINDOW_COUNT,
        "nestedSplit": {
            "trainRatio": TRAIN_RATIO,
            "validationRatio": VALIDATION_RATIO,
            "testRatio": round(1.0 - TRAIN_RATIO - VALIDATION_RATIO, 4),
        },
    }


def _gate_status(records: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> str:
    if not records:
        return "not_applicable"
    if len(rejected) == len(records):
        return "failed"
    return "partial" if rejected else "passed"


def _gate_failure_reason(records: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> str | None:
    if not records:
        return None
    return "all_combos_rejected_by_validation" if len(rejected) == len(records) else None


def _factor_name(hit: ComboHit) -> str:
    return "goal_combo__" + "__".join(hit.members)
