from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.services.factor_performance_metrics import BACKTEST_MIN_PERIODS

THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 3.00
THRESHOLD_STEP = 0.10
SEARCH_CANDIDATE_LIMIT = 80


@dataclass(frozen=True)
class GoalSearchConfig:
    candidate_limit: int = SEARCH_CANDIDATE_LIMIT
    signal_thresholds: tuple[float, ...] = ()
    min_trades: int = BACKTEST_MIN_PERIODS


def signal_thresholds(
    min_value: float = THRESHOLD_MIN,
    max_value: float = THRESHOLD_MAX,
    step: float = THRESHOLD_STEP,
) -> tuple[float, ...]:
    start = Decimal(str(min_value))
    end = Decimal(str(max_value))
    delta = Decimal(str(step))
    if start <= 0 or end < start or delta <= 0:
        raise ValueError("threshold min/max/step must define a positive ascending range")
    values = []
    current = start
    while current <= end:
        values.append(float(current))
        current += delta
    return tuple(values)


SIGNAL_THRESHOLDS = signal_thresholds()


def validated_search_config(
    config: GoalSearchConfig | None = None,
    default_min_trades: int = BACKTEST_MIN_PERIODS,
) -> GoalSearchConfig:
    cfg = config or GoalSearchConfig(signal_thresholds=SIGNAL_THRESHOLDS, min_trades=default_min_trades)
    thresholds = cfg.signal_thresholds or SIGNAL_THRESHOLDS
    if cfg.candidate_limit <= 0 or cfg.min_trades <= 0:
        raise ValueError("candidate_limit and min_trades must be positive")
    if not thresholds or any(float(value) <= 0 for value in thresholds):
        raise ValueError("signal_thresholds must contain positive values")
    return GoalSearchConfig(
        candidate_limit=int(cfg.candidate_limit),
        signal_thresholds=tuple(float(value) for value in thresholds),
        min_trades=int(cfg.min_trades),
    )
