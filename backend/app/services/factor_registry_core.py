from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.services.rule_config import SUPPORTED_RULE_DURATIONS

# 与规则 / 因子排名回测周期一致（特征多在 1m 序列上构造，此处表示「适用规则周期」）
_RULE_UI_TIME_ORDER = ("10m", "30m", "60m", "1d")
RULE_FACTOR_TIMEFRAMES: tuple[str, ...] = tuple(
    d for d in _RULE_UI_TIME_ORDER if d in SUPPORTED_RULE_DURATIONS
)


def _timeframes_with_rule_align(*feature_bar_intervals: str) -> tuple[str, ...]:
    """先列规则周期，再追加特征所用 K 线周期（去重）。"""
    ordered: list[str] = []
    seen: set[str] = set()
    for x in (*RULE_FACTOR_TIMEFRAMES, *feature_bar_intervals):
        if x in seen:
            continue
        seen.add(x)
        ordered.append(x)
    return tuple(ordered)


class FactorCategory(str, Enum):
    RETURN = "return"
    VOLATILITY = "volatility"
    MOVING_AVERAGE = "moving_average"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    STRUCTURE = "structure"
    MULTI_TIMEFRAME = "multi_timeframe"
    ORDERBOOK = "orderbook"
    FUNDING = "funding"
    POSITIONING = "positioning"
    TAKER_FLOW = "taker_flow"
    SMC = "smc"
    SENTIMENT = "sentiment"
    STATISTIC = "statistic"
    ONCHAIN = "onchain"
    PERFORMANCE = "performance"


class FactorDirection(str, Enum):
    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    category: FactorCategory
    description: str
    formula: str
    source_file: str = "kline_features.py"
    timeframes: tuple[str, ...] = RULE_FACTOR_TIMEFRAMES
    direction: FactorDirection = FactorDirection.NEUTRAL
    parameters: dict[str, Any] = field(default_factory=dict)


