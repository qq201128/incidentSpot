from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.event_regime_status import market_regime_status

TREND_UP = "trend_up"
TREND_DOWN = "trend_down"
RANGE = "range"
UNCERTAIN = "uncertain"
UP = "up"
DOWN = "down"
MARKET_REGIME_TRADE_GATE_VERSION = "market_regime_trade_gate_v1"


@dataclass(frozen=True)
class MarketRegimeTradeDecision:
    allowed: bool
    reason: str
    mode: str
    regime: dict[str, Any]


def evaluate_market_regime_trade_gate(
    *,
    symbol: str,
    duration: str,
    open_time: int,
    direction: str,
) -> MarketRegimeTradeDecision:
    normalized_direction = _normalized_direction(direction)
    regime = market_regime_status(symbol, duration, int(open_time))
    if regime.get("ready") is not True:
        return _blocked("market_regime_not_ready", "skip", regime)
    trend = str(regime.get("trendState") or UNCERTAIN)
    if trend == TREND_UP:
        return _trend_decision(normalized_direction, UP, regime)
    if trend == TREND_DOWN:
        return _trend_decision(normalized_direction, DOWN, regime)
    if trend == RANGE:
        return MarketRegimeTradeDecision(True, "range_environment_allowed", "range", regime)
    return _blocked(f"market_regime_{trend}_skip", "skip", regime)


def _trend_decision(
    direction: str,
    required_direction: str,
    regime: dict[str, Any],
) -> MarketRegimeTradeDecision:
    if direction == required_direction:
        return MarketRegimeTradeDecision(True, f"trend_{required_direction}_aligned", "trend", regime)
    return _blocked(f"counter_trend_{direction}_vs_{required_direction}", "skip", regime)


def _normalized_direction(direction: str) -> str:
    value = str(direction).strip().lower()
    if value in {UP, DOWN}:
        return value
    if value in {"buy", "long"}:
        return UP
    if value in {"sell", "short"}:
        return DOWN
    raise ValueError(f"unsupported trade direction for market regime gate: {direction}")


def _blocked(reason: str, mode: str, regime: dict[str, Any]) -> MarketRegimeTradeDecision:
    return MarketRegimeTradeDecision(False, reason, mode, regime)
