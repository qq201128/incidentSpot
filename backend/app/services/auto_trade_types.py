from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AutoTradeSettings:
    strategy_key: str
    enabled: bool
    symbol: str
    duration: str
    duration_minutes: int
    qty: float
    live_trading_enabled: bool

    def to_response(self) -> dict[str, Any]:
        return {
            "strategyKey": self.strategy_key,
            "enabled": self.enabled,
            "symbol": self.symbol,
            "duration": self.duration,
            "durationMinutes": self.duration_minutes,
            "qty": self.qty,
            "liveTradingEnabled": self.live_trading_enabled,
        }


@dataclass(frozen=True)
class AutoTradeEventPayload:
    strategyKey: str | None
    symbol: str
    title: str
    eventInterval: str
    ruleType: str
    strikeValue: float
    upperBound: float | None
    endTime: str
    aiProbabilityUp: float
    aiPredictedDirection: str
    aiQualityScore: float | None
    aiQualityPassed: bool | None
    aiHighWinrateGate: str | None
    aiHighWinrateRule: str | None
    aiHighWinratePassed: bool | None
    aiHighWinrateValue: float | None
    predictionId: int | None = None


@dataclass(frozen=True)
class AutoTradeOrderPayload:
    side: str
    qty: float
    price: float


@dataclass(frozen=True)
class AutoTradePayload:
    event: AutoTradeEventPayload
    order: AutoTradeOrderPayload
