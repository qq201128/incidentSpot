from __future__ import annotations

from pydantic import BaseModel, Field


class EventCreate(BaseModel):
    strategyKey: str | None = None
    symbol: str = Field(min_length=6)
    title: str
    eventInterval: str = "30m"
    ruleType: str = "ABOVE"
    strikeValue: float
    upperBound: float | None = None
    endTime: str
    aiProbabilityUp: float | None = None
    aiPredictedDirection: str | None = None
    aiQualityScore: float | None = None
    aiQualityPassed: bool | None = None
    aiHighWinrateGate: str | None = None
    aiHighWinrateRule: str | None = None
    aiHighWinratePassed: bool | None = None
    aiHighWinrateValue: float | None = None


class OrderCreate(BaseModel):
    side: str
    qty: float = Field(gt=0)
    price: float = Field(ge=0)


class QuickTradeCreate(BaseModel):
    event: EventCreate
    order: OrderCreate
    liveTradingEnabled: bool = False
