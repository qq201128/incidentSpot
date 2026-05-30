from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.strategy_registry import DEFAULT_STRATEGY_KEY


@dataclass(frozen=True)
class SubscriberKey:
    symbol: str
    duration: str
    strategy_key: str = DEFAULT_STRATEGY_KEY

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.symbol.upper(), self.duration, self.strategy_key)


@dataclass(frozen=True)
class PredictionSubscription:
    ws: Any
    key: SubscriberKey


def subscribe(
    subscribers: dict[tuple[str, str, str], set],
    ws: Any,
    key: SubscriberKey,
) -> None:
    subscribers.setdefault(key.as_tuple(), set()).add(ws)


def subscribe_prediction_ws(
    subscribers: dict[tuple[str, str, str], set],
    subscription: PredictionSubscription,
) -> None:
    subscribe(subscribers, subscription.ws, subscription.key)


def unsubscribe(
    subscribers: dict[tuple[str, str, str], set],
    ws: Any,
    key: SubscriberKey,
) -> None:
    active = subscribers.get(key.as_tuple())
    if active:
        active.discard(ws)


def unsubscribe_prediction_ws(
    subscribers: dict[tuple[str, str, str], set],
    subscription: PredictionSubscription,
) -> None:
    unsubscribe(subscribers, subscription.ws, subscription.key)
