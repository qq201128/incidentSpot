from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.services.auto_predict_service import _SUBSCRIBERS
from app.services.auto_predict_subscribers import (
    PredictionSubscription,
    SubscriberKey,
    subscribe_prediction_ws,
    unsubscribe_prediction_ws,
)
from app.services.strategy_registry import DEFAULT_STRATEGY_KEY

router = APIRouter(prefix="/ws", tags=["stream"])


@dataclass(frozen=True)
class PredictionWsParams:
    symbol: str
    duration: str
    strategy_key: str


def _prediction_ws_params(
    symbol: str = "btcusdt",
    duration: str = "10m",
    strategyKey: str = DEFAULT_STRATEGY_KEY,
) -> PredictionWsParams:
    return PredictionWsParams(symbol, duration, strategyKey)


@router.websocket("/predictions")
async def ws_predictions(
    websocket: WebSocket,
    params: PredictionWsParams = Depends(_prediction_ws_params),
) -> None:
    await websocket.accept()
    subscription = PredictionSubscription(
        websocket,
        SubscriberKey(params.symbol, params.duration, params.strategy_key),
    )
    subscribe_prediction_ws(_SUBSCRIBERS, subscription)
    try:
        while True:
            msg = await websocket.receive_text()
            if msg.lower() in {"ping", "keepalive"}:
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        return
    finally:
        unsubscribe_prediction_ws(_SUBSCRIBERS, subscription)
