from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.auto_predict_service import subscribe, unsubscribe
from app.services.strategy_registry import DEFAULT_STRATEGY_KEY

router = APIRouter(prefix="/ws", tags=["stream"])

@router.websocket("/predictions")
async def ws_predictions(
    websocket: WebSocket,
    symbol: str = "btcusdt",
    duration: str = "10m",
    strategyKey: str = DEFAULT_STRATEGY_KEY,
) -> None:
    await websocket.accept()
    subscribe(websocket, symbol, duration, strategyKey)
    try:
        while True:
            msg = await websocket.receive_text()
            if msg.lower() in {"ping", "keepalive"}:
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe(websocket, symbol, duration, strategyKey)
