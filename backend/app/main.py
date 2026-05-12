from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.api.auto_trade import router as auto_trade_router
from app.api.events import router as events_router
from app.api.factors import router as factors_router
from app.api.market import router as market_router
from app.api.rules import router as rules_router
from app.api.stream import router as stream_router
from app.config.env_file import load_backend_env_file
from app.db.session import init_db
from app.services.auto_predict_service import auto_predict_loop
from app.services.auto_settlement_service import auto_settlement_loop
from app.services.auto_trade_service import auto_trade_loop
from app.services.ws_service import proxy_index_kline_stream, proxy_kline_stream

load_backend_env_file()


def _cors_allow_origins() -> list[str]:
    """本地开发默认端口；部署时在环境变量 CORS_ORIGINS 追加公网前端来源，逗号分隔。"""
    origins = ["http://127.0.0.1:5173", "http://localhost:5173"]
    extra = os.getenv("CORS_ORIGINS", "").strip()
    if not extra:
        return origins
    for part in extra.split(","):
        o = part.strip()
        if o:
            origins.append(o)
    return origins


app = FastAPI(title="Incident Spot Backend")
app.state.settlement_task = None
app.state.settlement_stop_event = None
app.state.predict_task = None
app.state.predict_stop_event = None
ALLOWED_INTERVALS = {"10m", "30m", "60m", "1h", "4h", "1d"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market_router)
app.include_router(events_router)
app.include_router(stream_router)
app.include_router(auto_trade_router)
app.include_router(rules_router)
app.include_router(factors_router)


@app.on_event("startup")
async def on_startup() -> None:
    init_db()

    settlement_stop = asyncio.Event()
    app.state.settlement_stop_event = settlement_stop
    app.state.settlement_task = asyncio.create_task(auto_settlement_loop(settlement_stop))

    predict_stop = asyncio.Event()
    app.state.predict_stop_event = predict_stop
    app.state.predict_task = asyncio.create_task(auto_predict_loop(predict_stop))

    trade_stop = asyncio.Event()
    app.state.trade_stop_event = trade_stop
    app.state.trade_task = asyncio.create_task(auto_trade_loop(trade_stop))


@app.on_event("shutdown")
async def on_shutdown() -> None:
    for attr in ("settlement_stop_event", "predict_stop_event", "trade_stop_event"):
        ev = getattr(app.state, attr, None)
        if ev:
            ev.set()
    for attr in ("settlement_task", "predict_task", "trade_task"):
        task = getattr(app.state, attr, None)
        if task:
            await task


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.websocket("/ws/klines")
async def ws_klines(websocket: WebSocket, symbol: str = "btcusdt", interval: str = "30m") -> None:
    if interval not in ALLOWED_INTERVALS:
        await websocket.accept()
        await websocket.close(code=1008, reason=f"unsupported interval: {interval}")
        return

    try:
        await proxy_kline_stream(websocket, symbol, interval)
    except Exception:
        # Connection might already be closed by client.
        pass


@app.websocket("/ws/index-klines")
async def ws_index_klines(websocket: WebSocket, symbol: str = "btcusdt", interval: str = "30m") -> None:
    if interval not in ALLOWED_INTERVALS:
        await websocket.accept()
        await websocket.close(code=1008, reason=f"unsupported interval: {interval}")
        return

    try:
        await proxy_index_kline_stream(websocket, symbol, interval)
    except Exception:
        pass
