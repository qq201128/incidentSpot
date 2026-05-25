from __future__ import annotations

import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.app_startup import bootstrap_application, shutdown_application
from app.config.env_file import load_backend_env_file

load_backend_env_file()
logger = logging.getLogger(__name__)

ALLOWED_INTERVALS = {"10m", "30m", "60m", "1h", "4h", "1d"}


def _cors_allow_origins() -> list[str]:
    """本地开发默认端口；部署时在环境变量 CORS_ORIGINS 追加公网前端来源，逗号分隔。"""
    import os

    origins = ["http://127.0.0.1:5173", "http://localhost:5173"]
    extra = os.getenv("CORS_ORIGINS", "").strip()
    if not extra:
        return origins
    for part in extra.split(","):
        origin = part.strip()
        if origin:
            origins.append(origin)
    return origins


app = FastAPI(title="Incident Spot Backend")
app.state.bootstrap_task = None
app.state.settlement_task = None
app.state.settlement_stop_event = None
app.state.predict_task = None
app.state.predict_stop_event = None
app.state.factor_ranking_task = None
app.state.factor_ranking_stop_event = None
app.state.market_context_task = None
app.state.market_context_stop_event = None
app.state.factor_combo_daily_task = None
app.state.factor_combo_daily_stop_event = None
app.state.lstm_candidate_retry_task = None
app.state.lstm_candidate_retry_stop_event = None
app.state.lstm_daily_review_task = None
app.state.lstm_daily_review_stop_event = None
app.state.combo_event_governance_task = None
app.state.combo_event_governance_stop_event = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    await bootstrap_application(app)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await shutdown_application(app)


@app.get("/health")
async def health() -> dict:
    bootstrap_task = getattr(app.state, "bootstrap_task", None)
    ready = bootstrap_task is None or bootstrap_task.done()
    return {"ok": True, "ready": ready}


@app.websocket("/ws/klines")
async def ws_klines(websocket: WebSocket, symbol: str = "btcusdt", interval: str = "30m") -> None:
    from app.services.ws_service import proxy_kline_stream

    if interval not in ALLOWED_INTERVALS:
        await websocket.accept()
        await websocket.close(code=1008, reason=f"unsupported interval: {interval}")
        return

    try:
        await proxy_kline_stream(websocket, symbol, interval)
    except Exception:
        logger.exception("kline websocket failed: symbol=%s interval=%s", symbol, interval)


@app.websocket("/ws/agg-trades")
async def ws_agg_trades(websocket: WebSocket, symbol: str = "btcusdt", limit: int = 40) -> None:
    from app.services.ws_service import proxy_agg_trade_stream

    bounded_limit = max(1, min(int(limit), 200))
    try:
        await proxy_agg_trade_stream(websocket, symbol, limit=bounded_limit)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("agg trade websocket failed: symbol=%s", symbol)


@app.websocket("/ws/index-klines")
async def ws_index_klines(websocket: WebSocket, symbol: str = "btcusdt", interval: str = "30m") -> None:
    from app.services.ws_service import proxy_index_kline_stream

    if interval not in ALLOWED_INTERVALS:
        await websocket.accept()
        await websocket.close(code=1008, reason=f"unsupported interval: {interval}")
        return

    try:
        await proxy_index_kline_stream(websocket, symbol, interval)
    except Exception:
        logger.exception("index kline websocket failed: symbol=%s interval=%s", symbol, interval)
