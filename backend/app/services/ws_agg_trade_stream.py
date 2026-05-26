from __future__ import annotations

import asyncio
import logging
from asyncio import sleep
from dataclasses import dataclass, replace
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed
from websockets.legacy.client import connect as upstream_ws_connect

from app.services.agg_trade_normalize import normalize_agg_trade_row
from app.services.binance_market_data import fetch_agg_trades_display
from app.services.binance_upstream_connect import upstream_websocket_connect_kwargs
from app.services.ws_kline_transform import unwrap_fstream_ws_message

logger = logging.getLogger(__name__)

UPSTREAM_OPEN_TIMEOUT_SECONDS = 20
FSTREAM_MARKET_WS_BASE_URL = "wss://fstream.binance.com/market/ws"
AGG_TRADE_STREAM_SUFFIX = "aggTrade"
DEFAULT_AGG_TRADE_SNAPSHOT_LIMIT = 40


@dataclass(frozen=True)
class AggTradeStreamState:
    symbol: str
    stream_url: str
    retry_wait_seconds: int
    max_retry_wait_seconds: int


async def proxy_agg_trade_stream(
    client_ws: WebSocket,
    symbol: str,
    *,
    limit: int = DEFAULT_AGG_TRADE_SNAPSHOT_LIMIT,
) -> None:
    sym = symbol.upper()
    state = AggTradeStreamState(
        symbol=sym,
        stream_url=f"{FSTREAM_MARKET_WS_BASE_URL}/{sym.lower()}@{AGG_TRADE_STREAM_SUFFIX}",
        retry_wait_seconds=1,
        max_retry_wait_seconds=15,
    )
    await client_ws.accept()
    try:
        await send_agg_trade_snapshot(client_ws, sym, limit=limit)
    except (WebSocketDisconnect, ConnectionClosed):
        return
    await run_agg_trade_loop(client_ws, state)


async def run_agg_trade_loop(client_ws: WebSocket, state: AggTradeStreamState) -> None:
    while True:
        try:
            state = await run_agg_trade_stream(client_ws, state)
        except (WebSocketDisconnect, ConnectionClosed):
            break
        except TimeoutError:
            log_agg_trade_timeout(state)
            state = await agg_trade_backoff(state)
        except (ConnectionResetError, OSError) as exc:
            log_agg_trade_connection_error(state, exc)
            state = await agg_trade_backoff(state)
        except RuntimeError as exc:
            if "close message has been sent" in str(exc):
                break
            logger.exception("runtime error in agg trade stream")
            state = await agg_trade_backoff(state)
        except Exception:
            logger.exception("agg trade stream disconnected, retrying")
            state = await agg_trade_backoff(state)


async def send_agg_trade_snapshot(client_ws: WebSocket, symbol: str, *, limit: int) -> None:
    try:
        rows = await asyncio.to_thread(fetch_agg_trades_display, symbol, limit=limit)
    except Exception:
        logger.exception("agg trade REST snapshot failed for %s", symbol)
        return
    await client_ws.send_json({"type": "snapshot", "data": rows})


async def run_agg_trade_stream(client_ws: WebSocket, state: AggTradeStreamState) -> AggTradeStreamState:
    state = replace(state, retry_wait_seconds=1)
    connect_kw: dict[str, Any] = await asyncio.to_thread(upstream_websocket_connect_kwargs, state.stream_url)
    async with upstream_ws_connect(
        state.stream_url,
        ping_interval=20,
        ping_timeout=20,
        open_timeout=UPSTREAM_OPEN_TIMEOUT_SECONDS,
        **connect_kw,
    ) as ws:
        async for message in ws:
            await send_agg_trade_message(client_ws, message)
    return state


async def send_agg_trade_message(client_ws: WebSocket, message: str | bytes) -> None:
    payload = unwrap_fstream_ws_message(message)
    if payload.get("e") != "aggTrade":
        return
    trade = normalize_agg_trade_row(payload)
    if trade is None:
        return
    await client_ws.send_json({"type": "aggTrade", "data": trade})


async def agg_trade_backoff(state: AggTradeStreamState) -> AggTradeStreamState:
    await sleep(state.retry_wait_seconds)
    next_wait = min(state.retry_wait_seconds * 2, state.max_retry_wait_seconds)
    return replace(state, retry_wait_seconds=next_wait)


def log_agg_trade_timeout(state: AggTradeStreamState) -> None:
    logger.warning("agg trade upstream connect timed out for %s, retry in %ss", state.symbol, state.retry_wait_seconds)


def log_agg_trade_connection_error(state: AggTradeStreamState, exc: BaseException) -> None:
    logger.warning(
        "agg trade upstream connection error for %s (%s), retry in %ss",
        state.symbol,
        type(exc).__name__,
        state.retry_wait_seconds,
    )
