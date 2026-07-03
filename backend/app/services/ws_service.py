from __future__ import annotations

import logging
from asyncio import sleep
from typing import Any
from dataclasses import dataclass, replace

from fastapi import WebSocket
from websockets.legacy.client import connect as upstream_ws_connect

from app.db.session import get_conn
from app.services.binance_service import (
    fetch_klines,
    kline_ws_stream_name,
)
from app.services.index_kline_fallback import send_index_rest_fallback
from app.services.ws_client_disconnect import CLIENT_WS_GONE_EXC
from app.services.binance_upstream_connect import upstream_websocket_connect_kwargs
from app.services.ws_kline_transform import (
    candle_for_interval,
    candle_from_index_price_event,
    candle_from_k_obj,
    unwrap_fstream_ws_message,
)
from app.services.ws_agg_trade_stream import DEFAULT_AGG_TRADE_SNAPSHOT_LIMIT, proxy_agg_trade_stream
from app.services.ws_connection_manager import (
    WebSocketManager,
    ConnectionConfig,
    ConnectionState,
)
import asyncio

logger = logging.getLogger(__name__)

UPSTREAM_OPEN_TIMEOUT_SECONDS = 20
FSTREAM_MARKET_WS_BASE_URL = "wss://fstream.binance.com/market/ws"
KLINE_STREAM_KIND = "kline"
INDEX_PRICE_STREAM_KIND = "index_price_tick"
INDEX_PRICE_STREAM_NAME = "markPrice@1s"

# 全局WebSocket管理器（可选：用于管理上游连接池）
_ws_manager: WebSocketManager | None = None


def get_ws_manager() -> WebSocketManager:
    """获取全局WebSocket管理器"""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager


@dataclass(frozen=True)
class KlineStreamState:
    symbol: str
    interval: str
    stream_url: str
    retry_wait_seconds: int
    max_retry_wait_seconds: int
    stream_kind: str = KLINE_STREAM_KIND
    synthetic_state: dict | None = None
    last_closed_open_time: int | None = None
    reconnect_count: int = 0  # 新增：重连次数统计


async def proxy_kline_stream(client_ws: WebSocket, symbol: str, interval: str) -> None:
    """
    K线流代理（增强版）

    改进：
    1. 更好的重连日志
    2. 重连次数统计
    3. 重连次数过多时告警
    """
    state = _contract_stream_state(symbol, interval.strip())
    await client_ws.accept()

    while True:
        try:
            await _backfill_contract_state(state)
            state = await _run_kline_stream(client_ws, state, persist=True)
        except CLIENT_WS_GONE_EXC:
            logger.info(f"Client disconnected from kline stream: {symbol}@{interval}")
            break
        except TimeoutError:
            _log_stream_timeout("kline", state)
            state = await _backoff_with_tracking(state, "timeout")
        except (ConnectionResetError, OSError) as exc:
            _log_stream_connection_error("kline", state, exc)
            state = await _backoff_with_tracking(state, f"connection_error:{type(exc).__name__}")
        except RuntimeError as exc:
            if "close message has been sent" in str(exc):
                logger.info(f"Kline stream closed normally: {symbol}@{interval}")
                break
            logger.exception(f"Runtime error in kline stream: {symbol}@{interval}")
            state = await _backoff_with_tracking(state, "runtime_error")
        except Exception as exc:
            logger.exception(f"Kline stream unexpected error: {symbol}@{interval}")
            state = await _backoff_with_tracking(state, f"unexpected_error:{type(exc).__name__}")


async def proxy_index_kline_stream(client_ws: WebSocket, symbol: str, interval: str) -> None:
    """
    指数K线流代理（增强版）

    改进：
    1. 更好的重连日志
    2. REST降级通知
    3. 重连次数统计
    """
    state = _index_stream_state(symbol, interval.strip())
    await client_ws.accept()

    while True:
        try:
            state = await _run_kline_stream(client_ws, state, persist=False)
        except CLIENT_WS_GONE_EXC:
            logger.info(f"Client disconnected from index kline stream: {symbol}@{interval}")
            break
        except TimeoutError:
            _log_stream_timeout("index kline", state)
            await send_index_rest_fallback(client_ws, state.symbol, state.interval, "connect timeout")
            state = await _backoff_with_tracking(state, "timeout")
        except (ConnectionResetError, OSError) as exc:
            _log_stream_connection_error("index kline", state, exc)
            await send_index_rest_fallback(client_ws, state.symbol, state.interval, type(exc).__name__)
            state = await _backoff_with_tracking(state, f"connection_error:{type(exc).__name__}")
        except RuntimeError as exc:
            if "close message has been sent" in str(exc):
                logger.info(f"Index kline stream closed normally: {symbol}@{interval}")
                break
            logger.exception(f"Runtime error in index kline stream: {symbol}@{interval}")
            state = await _backoff_with_tracking(state, "runtime_error")
        except Exception as exc:
            logger.exception(f"Index kline stream unexpected error: {symbol}@{interval}")
            state = await _backoff_with_tracking(state, f"unexpected_error:{type(exc).__name__}")


def _contract_stream_state(symbol: str, interval: str) -> KlineStreamState:
    stream_url = f"{FSTREAM_MARKET_WS_BASE_URL}/{symbol.lower()}@{kline_ws_stream_name(interval)}"
    return KlineStreamState(
        symbol=symbol.upper(),
        interval=interval,
        stream_url=stream_url,
        retry_wait_seconds=1,
        max_retry_wait_seconds=15,
        last_closed_open_time=get_last_closed_open_time(symbol.upper(), interval),
    )


def _index_stream_state(symbol: str, interval: str) -> KlineStreamState:
    stream_url = f"{FSTREAM_MARKET_WS_BASE_URL}/{symbol.lower()}@{INDEX_PRICE_STREAM_NAME}"
    return KlineStreamState(
        symbol=symbol.upper(),
        interval=interval,
        stream_url=stream_url,
        retry_wait_seconds=1,
        max_retry_wait_seconds=15,
        stream_kind=INDEX_PRICE_STREAM_KIND,
    )


async def _backfill_contract_state(state: KlineStreamState) -> None:
    await asyncio.to_thread(
        backfill_closed_klines,
        state.symbol,
        state.interval,
        state.last_closed_open_time,
    )


async def _run_kline_stream(
    client_ws: WebSocket,
    state: KlineStreamState,
    *,
    persist: bool,
) -> KlineStreamState:
    state = _reset_stream_runtime_state(state)
    connect_kw: dict[str, Any] = await asyncio.to_thread(
        upstream_websocket_connect_kwargs, state.stream_url
    )
    async with upstream_ws_connect(
        state.stream_url,
        ping_interval=20,
        ping_timeout=20,
        open_timeout=UPSTREAM_OPEN_TIMEOUT_SECONDS,
        **connect_kw,
    ) as ws:
        async for message in ws:
            state = await _send_kline_message(client_ws, message, state, persist=persist)
    return state


def _reset_stream_runtime_state(state: KlineStreamState) -> KlineStreamState:
    if state.stream_kind == INDEX_PRICE_STREAM_KIND:
        return replace(state, retry_wait_seconds=1)
    return replace(state, retry_wait_seconds=1, synthetic_state=None)


async def _send_kline_message(
    client_ws: WebSocket,
    message: str | bytes,
    state: KlineStreamState,
    *,
    persist: bool,
) -> KlineStreamState:
    payload = unwrap_fstream_ws_message(message)
    candle, synthetic_state = _candle_from_stream_payload(payload, state)
    if candle is None:
        return state
    next_state = replace(state, synthetic_state=synthetic_state)
    if candle["isClosed"] and persist:
        save_closed_kline(state.symbol, state.interval, candle)
        next_state = replace(next_state, last_closed_open_time=candle["openTime"])
    await client_ws.send_json({"type": "kline", "data": candle})
    return next_state


def _candle_from_stream_payload(
    payload: dict[str, Any],
    state: KlineStreamState,
) -> tuple[dict | None, dict | None]:
    if state.stream_kind == INDEX_PRICE_STREAM_KIND:
        return candle_from_index_price_event(payload, state.interval, state.synthetic_state)
    k = payload.get("k")
    if not isinstance(k, dict):
        return None, state.synthetic_state
    c1 = candle_from_k_obj(k)
    if c1 is None:
        return None, state.synthetic_state
    return candle_for_interval(c1, state.interval, state.synthetic_state)


async def _backoff(state: KlineStreamState) -> KlineStreamState:
    """原始退避逻辑（保持向后兼容）"""
    await sleep(state.retry_wait_seconds)
    next_wait = min(state.retry_wait_seconds * 2, state.max_retry_wait_seconds)
    return replace(state, retry_wait_seconds=next_wait)


async def _backoff_with_tracking(state: KlineStreamState, reason: str) -> KlineStreamState:
    """
    带统计的退避逻辑

    Args:
        state: 当前状态
        reason: 重连原因

    Returns:
        更新后的状态（包含重连次数统计）
    """
    reconnect_count = state.reconnect_count + 1

    # 告警阈值
    if reconnect_count % 10 == 0:
        logger.warning(
            f"WebSocket reconnection threshold reached: "
            f"{state.symbol}@{state.interval} reconnected {reconnect_count} times "
            f"(reason: {reason})"
        )

    await sleep(state.retry_wait_seconds)
    next_wait = min(state.retry_wait_seconds * 2, state.max_retry_wait_seconds)

    return replace(
        state,
        retry_wait_seconds=next_wait,
        reconnect_count=reconnect_count,
    )


def _log_stream_timeout(kind: str, state: KlineStreamState) -> None:
    logger.warning(
        "%s upstream connect timed out for %s@%s, retry in %ss (reconnect_count=%d)",
        kind,
        state.symbol,
        state.interval,
        state.retry_wait_seconds,
        state.reconnect_count,
    )


def _log_stream_connection_error(kind: str, state: KlineStreamState, exc: BaseException) -> None:
    logger.warning(
        "%s upstream connection error for %s@%s: %s, retry in %ss (reconnect_count=%d)",
        kind,
        state.symbol,
        state.interval,
        type(exc).__name__,
        state.retry_wait_seconds,
        state.reconnect_count,
    )


def save_closed_kline(symbol: str, interval: str, candle: dict) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO klines(symbol, interval, open_time, open, high, low, close, volume, close_time)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, interval, open_time) DO UPDATE SET
          open=excluded.open,
          high=excluded.high,
          low=excluded.low,
          close=excluded.close,
          volume=excluded.volume,
          close_time=excluded.close_time
        """,
        (
            symbol,
            interval,
            candle["openTime"],
            candle["open"],
            candle["high"],
            candle["low"],
            candle["close"],
            candle["volume"],
            candle["closeTime"],
        ),
    )
    conn.commit()
    conn.close()


def get_last_closed_open_time(symbol: str, interval: str) -> int | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT MAX(open_time) AS max_open_time FROM klines WHERE symbol = ? AND interval = ?",
        (symbol, interval),
    ).fetchone()
    conn.close()
    return int(row["max_open_time"]) if row and row["max_open_time"] is not None else None


def backfill_closed_klines(symbol: str, interval: str, last_closed_open_time: int | None) -> None:
    rows = fetch_klines(symbol, interval, limit=500)
    for candle in rows:
        if last_closed_open_time is None or candle["openTime"] > last_closed_open_time:
            save_closed_kline(symbol, interval, candle)
