from __future__ import annotations

import asyncio
import logging
import time

from fastapi import WebSocket, WebSocketDisconnect
from requests.exceptions import RequestException
from websockets.exceptions import ConnectionClosed

from app.services.binance_service import fetch_index_price_klines

logger = logging.getLogger(__name__)

REST_FALLBACK_TIMEOUT_SECONDS = 8
FALLBACK_KLINE_LIMIT = 2
FALLBACK_REQUEST_OPTIONS = {"max_attempts": 1, "timeout": (2, 4)}
NETWORK_ERROR_COOLDOWN_SECONDS = 60
_NEXT_REST_FALLBACK_AT: dict[tuple[str, str], float] = {}


async def send_index_rest_fallback(
    client_ws: WebSocket,
    symbol: str,
    interval: str,
    reason: str,
) -> None:
    if _is_in_network_cooldown(symbol, interval):
        return
    try:
        rows = await asyncio.wait_for(
            asyncio.to_thread(
                fetch_index_price_klines,
                symbol.upper(),
                interval,
                limit=FALLBACK_KLINE_LIMIT,
                request_options=FALLBACK_REQUEST_OPTIONS,
            ),
            timeout=REST_FALLBACK_TIMEOUT_SECONDS,
        )
        if rows:
            await client_ws.send_json(
                {"type": "kline", "data": rows[-1], "source": "rest-fallback"}
            )
            logger.info(
                "index kline REST fallback sent for %s@%s after %s",
                symbol.upper(),
                interval,
                reason,
            )
    except (WebSocketDisconnect, ConnectionClosed):
        raise
    except TimeoutError:
        _start_network_cooldown(symbol, interval)
        logger.warning(
            "index kline REST fallback timed out for %s@%s after %s",
            symbol.upper(),
            interval,
            reason,
        )
    except RequestException as exc:
        _start_network_cooldown(symbol, interval)
        logger.warning(
            "index kline REST fallback network error for %s@%s after %s; cooldown %ss: %s",
            symbol.upper(),
            interval,
            reason,
            NETWORK_ERROR_COOLDOWN_SECONDS,
            exc,
        )
    except Exception:
        logger.exception(
            "index kline REST fallback failed for %s@%s after %s",
            symbol.upper(),
            interval,
            reason,
        )


def _is_in_network_cooldown(symbol: str, interval: str) -> bool:
    key = (symbol.upper(), interval)
    remaining = _NEXT_REST_FALLBACK_AT.get(key, 0.0) - time.monotonic()
    if remaining <= 0:
        return False
    logger.info(
        "index kline REST fallback skipped for %s@%s; network cooldown %.0fs remaining",
        key[0],
        key[1],
        remaining,
    )
    return True


def _start_network_cooldown(symbol: str, interval: str) -> None:
    _NEXT_REST_FALLBACK_AT[(symbol.upper(), interval)] = (
        time.monotonic() + NETWORK_ERROR_COOLDOWN_SECONDS
    )
