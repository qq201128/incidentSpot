from __future__ import annotations

import time
from typing import Any

from app.services.binance_service import FAPI_BASE_URL, LAST_ORDERBOOK, _retry_get
from app.services.orderbook_feature_service import OrderbookSnapshotRequest, build_orderbook_snapshot


MIN_DEPTH_LEVELS = 1


def fetch_orderbook_depth_levels(
    symbol: str,
    levels: int,
    *,
    require_full_depth: bool = True,
) -> dict[str, Any]:
    requested = _validated_levels(levels)
    sym = symbol.upper()
    data = _request_depth(sym, requested)
    bids = _levels(data.get("bids") or [], "bids", levels=requested, require_full_depth=require_full_depth)
    asks = _levels(data.get("asks") or [], "asks", levels=requested, require_full_depth=require_full_depth)
    snapshot = _snapshot(sym, bids, asks)
    return {
        "symbol": sym,
        "lastUpdateId": data.get("lastUpdateId"),
        "bids": bids,
        "asks": asks,
        "timestamp": snapshot["timestamp"],
        "bestBid": snapshot["best_bid"],
        "bestAsk": snapshot["best_ask"],
    }


def _validated_levels(levels: int) -> int:
    requested = int(levels)
    if requested < MIN_DEPTH_LEVELS:
        raise ValueError(f"orderbook levels must be >= {MIN_DEPTH_LEVELS}, got {levels}")
    return requested


def _request_depth(symbol: str, levels: int) -> dict[str, Any]:
    data = _retry_get(
        f"{FAPI_BASE_URL}/fapi/v1/depth",
        {"symbol": symbol, "limit": levels},
        timeout=(10, 20),
    )
    if not isinstance(data, dict):
        raise ValueError(f"orderbook depth response is not an object for {symbol}")
    return data


def _levels(
    raw_levels: list[Any],
    side: str,
    *,
    levels: int,
    require_full_depth: bool,
) -> list[list[str]]:
    if require_full_depth and len(raw_levels) < levels:
        raise ValueError(f"orderbook {side} returned {len(raw_levels)} levels, expected {levels}")
    return [list(level) for level in raw_levels[:levels]]


def _snapshot(symbol: str, bids: list[list[str]], asks: list[list[str]]) -> dict[str, Any]:
    return build_orderbook_snapshot(
        OrderbookSnapshotRequest(
            symbol=symbol,
            bids=bids,
            asks=asks,
            cache=LAST_ORDERBOOK,
            quote_time=int(time.time() * 1000),
        )
    )
