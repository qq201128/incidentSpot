from __future__ import annotations

import time
from typing import Any

from app.services.agg_trade_normalize import normalize_agg_trade_row
from app.services.binance_http import FAPI_BASE_URL, retry_get
from app.services.index_price_tick_service import persist_index_price_tick
from app.services.orderbook_feature_service import OrderbookSnapshotRequest, build_orderbook_snapshot

MS_PER_SECOND = 1000
BPS_SCALE = 10_000.0
MIN_DISPLAY_DEPTH = 5
MAX_DISPLAY_DEPTH = 1000
MIN_AGG_TRADE_LIMIT = 1
BINANCE_DEPTH_LIMITS = (5, 10, 20, 50, 100, 500, 1000)
DISPLAY_MAX_ATTEMPTS = 2
DISPLAY_TIMEOUT = (3, 6)

LAST_ORDERBOOK: dict[str, dict[str, Any]] = {}
LAST_TICKER: dict[str, dict[str, Any]] = {}
LAST_FUNDING_RATE: dict[str, float | None] = {}


def fetch_premium_index(symbol: str) -> dict:
    data = _display_retry_get("/fapi/v1/premiumIndex", {"symbol": symbol.upper()})
    row = _premium_index_row(data)
    result = {
        "symbol": row.get("symbol"),
        "markPrice": float(row.get("markPrice", 0) or 0),
        "indexPrice": float(row.get("indexPrice", 0) or 0),
        "lastFundingRate": float(row.get("lastFundingRate", 0) or 0),
        "nextFundingTime": int(row.get("nextFundingTime", 0) or 0),
        "time": int(row.get("time", 0) or 0),
    }
    persist_index_price_tick(result)
    return result


def fetch_orderbook(symbol: str, limit: int = 500) -> dict:
    sym = symbol.upper()
    params = {"symbol": sym, "limit": limit}
    data = _display_retry_get("/fapi/v1/depth", params)
    return build_orderbook_snapshot(
        OrderbookSnapshotRequest(
            symbol=sym,
            bids=data.get("bids", []),
            asks=data.get("asks", []),
            cache=LAST_ORDERBOOK,
            quote_time=_now_ms(),
        )
    )


def fetch_agg_trades_display(symbol: str, limit: int = 50) -> list[dict[str, Any]]:
    sym = symbol.upper()
    bounded_limit = max(MIN_AGG_TRADE_LIMIT, min(int(limit), MAX_DISPLAY_DEPTH))
    rows = retry_get(
        f"{FAPI_BASE_URL}/fapi/v1/aggTrades",
        {"symbol": sym, "limit": bounded_limit},
        max_attempts=DISPLAY_MAX_ATTEMPTS,
        timeout=DISPLAY_TIMEOUT,
    )
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized = normalize_agg_trade_row(row)
        if normalized is not None:
            out.append(normalized)
    out.sort(key=lambda row: int(row["time"]), reverse=True)
    return out


def fetch_orderbook_depth_display(symbol: str, levels: int = 20) -> dict[str, Any]:
    sym = symbol.upper()
    bounded_levels = max(MIN_DISPLAY_DEPTH, min(int(levels), MAX_DISPLAY_DEPTH))
    data = _display_retry_get(
        "/fapi/v1/depth",
        {"symbol": sym, "limit": _binance_depth_fetch_limit(bounded_levels)},
    )
    snapshot = _orderbook_snapshot(sym, data)
    bids = [[float(bid[0]), float(bid[1])] for bid in (data.get("bids") or [])[:bounded_levels]]
    asks = [[float(ask[0]), float(ask[1])] for ask in (data.get("asks") or [])[:bounded_levels]]
    spread = float(snapshot["best_ask"]) - float(snapshot["best_bid"])
    mid = (float(snapshot["best_bid"]) + float(snapshot["best_ask"])) / 2.0
    return {
        "symbol": sym,
        "lastUpdateId": data.get("lastUpdateId"),
        "bids": bids,
        "asks": asks,
        "bestBid": snapshot["best_bid"],
        "bestAsk": snapshot["best_ask"],
        "spread": spread,
        "spreadBps": (spread / mid * BPS_SCALE) if mid > 0 else 0.0,
        "timestamp": snapshot["timestamp"],
    }


def fetch_24h_ticker(symbol: str) -> dict:
    sym = symbol.upper()
    data = _display_retry_get("/fapi/v1/ticker/24hr", {"symbol": sym})
    result = {
        "symbol": data.get("symbol"),
        "priceChange": float(data.get("priceChange", 0)),
        "priceChangePercent": float(data.get("priceChangePercent", 0)),
        "weightedAvgPrice": float(data.get("weightedAvgPrice", 0)),
        "volume": float(data.get("volume", 0)),
        "quoteVolume": float(data.get("quoteVolume", 0)),
        "openPrice": float(data.get("openPrice", 0)),
        "highPrice": float(data.get("highPrice", 0)),
        "lowPrice": float(data.get("lowPrice", 0)),
        "lastPrice": float(data.get("lastPrice", 0)),
        "count": int(data.get("count", 0)),
        "timestamp": _now_ms(),
    }
    LAST_TICKER[sym] = result
    return result


def fetch_funding_rate(symbol: str) -> float | None:
    sym = symbol.upper()
    data = _display_retry_get("/fapi/v1/fundingRate", {"symbol": sym, "limit": 1})
    if data and isinstance(data, list):
        rate = float(data[0].get("fundingRate", 0))
        LAST_FUNDING_RATE[sym] = rate
        return rate
    return LAST_FUNDING_RATE.get(sym)


def get_cached_orderbook(symbol: str) -> dict | None:
    return LAST_ORDERBOOK.get(symbol.upper())


def get_cached_ticker(symbol: str) -> dict | None:
    return LAST_TICKER.get(symbol.upper())


def get_cached_funding_rate(symbol: str) -> float | None:
    return LAST_FUNDING_RATE.get(symbol.upper())


def _display_retry_get(endpoint: str, params: dict[str, Any]) -> dict | list:
    return retry_get(
        f"{FAPI_BASE_URL}{endpoint}",
        params,
        max_attempts=DISPLAY_MAX_ATTEMPTS,
        timeout=DISPLAY_TIMEOUT,
    )


def _premium_index_row(data: dict | list) -> dict[str, Any]:
    if isinstance(data, list):
        if not data:
            raise ValueError("empty premium index response")
        return data[0]
    return data


def _orderbook_snapshot(sym: str, data: dict[str, Any]) -> dict[str, Any]:
    return build_orderbook_snapshot(
        OrderbookSnapshotRequest(
            symbol=sym,
            bids=data.get("bids") or [],
            asks=data.get("asks") or [],
            cache=LAST_ORDERBOOK,
            quote_time=_now_ms(),
        )
    )


def _binance_depth_fetch_limit(levels: int) -> int:
    for cap in BINANCE_DEPTH_LIMITS:
        if cap >= levels:
            return cap
    return MAX_DISPLAY_DEPTH


def _now_ms() -> int:
    return int(time.time() * MS_PER_SECOND)
