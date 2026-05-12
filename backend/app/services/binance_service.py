from __future__ import annotations

import time
from typing import Any

import requests
from requests.exceptions import RequestException

from app.services.index_price_tick_service import persist_index_price_tick
from app.services.orderbook_feature_service import OrderbookSnapshotRequest, build_orderbook_snapshot

FAPI_BASE_URL = "https://fapi.binance.com"

LAST_ORDERBOOK: dict[str, dict[str, Any]] = {}
LAST_TICKER: dict[str, dict[str, Any]] = {}
LAST_FUNDING_RATE: dict[str, float | None] = {}


def _retry_get(
    url: str,
    params: dict,
    *,
    max_attempts: int = 6,
    timeout: tuple = (10, 40),
) -> dict | list:
    """Generic retry wrapper for Binance GET requests."""
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except RequestException as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                break
            sleep_s = min(2 ** attempt, 20)
            time.sleep(sleep_s)
    raise last_error  # type: ignore[misc]


def _raw_klines_from_response(rows: list) -> list[dict]:
    klines: list[dict] = []
    for item in rows:
        klines.append(
            {
                "openTime": int(item[0]),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "closeTime": int(item[6]),
            }
        )
    return klines


def _aggregate_1m_klines(rows_1m: list[dict], bar_ms: int) -> list[dict]:
    """Merge consecutive 1m candles into higher timeframe bars (openTime aligned to bar_ms UTC)."""
    if not rows_1m:
        return []
    rows_1m = sorted(rows_1m, key=lambda r: r["openTime"])
    out: list[dict] = []
    cur: dict | None = None
    bucket: int | None = None
    for r in rows_1m:
        ot = int(r["openTime"])
        b = (ot // bar_ms) * bar_ms
        if bucket is None or b != bucket:
            if cur is not None:
                out.append(cur)
            bucket = b
            cur = {
                "openTime": b,
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r["volume"]),
                "closeTime": int(r["closeTime"]),
            }
        else:
            assert cur is not None
            cur["high"] = max(cur["high"], float(r["high"]))
            cur["low"] = min(cur["low"], float(r["low"]))
            cur["close"] = float(r["close"])
            cur["volume"] += float(r["volume"])
            cur["closeTime"] = int(r["closeTime"])
    if cur is not None:
        out.append(cur)
    return out


def _trim_leading_aggregate_if_first_bucket_incomplete(
    raw_rows_asc: list[dict], aggregated: list[dict], bar_ms: int
) -> list[dict]:
    """
    If the oldest 1m candle starts after the open of its aggregate bucket (common when
    ``limit`` pulls history mid-bucket), the first merged bar uses the wrong open —
    bull/bear streak logic diverges from exchange-grade OHLC. Drop that bar.
    """
    if not aggregated or not raw_rows_asc:
        return aggregated
    first_ot = int(raw_rows_asc[0]["openTime"])
    bucket_start = (first_ot // bar_ms) * bar_ms
    if first_ot > bucket_start:
        return aggregated[1:]
    return aggregated


def fetch_klines(
    symbol: str,
    interval: str,
    *,
    limit: int = 500,
    start_time: int | None = None,
    end_time: int | None = None,
) -> list[dict]:
    sym = symbol.upper()
    if interval == "10m":
        ten = 10 * 60 * 1000
        need = min(1500, max(limit, 50) * 10 + 20)
        params: dict[str, Any] = {"symbol": sym, "interval": "1m", "limit": need}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        rows = _retry_get(f"{FAPI_BASE_URL}/fapi/v1/klines", params)
        k1 = _raw_klines_from_response(rows)
        agg = _aggregate_1m_klines(k1, ten)
        agg = _trim_leading_aggregate_if_first_bucket_incomplete(k1, agg, ten)
        return agg[-limit:] if len(agg) > limit else agg

    params = {"symbol": sym, "interval": parse_interval(interval), "limit": limit}
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time

    rows = _retry_get(f"{FAPI_BASE_URL}/fapi/v1/klines", params)
    return _raw_klines_from_response(rows)


def fetch_index_price_klines(
    pair: str,
    interval: str,
    *,
    limit: int = 500,
    start_time: int | None = None,
    end_time: int | None = None,
    request_options: dict[str, Any] | None = None,
) -> list[dict]:
    """
    Index price OHLCV from GET /fapi/v1/indexPriceKlines (pair + interval).

    Used as the underlying index series for USD-M futures; aligns with mark/index
    context for event-style settlement references. Binance has no native 10m:
    we aggregate from 1m index klines.
    """
    pair_u = pair.upper()
    options = request_options or {}
    max_attempts = int(options.get("max_attempts", 6))
    timeout = options.get("timeout", (10, 40))
    if interval == "10m":
        ten = 10 * 60 * 1000
        need = min(1500, max(limit, 50) * 10 + 20)
        params: dict[str, Any] = {"pair": pair_u, "interval": "1m", "limit": need}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        rows = _retry_get(
            f"{FAPI_BASE_URL}/fapi/v1/indexPriceKlines",
            params,
            max_attempts=max_attempts,
            timeout=timeout,
        )
        k1 = _raw_klines_from_response(rows)
        agg = _aggregate_1m_klines(k1, ten)
        agg = _trim_leading_aggregate_if_first_bucket_incomplete(k1, agg, ten)
        return agg[-limit:] if len(agg) > limit else agg

    params = {"pair": pair_u, "interval": parse_interval(interval), "limit": limit}
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time
    rows = _retry_get(
        f"{FAPI_BASE_URL}/fapi/v1/indexPriceKlines",
        params,
        max_attempts=max_attempts,
        timeout=timeout,
    )
    return _raw_klines_from_response(rows)


def fetch_premium_index(symbol: str) -> dict:
    """Latest mark price and index price from GET /fapi/v1/premiumIndex (per symbol)."""
    data = _retry_get(
        f"{FAPI_BASE_URL}/fapi/v1/premiumIndex",
        {"symbol": symbol.upper()},
        timeout=(10, 20),
    )
    row: dict[str, Any]
    if isinstance(data, list):
        if not data:
            raise ValueError("empty premium index response")
        row = data[0]
    else:
        row = data
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
    """Fetch orderbook depth and compute OFI/microprice features."""
    sym = symbol.upper()
    params = {"symbol": sym, "limit": limit}
    data = _retry_get(f"{FAPI_BASE_URL}/fapi/v1/depth", params, timeout=(10, 20))
    return build_orderbook_snapshot(
        OrderbookSnapshotRequest(
            symbol=sym,
            bids=data.get("bids", []),
            asks=data.get("asks", []),
            cache=LAST_ORDERBOOK,
            quote_time=int(time.time() * 1000),
        )
    )


def _binance_depth_fetch_limit(levels: int) -> int:
    """Binance USD-M depth ``limit`` must be one of 5, 10, 20, 50, 100, 500, 1000."""
    for cap in (5, 10, 20, 50, 100, 500, 1000):
        if cap >= levels:
            return cap
    return 1000


def fetch_agg_trades_display(symbol: str, limit: int = 50) -> list[dict[str, Any]]:
    """Recent aggregate trades for UI (GET ``/fapi/v1/aggTrades``). Newest first."""
    sym = symbol.upper()
    limit = max(1, min(int(limit), 1000))
    params = {"symbol": sym, "limit": limit}
    rows = _retry_get(f"{FAPI_BASE_URL}/fapi/v1/aggTrades", params, timeout=(10, 20))
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for t in rows:
        price = float(t.get("p", 0) or 0)
        qty = float(t.get("q", 0) or 0)
        ts = int(t.get("T", 0) or 0)
        buyer_maker = bool(t.get("m", False))
        # Buyer was maker => aggressive sell => red/sell side in UI
        side = "sell" if buyer_maker else "buy"
        out.append(
            {
                "price": price,
                "qty": qty,
                "quoteQty": price * qty,
                "time": ts,
                "side": side,
            }
        )
    out.sort(key=lambda r: int(r["time"]), reverse=True)
    return out


def fetch_orderbook_depth_display(symbol: str, levels: int = 20) -> dict[str, Any]:
    """REST depth for UI: trimmed bid/ask ladders plus best quotes (updates LAST_ORDERBOOK cache)."""
    sym = symbol.upper()
    levels = max(5, min(int(levels), 1000))
    fetch_limit = _binance_depth_fetch_limit(levels)
    params = {"symbol": sym, "limit": fetch_limit}
    data = _retry_get(f"{FAPI_BASE_URL}/fapi/v1/depth", params, timeout=(10, 20))
    bids_raw = data.get("bids") or []
    asks_raw = data.get("asks") or []
    snapshot = build_orderbook_snapshot(
        OrderbookSnapshotRequest(
            symbol=sym,
            bids=bids_raw,
            asks=asks_raw,
            cache=LAST_ORDERBOOK,
            quote_time=int(time.time() * 1000),
        )
    )
    bids = [[float(b[0]), float(b[1])] for b in bids_raw[:levels]]
    asks = [[float(a[0]), float(a[1])] for a in asks_raw[:levels]]
    spread = float(snapshot["best_ask"]) - float(snapshot["best_bid"])
    mid = (float(snapshot["best_bid"]) + float(snapshot["best_ask"])) / 2.0
    spread_bps = (spread / mid * 10_000.0) if mid > 0 else 0.0
    return {
        "symbol": sym,
        "lastUpdateId": data.get("lastUpdateId"),
        "bids": bids,
        "asks": asks,
        "bestBid": snapshot["best_bid"],
        "bestAsk": snapshot["best_ask"],
        "spread": spread,
        "spreadBps": spread_bps,
        "timestamp": snapshot["timestamp"],
    }


def fetch_24h_ticker(symbol: str) -> dict:
    """Fetch 24h price statistics (volume, price change, weighted avg price, etc.)."""
    params = {"symbol": symbol.upper()}
    data = _retry_get(f"{FAPI_BASE_URL}/fapi/v1/ticker/24hr", params, timeout=(10, 20))
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
        "timestamp": int(time.time() * 1000),
    }
    LAST_TICKER[symbol.upper()] = result
    return result

def fetch_funding_rate(symbol: str) -> float | None:
    """Fetch latest funding rate for symbol."""
    params = {"symbol": symbol.upper(), "limit": 1}
    data = _retry_get(f"{FAPI_BASE_URL}/fapi/v1/fundingRate", params, timeout=(10, 20))
    if data and isinstance(data, list) and len(data) > 0:
        rate = float(data[0].get("fundingRate", 0))
        LAST_FUNDING_RATE[symbol.upper()] = rate
        return rate
    return LAST_FUNDING_RATE.get(symbol.upper())


def get_cached_orderbook(symbol: str) -> dict | None:
    return LAST_ORDERBOOK.get(symbol.upper())


def get_cached_ticker(symbol: str) -> dict | None:
    return LAST_TICKER.get(symbol.upper())


def get_cached_funding_rate(symbol: str) -> float | None:
    return LAST_FUNDING_RATE.get(symbol.upper())


def parse_interval(interval: str) -> str:
    mapping = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "1h",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "3d": "3d",
        "1w": "1w",
    }
    if interval not in mapping:
        raise ValueError(f"unsupported interval: {interval}")
    return mapping[interval]


def kline_ws_stream_name(interval: str) -> str:
    """Stream suffix for fstream URL, e.g. btcusdt@kline_30m. 10m uses underlying 1m."""
    if interval == "10m":
        return "kline_1m"
    return f"kline_{parse_interval(interval)}"
