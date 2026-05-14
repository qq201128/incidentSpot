from __future__ import annotations

from typing import Any

from app.db.session import get_conn

CACHE_SCHEMA_VERSION = 2
CACHE_STATUS_USABLE = "usable"
CACHE_STATUS_STALE = "stale"
CACHE_STATUS_LEGACY = "legacy_without_fingerprint"
CACHE_STATUS_MARKET_CHANGED = "market_data_changed"
CACHE_STATUS_NO_MARKET_DATA = "market_data_missing"


def ranking_cache_metadata(symbol: str, duration: str) -> dict[str, Any]:
    return {
        "schemaVersion": CACHE_SCHEMA_VERSION,
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "marketData": market_data_fingerprint(symbol),
    }


def market_data_fingerprint(symbol: str) -> dict[str, int | None]:
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS row_count, MAX(open_time) AS max_open_time
            FROM klines
            WHERE symbol = ? AND interval = '1m'
            """,
            (symbol.strip().upper(),),
        ).fetchone()
    finally:
        conn.close()
    return {
        "rowCount": int(row["row_count"] if row else 0),
        "maxOpenTime": None if row is None or row["max_open_time"] is None else int(row["max_open_time"]),
    }


def cache_status(cache_meta: dict[str, Any] | None, symbol: str) -> dict[str, Any]:
    if not isinstance(cache_meta, dict):
        return _status(False, CACHE_STATUS_LEGACY, None, market_data_fingerprint(symbol))
    if int(cache_meta.get("schemaVersion") or 0) != CACHE_SCHEMA_VERSION:
        return _status(False, CACHE_STATUS_LEGACY, cache_meta.get("marketData"), market_data_fingerprint(symbol))
    cached = _market_data(cache_meta)
    current = market_data_fingerprint(symbol)
    if current["maxOpenTime"] is None:
        return _status(False, CACHE_STATUS_NO_MARKET_DATA, cached, current)
    if cached != current:
        return _status(False, CACHE_STATUS_MARKET_CHANGED, cached, current)
    return _status(True, CACHE_STATUS_USABLE, cached, current)


def cache_is_usable(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    status = payload.get("cacheStatus")
    return not isinstance(status, dict) or bool(status.get("usable"))


def assert_cache_usable(payload: dict[str, Any], label: str) -> None:
    if cache_is_usable(payload):
        return
    status = payload.get("cacheStatus") or {}
    reason = status.get("reason") or CACHE_STATUS_STALE
    raise ValueError(f"{label} cache is stale: {reason}")


def _market_data(cache_meta: dict[str, Any]) -> dict[str, int | None] | None:
    value = cache_meta.get("marketData")
    if not isinstance(value, dict):
        return None
    return {
        "rowCount": int(value.get("rowCount") or 0),
        "maxOpenTime": None if value.get("maxOpenTime") is None else int(value["maxOpenTime"]),
    }


def _status(
    usable: bool,
    reason: str,
    cached: Any,
    current: dict[str, int | None],
) -> dict[str, Any]:
    return {
        "usable": usable,
        "state": CACHE_STATUS_USABLE if usable else CACHE_STATUS_STALE,
        "reason": reason,
        "cachedMarketData": cached,
        "currentMarketData": current,
    }
