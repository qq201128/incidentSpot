from __future__ import annotations

from typing import Any

from app.db.session import get_conn

CACHE_SCHEMA_VERSION = 2
CACHE_STATUS_USABLE = "usable"
CACHE_STATUS_STALE = "stale"
CACHE_STATUS_LEGACY = "legacy_without_fingerprint"
CACHE_STATUS_MARKET_CHANGED = "market_data_changed"
CACHE_STATUS_MARKET_APPENDED = "market_data_appended"
CACHE_STATUS_NO_MARKET_DATA = "market_data_missing"
BAR_ALIGNED_FEATURE_TABLES = ("funding_features", "orderbook_features")


def ranking_cache_metadata(symbol: str, duration: str) -> dict[str, Any]:
    return {
        "schemaVersion": CACHE_SCHEMA_VERSION,
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "marketData": market_data_fingerprint(symbol, duration),
        "barAlignedFeatures": bar_aligned_feature_fingerprint(symbol, duration),
    }


def market_data_fingerprint(symbol: str, duration: str) -> dict[str, int | None]:
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS row_count, MAX(open_time) AS max_open_time
            FROM klines
            WHERE symbol = ? AND interval = ?
            """,
            (symbol.strip().upper(), duration),
        ).fetchone()
    finally:
        conn.close()
    return {
        "rowCount": int(row["row_count"] if row else 0),
        "maxOpenTime": None if row is None or row["max_open_time"] is None else int(row["max_open_time"]),
    }


def bar_aligned_feature_fingerprint(symbol: str, duration: str) -> dict[str, dict[str, int | None]]:
    sym = symbol.strip().upper()
    conn = get_conn()
    try:
        return {
            table: _feature_table_fingerprint(conn, table=table, symbol=sym, duration=duration)
            for table in BAR_ALIGNED_FEATURE_TABLES
            if _table_exists(conn, table)
        }
    finally:
        conn.close()


def bar_aligned_features_match(cache_meta: dict[str, Any] | None, symbol: str, duration: str) -> bool:
    if not isinstance(cache_meta, dict):
        return False
    cached = cache_meta.get("barAlignedFeatures")
    if not isinstance(cached, dict):
        return False
    return cached == bar_aligned_feature_fingerprint(symbol, duration)


def _feature_table_fingerprint(
    conn: Any,
    *,
    table: str,
    symbol: str,
    duration: str,
) -> dict[str, int | None]:
    row = conn.execute(
        f"""
        SELECT COUNT(DISTINCT k.open_time) AS expected_count,
               COUNT(DISTINCT f.open_time) AS matched_count,
               MIN(f.open_time) AS min_open_time,
               MAX(f.open_time) AS max_open_time
        FROM klines k
        LEFT JOIN {table} f ON f.symbol = k.symbol AND f.open_time = k.open_time
        WHERE k.symbol = ? AND k.interval = ?
        """,
        (symbol, duration),
    ).fetchone()
    return {
        "expectedCount": int(row["expected_count"] if row else 0),
        "matchedCount": int(row["matched_count"] if row else 0),
        "minOpenTime": None if row is None or row["min_open_time"] is None else int(row["min_open_time"]),
        "maxOpenTime": None if row is None or row["max_open_time"] is None else int(row["max_open_time"]),
    }


def _table_exists(conn: Any, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def cache_status(cache_meta: dict[str, Any] | None, symbol: str) -> dict[str, Any]:
    duration = _cache_duration(cache_meta)
    if not isinstance(cache_meta, dict):
        return _status(False, CACHE_STATUS_LEGACY, None, market_data_fingerprint(symbol, duration))
    if int(cache_meta.get("schemaVersion") or 0) != CACHE_SCHEMA_VERSION:
        return _status(False, CACHE_STATUS_LEGACY, cache_meta.get("marketData"), market_data_fingerprint(symbol, duration))
    cached = _market_data(cache_meta)
    current = market_data_fingerprint(symbol, duration)
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


def cache_is_usable_for_live_signal(payload: dict[str, Any] | None) -> bool:
    return cache_is_usable(payload) or _market_data_only_appended(payload)


def assert_cache_usable(payload: dict[str, Any], label: str) -> None:
    if cache_is_usable(payload):
        return
    status = payload.get("cacheStatus") or {}
    reason = status.get("reason") or CACHE_STATUS_STALE
    raise ValueError(f"{label} cache is stale: {reason}")


def assert_cache_usable_for_live_signal(payload: dict[str, Any], label: str) -> None:
    if cache_is_usable_for_live_signal(payload):
        return
    status = payload.get("cacheStatus") or {}
    reason = status.get("reason") or CACHE_STATUS_STALE
    raise ValueError(f"{label} cache is stale: {reason}")


def live_signal_cache_reason(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return CACHE_STATUS_STALE
    status = payload.get("cacheStatus")
    if not isinstance(status, dict) or bool(status.get("usable")):
        return CACHE_STATUS_USABLE
    if _market_data_only_appended(payload):
        return CACHE_STATUS_MARKET_APPENDED
    return str(status.get("reason") or CACHE_STATUS_STALE)


def _market_data_only_appended(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    status = payload.get("cacheStatus")
    if not isinstance(status, dict):
        return False
    if status.get("reason") != CACHE_STATUS_MARKET_CHANGED:
        return False
    cached = _normalized_market_snapshot(status.get("cachedMarketData"))
    current = _normalized_market_snapshot(status.get("currentMarketData"))
    if cached is None or current is None:
        return False
    if cached["maxOpenTime"] is None or current["maxOpenTime"] is None:
        return False
    return (
        current["rowCount"] >= cached["rowCount"]
        and current["maxOpenTime"] >= cached["maxOpenTime"]
    )


def _market_data(cache_meta: dict[str, Any]) -> dict[str, int | None] | None:
    value = cache_meta.get("marketData")
    if not isinstance(value, dict):
        return None
    return {
        "rowCount": int(value.get("rowCount") or 0),
        "maxOpenTime": None if value.get("maxOpenTime") is None else int(value["maxOpenTime"]),
    }


def _normalized_market_snapshot(value: Any) -> dict[str, int | None] | None:
    if not isinstance(value, dict):
        return None
    return {
        "rowCount": int(value.get("rowCount") or 0),
        "maxOpenTime": None if value.get("maxOpenTime") is None else int(value["maxOpenTime"]),
    }


def _cache_duration(cache_meta: dict[str, Any] | None) -> str:
    if isinstance(cache_meta, dict) and cache_meta.get("duration"):
        return str(cache_meta["duration"])
    return "10m"


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
