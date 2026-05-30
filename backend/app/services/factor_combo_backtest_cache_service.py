from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.cache_payloads import decode_cache_payload
from app.services.factor_cache_metadata import cache_is_usable, cache_status, ranking_cache_metadata


@dataclass(frozen=True, slots=True)
class ComboBacktestCacheWrite:
    symbol: str
    duration: str
    factor_name: str
    metrics: dict[str, Any]

def get_cached_combo_backtest(symbol: str, duration: str, factor_name: str) -> dict[str, Any] | None:
    sym = symbol.strip().upper()
    row = _cache_row(sym, duration, factor_name)
    if row is None:
        return None
    payload = decode_cache_payload(
        row["payload"],
        cache_name="factor_combo_backtest_cache",
        identity={"symbol": sym, "duration": duration, "factorName": str(factor_name)},
    )
    if not isinstance(payload, dict):
        return None
    cache_meta = payload.get("cacheMeta")
    return {
        **payload,
        "updatedAt": str(row["updated_at"]),
        "cacheStatus": cache_status(cache_meta, sym),
    }


def get_usable_combo_backtest(symbol: str, duration: str, factor_name: str) -> dict[str, Any] | None:
    cached = get_cached_combo_backtest(symbol, duration, factor_name)
    if not cache_is_usable(cached):
        return None
    metrics = cached.get("metrics")
    return dict(metrics) if isinstance(metrics, dict) else None


def save_cached_combo_backtest(record: ComboBacktestCacheWrite) -> None:
    sym = record.symbol.strip().upper()
    duration = str(record.duration)
    name = str(record.factor_name)
    payload = json.dumps(
        {
            "symbol": sym,
            "duration": duration,
            "factorName": name,
            "metrics": record.metrics,
            "cacheMeta": ranking_cache_metadata(sym, duration),
        },
        ensure_ascii=False,
    )
    conn = get_conn()
    try:
        _ensure_table(conn)
        conn.execute(
            """
            INSERT INTO factor_combo_backtest_cache(symbol, duration, factor_name, updated_at, payload)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(symbol, duration, factor_name) DO UPDATE SET
              updated_at = excluded.updated_at,
              payload = excluded.payload
            """,
            (sym, duration, name, _utc_now(), payload),
        )
        conn.commit()
    finally:
        conn.close()


def _cache_row(symbol: str, duration: str, factor_name: str) -> Any | None:
    conn = get_conn()
    try:
        _ensure_table(conn)
        return conn.execute(
            """
            SELECT payload, updated_at
            FROM factor_combo_backtest_cache
            WHERE symbol = ? AND duration = ? AND factor_name = ?
            """,
            (symbol.strip().upper(), duration, str(factor_name)),
        ).fetchone()
    finally:
        conn.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_table(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS factor_combo_backtest_cache (
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          factor_name TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          payload TEXT NOT NULL,
          PRIMARY KEY (symbol, duration, factor_name)
        )
        """
    )
