from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.factor_cache_metadata import cache_status, ranking_cache_metadata

logger = logging.getLogger("uvicorn.error")


def _norm_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def get_cached_ranking(symbol: str, duration: str) -> dict[str, Any] | None:
    """Return { ranking, total, updatedAt, cacheStatus } or None if no row."""
    sym = _norm_symbol(symbol)
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT payload, total, updated_at
            FROM factor_ranking_cache
            WHERE symbol = ? AND duration = ?
            """,
            (sym, duration),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        payload = json.loads(row["payload"])
    except json.JSONDecodeError:
        logger.warning("factor_ranking_cache corrupt JSON for %s %s", sym, duration)
        return None
    ranking, cache_meta = _ranking_payload(payload)
    if not isinstance(ranking, list):
        return None
    return {
        "ranking": ranking,
        "total": int(row["total"]),
        "updatedAt": str(row["updated_at"]),
        "cacheMeta": cache_meta,
        "cacheStatus": cache_status(cache_meta, sym),
    }


def save_cached_ranking(symbol: str, duration: str, ranking: list[dict[str, Any]]) -> None:
    sym = _norm_symbol(symbol)
    payload = json.dumps(
        {
            "ranking": ranking,
            "cacheMeta": ranking_cache_metadata(sym, duration),
        },
        ensure_ascii=False,
    )
    total = len(ranking)
    ts = datetime.now(timezone.utc).isoformat()
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO factor_ranking_cache(symbol, duration, updated_at, total, payload)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(symbol, duration) DO UPDATE SET
              updated_at = excluded.updated_at,
              total = excluded.total,
              payload = excluded.payload
            """,
            (sym, duration, ts, total, payload),
        )
        conn.commit()
    finally:
        conn.close()


def factor_ranking_precomputed_symbols() -> list[str]:
    raw = os.getenv("FACTOR_RANKING_SYMBOLS", "BTCUSDT").strip()
    if not raw:
        return ["BTCUSDT"]
    parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
    return parts or ["BTCUSDT"]


def _ranking_payload(payload: Any) -> tuple[Any, dict[str, Any] | None]:
    if isinstance(payload, list):
        return payload, None
    if not isinstance(payload, dict):
        return None, None
    return payload.get("ranking"), payload.get("cacheMeta")
