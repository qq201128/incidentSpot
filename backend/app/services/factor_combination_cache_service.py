from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.factor_cache_metadata import cache_status, ranking_cache_metadata

logger = logging.getLogger("uvicorn.error")


def get_cached_combination_ranking(symbol: str, duration: str) -> dict[str, Any] | None:
    sym = symbol.strip().upper()
    row = _cache_row(symbol, duration)
    if row is None:
        return None
    try:
        payload = json.loads(row["payload"])
    except json.JSONDecodeError:
        logger.warning("factor_combo_ranking_cache corrupt JSON for %s %s", symbol, duration)
        return None
    if not isinstance(payload, dict):
        return None
    cache_meta = payload.get("cacheMeta")
    return {
        **payload,
        "updatedAt": str(row["updated_at"]),
        "cacheStatus": cache_status(cache_meta, sym),
    }


def save_cached_combination_ranking(report: dict[str, Any]) -> None:
    symbol = str(report["symbol"]).strip().upper()
    duration = str(report["duration"])
    ranking = report.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError("combination ranking report must contain a ranking list")
    persisted = {**report, "cacheMeta": ranking_cache_metadata(symbol, duration)}
    payload = json.dumps(persisted, ensure_ascii=False)
    config = json.dumps(persisted.get("searchConfig") or {}, ensure_ascii=False)
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO factor_combo_ranking_cache(symbol, duration, updated_at, total, search_config, payload)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, duration) DO UPDATE SET
              updated_at = excluded.updated_at,
              total = excluded.total,
              search_config = excluded.search_config,
              payload = excluded.payload
            """,
            (symbol, duration, _utc_now(), len(ranking), config, payload),
        )
        conn.commit()
    finally:
        conn.close()


def _cache_row(symbol: str, duration: str) -> Any | None:
    conn = get_conn()
    try:
        return conn.execute(
            """
            SELECT payload, total, updated_at
            FROM factor_combo_ranking_cache
            WHERE symbol = ? AND duration = ?
            """,
            (symbol.strip().upper(), duration),
        ).fetchone()
    finally:
        conn.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
