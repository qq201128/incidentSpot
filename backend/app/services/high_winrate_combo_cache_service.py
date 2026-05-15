from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.factor_cache_metadata import cache_status, ranking_cache_metadata

logger = logging.getLogger("uvicorn.error")


def get_cached_high_winrate_combo_ranking(symbol: str, duration: str) -> dict[str, Any] | None:
    sym = symbol.strip().upper()
    row = _cache_row(sym, duration)
    if row is None:
        return None
    try:
        payload = json.loads(row["payload"])
    except json.JSONDecodeError:
        logger.warning("high_winrate_combo_ranking_cache corrupt JSON for %s %s", sym, duration)
        return None
    if not isinstance(payload, dict):
        return None
    cache_meta = payload.get("cacheMeta")
    return {
        **payload,
        "updatedAt": str(row["updated_at"]),
        "cacheStatus": cache_status(cache_meta, sym),
    }


def save_cached_high_winrate_combo_ranking(report: dict[str, Any]) -> None:
    symbol = str(report["symbol"]).strip().upper()
    duration = str(report["duration"])
    ranking = report.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError("high-winrate combo report must contain a ranking list")
    persisted = {**report, "cacheMeta": ranking_cache_metadata(symbol, duration)}
    payload = json.dumps(persisted, ensure_ascii=False)
    config = json.dumps(persisted.get("target") or {}, ensure_ascii=False)
    conn = get_conn()
    try:
        _ensure_table(conn)
        conn.execute(
            """
            INSERT INTO high_winrate_combo_ranking_cache(symbol, duration, updated_at, total, search_config, payload)
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
        _ensure_table(conn)
        return conn.execute(
            """
            SELECT payload, total, updated_at
            FROM high_winrate_combo_ranking_cache
            WHERE symbol = ? AND duration = ?
            """,
            (symbol.strip().upper(), duration),
        ).fetchone()
    finally:
        conn.close()


def _ensure_table(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS high_winrate_combo_ranking_cache (
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          total INTEGER NOT NULL,
          search_config TEXT NOT NULL,
          payload TEXT NOT NULL,
          PRIMARY KEY (symbol, duration)
        )
        """
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
