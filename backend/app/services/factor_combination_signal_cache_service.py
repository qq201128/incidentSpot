from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn

logger = logging.getLogger("uvicorn.error")


def get_cached_combination_signals(symbol: str) -> dict[str, Any] | None:
    row = _cache_row(symbol)
    if row is None:
        return None
    try:
        payload = json.loads(row["payload"])
    except json.JSONDecodeError:
        logger.warning("factor_combo_signal_cache corrupt JSON for %s", symbol)
        return None
    if not isinstance(payload, dict):
        return None
    return {**payload, "updatedAt": str(row["updated_at"]), "source": "signal_cache"}


def save_cached_combination_signals(payload: dict[str, Any]) -> None:
    symbol = str(payload["symbol"]).strip().upper()
    top_per_duration = int(payload.get("topPerDuration") or 0)
    limit_count = int(payload.get("limit") or payload.get("total") or 0)
    raw = json.dumps(payload, ensure_ascii=False)
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO factor_combo_signal_cache(symbol, updated_at, top_per_duration, limit_count, payload)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
              updated_at = excluded.updated_at,
              top_per_duration = excluded.top_per_duration,
              limit_count = excluded.limit_count,
              payload = excluded.payload
            """,
            (symbol, _utc_now(), top_per_duration, limit_count, raw),
        )
        conn.commit()
    finally:
        conn.close()


def _cache_row(symbol: str) -> Any | None:
    conn = get_conn()
    try:
        return conn.execute(
            """
            SELECT updated_at, top_per_duration, limit_count, payload
            FROM factor_combo_signal_cache
            WHERE symbol = ?
            """,
            (symbol.strip().upper(),),
        ).fetchone()
    finally:
        conn.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
