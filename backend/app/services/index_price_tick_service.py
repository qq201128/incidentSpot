from __future__ import annotations

import sqlite3
import time
from typing import Any

from app.db.session import get_conn


def persist_index_price_tick(row: dict[str, Any]) -> None:
    symbol = str(row.get("symbol") or "").upper()
    quote_time = int(row.get("time") or 0)
    index_price = float(row.get("indexPrice") or 0)
    mark_price = float(row.get("markPrice") or 0)
    if not symbol or quote_time <= 0 or index_price <= 0:
        raise ValueError("premium index response cannot be persisted")

    sql = """
            INSERT OR REPLACE INTO index_price_ticks(symbol, quote_time, index_price, mark_price)
            VALUES(?, ?, ?, ?)
            """
    params = (symbol, quote_time, index_price, mark_price)
    backoff_s = (0.0, 0.04, 0.1, 0.25)
    last: sqlite3.OperationalError | None = None
    for delay in backoff_s:
        if delay:
            time.sleep(delay)
        conn = get_conn()
        try:
            conn.execute(sql, params)
            conn.commit()
            return
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "locked" not in msg and "busy" not in msg:
                raise
            last = exc
        finally:
            conn.close()
    if last is not None:
        raise last


def nearest_index_price_tick(symbol: str, target_time_ms: int, max_drift_ms: int):
    conn = get_conn()
    try:
        return conn.execute(
            """
            SELECT quote_time, index_price
            FROM index_price_ticks
            WHERE symbol = ? AND quote_time BETWEEN ? AND ?
            ORDER BY ABS(quote_time - ?) ASC
            LIMIT 1
            """,
            (
                symbol.upper(),
                target_time_ms - max_drift_ms,
                target_time_ms + max_drift_ms,
                target_time_ms,
            ),
        ).fetchone()
    finally:
        conn.close()


def nearest_available_index_price_tick(symbol: str, target_time_ms: int):
    conn = get_conn()
    try:
        return conn.execute(
            """
            SELECT quote_time, index_price
            FROM index_price_ticks
            WHERE symbol = ?
            ORDER BY ABS(quote_time - ?) ASC
            LIMIT 1
            """,
            (symbol.upper(), target_time_ms),
        ).fetchone()
    finally:
        conn.close()
