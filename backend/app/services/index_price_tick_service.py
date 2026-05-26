from __future__ import annotations

from typing import Any

from app.db.session import get_conn, run_db_write_with_retry


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

    def _persist() -> None:
        conn = get_conn()
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    run_db_write_with_retry(_persist)


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
