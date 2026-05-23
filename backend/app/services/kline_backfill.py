from __future__ import annotations

from app.db.session import get_conn, run_db_write_with_retry
from app.services.binance_service import fetch_klines


def upsert_klines_rows(symbol: str, interval: str, rows: list[dict]) -> None:
    def _upsert() -> None:
        conn = get_conn()
        try:
            for item in rows:
                conn.execute(
                    """
                    INSERT INTO klines(symbol, interval, open_time, open, high, low, close, volume, close_time)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, interval, open_time) DO UPDATE SET
                      open=excluded.open,
                      high=excluded.high,
                      low=excluded.low,
                      close=excluded.close,
                      volume=excluded.volume,
                      close_time=excluded.close_time
                    """,
                    (
                        symbol.upper(),
                        interval,
                        item["openTime"],
                        item["open"],
                        item["high"],
                        item["low"],
                        item["close"],
                        item["volume"],
                        item["closeTime"],
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    run_db_write_with_retry(_upsert)


def count_klines(symbol: str, interval: str) -> int:
    conn = get_conn()
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM klines WHERE symbol = ? AND interval = ?",
        (symbol.upper(), interval),
    ).fetchone()
    conn.close()
    return int(row["c"])


def oldest_open_time(symbol: str, interval: str) -> int | None:
    conn = get_conn()
    row = conn.execute(
        """
        SELECT MIN(open_time) AS min_open_time
        FROM klines WHERE symbol = ? AND interval = ?
        """,
        (symbol.upper(), interval),
    ).fetchone()
    conn.close()
    if not row or row["min_open_time"] is None:
        return None
    return int(row["min_open_time"])


def backfill_1m_history(
    symbol: str,
    target_rows: int,
    *,
    chunk: int = 1000,
    max_rounds: int = 200,
) -> int:
    """
    Pull older 1m candles into SQLite until we reach target_rows (best effort).

    Returns final row count in DB for (symbol, 1m).
    """
    sym = symbol.upper()
    if chunk > 1000:
        chunk = 1000
    if chunk < 50:
        chunk = 50

    current = count_klines(sym, "1m")
    if current >= target_rows:
        return current

    oldest = oldest_open_time(sym, "1m")
    end_time = int(oldest) - 1 if oldest is not None else None

    rounds = 0
    while current < target_rows and rounds < max_rounds:
        rounds += 1
        try:
            rows = fetch_klines(sym, "1m", limit=chunk, end_time=end_time)
        except Exception:
            # Network hiccups happen; shrink the chunk and retry within the same round budget.
            if chunk > 200:
                chunk = max(200, chunk // 2)
            continue

        if not rows:
            break

        upsert_klines_rows(sym, "1m", rows)
        new_oldest = min(int(r["openTime"]) for r in rows)
        if end_time is not None and new_oldest >= end_time:
            # Avoid infinite loops if the API keeps returning the same window.
            break
        end_time = new_oldest - 1

        current = count_klines(sym, "1m")

    return current
