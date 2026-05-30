from __future__ import annotations

from dataclasses import dataclass

from app.db.session import get_conn, run_db_write_with_retry
from app.services.binance_service import fetch_klines


@dataclass(frozen=True)
class BackfillState:
    current: int
    end_time: int | None
    rounds: int = 0


class KlineBackfillError(RuntimeError):
    pass


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
    Pull older 1m candles into SQLite until we reach target_rows.

    Returns final row count in DB for (symbol, 1m).
    """
    sym = symbol.upper()
    bounded_chunk = _bounded_chunk(chunk)
    state = _initial_state(sym)
    if state.current >= target_rows:
        return state.current

    while state.current < target_rows and state.rounds < max_rounds:
        rows = _fetch_backfill_rows(sym, bounded_chunk, state.end_time)
        if not rows:
            raise KlineBackfillError(f"no historical 1m klines returned for {sym} before {state.end_time}")

        upsert_klines_rows(sym, "1m", rows)
        state = _next_state(sym, state, rows)

    if state.current < target_rows:
        raise KlineBackfillError(
            f"1m kline backfill for {sym} stopped after {state.rounds} rounds: "
            f"{state.current}/{target_rows} rows"
        )
    return state.current


def _bounded_chunk(chunk: int) -> int:
    return min(1000, max(50, chunk))


def _initial_state(symbol: str) -> BackfillState:
    oldest = oldest_open_time(symbol, "1m")
    return BackfillState(current=count_klines(symbol, "1m"), end_time=int(oldest) - 1 if oldest is not None else None)


def _fetch_backfill_rows(symbol: str, chunk: int, end_time: int | None) -> list[dict]:
    try:
        return fetch_klines(symbol, "1m", limit=chunk, end_time=end_time)
    except Exception as exc:
        raise KlineBackfillError(
            f"failed to fetch 1m klines for {symbol} before {end_time} with chunk={chunk}: {exc}"
        ) from exc


def _next_state(symbol: str, state: BackfillState, rows: list[dict]) -> BackfillState:
    new_oldest = min(int(row["openTime"]) for row in rows)
    if state.end_time is not None and new_oldest >= state.end_time:
        raise KlineBackfillError(f"historical 1m kline backfill did not move earlier for {symbol}")
    return BackfillState(current=count_klines(symbol, "1m"), end_time=new_oldest - 1, rounds=state.rounds + 1)
