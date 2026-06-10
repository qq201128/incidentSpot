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
    return backfill_interval_history(
        symbol,
        "1m",
        target_rows=target_rows,
        chunk=chunk,
        max_rounds=max_rounds,
    )


def backfill_interval_history(
    symbol: str,
    interval: str,
    *,
    target_rows: int | None = None,
    chunk: int = 1000,
    max_rounds: int | None = 200,
) -> int:
    sym = symbol.upper()
    bounded_chunk = _bounded_chunk(chunk)
    state = _initial_state(sym, interval)
    while _needs_more_history(state, target_rows, max_rounds):
        rows = _fetch_backfill_rows(sym, interval, bounded_chunk, state.end_time)
        if not rows:
            return _complete_or_raise(sym, interval, state, target_rows)
        upsert_klines_rows(sym, interval, rows)
        state = _next_state(sym, interval, state, rows)
    return _complete_or_raise(sym, interval, state, target_rows)


def _bounded_chunk(chunk: int) -> int:
    return min(1000, max(50, chunk))


def _initial_state(symbol: str, interval: str) -> BackfillState:
    oldest = oldest_open_time(symbol, interval)
    end_time = int(oldest) - 1 if oldest is not None else None
    return BackfillState(current=count_klines(symbol, interval), end_time=end_time)


def _fetch_backfill_rows(symbol: str, interval: str, chunk: int, end_time: int | None) -> list[dict]:
    try:
        return fetch_klines(symbol, interval, limit=chunk, end_time=end_time)
    except Exception as exc:
        raise KlineBackfillError(
            f"failed to fetch {interval} klines for {symbol} before {end_time} with chunk={chunk}: {exc}"
        ) from exc


def _next_state(symbol: str, interval: str, state: BackfillState, rows: list[dict]) -> BackfillState:
    new_oldest = min(int(row["openTime"]) for row in rows)
    if state.end_time is not None and new_oldest >= state.end_time:
        raise KlineBackfillError(f"historical {interval} kline backfill did not move earlier for {symbol}")
    return BackfillState(current=count_klines(symbol, interval), end_time=new_oldest - 1, rounds=state.rounds + 1)


def _needs_more_history(
    state: BackfillState,
    target_rows: int | None,
    max_rounds: int | None,
) -> bool:
    if target_rows is not None and state.current >= target_rows:
        return False
    if max_rounds is not None and state.rounds >= max_rounds:
        return False
    return True


def _complete_or_raise(
    symbol: str,
    interval: str,
    state: BackfillState,
    target_rows: int | None,
) -> int:
    if target_rows is None and state.current > 0:
        return state.current
    if target_rows is None:
        raise KlineBackfillError(f"no historical {interval} klines returned for {symbol}")
    if state.current >= target_rows:
        return state.current
    raise KlineBackfillError(
        f"{interval} kline backfill for {symbol} stopped after {state.rounds} rounds: "
        f"{state.current}/{target_rows} rows"
    )
