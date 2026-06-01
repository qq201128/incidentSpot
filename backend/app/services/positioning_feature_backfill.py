from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from app.services.binance_context_data import (
    fetch_global_long_short_ratio,
    fetch_open_interest_statistics,
    fetch_taker_buy_sell_volume,
)
from app.services.external_factor_data import upsert_positioning_rows
from app.services.kline_timing import MS_PER_MINUTE

POSITIONING_PERIOD = "5m"
POSITIONING_STEP_MS = 5 * MS_PER_MINUTE
POSITIONING_FETCH_LIMIT = 500
MAX_POSITIONING_BACKFILL_ROUNDS = 24


@dataclass(frozen=True)
class _PositioningBackfillState:
    rows_by_time: dict[int, dict[str, Any]]
    end_time: int | None = None


@dataclass(frozen=True)
class _PositioningBatch:
    rows: list[dict[str, Any]]
    oldest: int


def refresh_positioning_features_for_lookback(symbol: str, lookback_start_ms: int) -> int:
    sym = symbol.strip().upper()
    start_ms = max(0, int(lookback_start_ms))
    rows = _paginated_positioning_rows(sym, start_ms)
    if not rows:
        raise ValueError(f"no positioning features returned for {sym} from {start_ms}")
    upsert_positioning_rows(sym, rows)
    return len(rows)


def _paginated_positioning_rows(symbol: str, lookback_start_ms: int) -> list[dict[str, Any]]:
    state = _PositioningBackfillState(defaultdict(dict))
    for _round in range(MAX_POSITIONING_BACKFILL_ROUNDS):
        batch = _positioning_batch(symbol, state.end_time)
        if batch is None:
            break
        state = _next_positioning_state(state, batch, symbol)
        if batch.oldest <= lookback_start_ms:
            break
    _assert_positioning_coverage(symbol, lookback_start_ms, state.rows_by_time)
    return _positioning_payload_rows(state.rows_by_time)


def _next_positioning_state(
    state: _PositioningBackfillState,
    batch: _PositioningBatch,
    symbol: str,
) -> _PositioningBackfillState:
    rows_by_time = defaultdict(dict, state.rows_by_time)
    for row in batch.rows:
        rows_by_time[int(row["open_time"])].update(row)
    next_end = batch.oldest - POSITIONING_STEP_MS
    if state.end_time is not None and next_end >= state.end_time:
        raise ValueError(f"positioning backfill did not move earlier for {symbol}")
    return _PositioningBackfillState(rows_by_time, next_end)


def _assert_positioning_coverage(
    symbol: str,
    lookback_start_ms: int,
    rows_by_time: dict[int, dict[str, Any]],
) -> None:
    if any(open_time >= lookback_start_ms for open_time in rows_by_time):
        return
    oldest = min(rows_by_time) if rows_by_time else None
    raise ValueError(
        f"positioning backfill did not reach lookback window for {symbol}: "
        f"start={lookback_start_ms} oldest={oldest}"
    )


def _positioning_batch(symbol: str, end_time: int | None) -> _PositioningBatch | None:
    rows = _merged_positioning_batch(symbol, end_time=end_time)
    if not rows:
        return None
    oldest = min(int(row["open_time"]) for row in rows)
    return _PositioningBatch(rows, oldest)


def _merged_positioning_batch(symbol: str, *, end_time: int | None) -> list[dict[str, Any]]:
    kwargs = {
        "period": POSITIONING_PERIOD,
        "limit": POSITIONING_FETCH_LIMIT,
        "end_time": end_time,
    }
    rows_by_time: dict[int, dict[str, Any]] = defaultdict(dict)
    for row in fetch_open_interest_statistics(symbol, **kwargs):
        rows_by_time[int(row["open_time"])].update(row)
    for row in fetch_global_long_short_ratio(symbol, **kwargs):
        rows_by_time[int(row["open_time"])].update(row)
    for row in fetch_taker_buy_sell_volume(symbol, **kwargs):
        rows_by_time[int(row["open_time"])].update(row)
    return [
        {"open_time": open_time, **payload}
        for open_time, payload in sorted(rows_by_time.items())
    ]


def _positioning_payload_rows(rows_by_time: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"open_time": open_time, **payload}
        for open_time, payload in sorted(rows_by_time.items())
    ]
