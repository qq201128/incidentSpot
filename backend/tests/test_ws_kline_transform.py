from __future__ import annotations

from app.services.ws_kline_transform import candle_from_index_price_event

BUCKET_OPEN_TIME = 1778160600000
FIRST_EVENT_TIME = BUCKET_OPEN_TIME + 12_000
SECOND_EVENT_TIME = BUCKET_OPEN_TIME + 22_000
TEN_MINUTES = 10
SECONDS_PER_MINUTE = 60
MS_PER_SECOND = 1000
TEN_M_MS = TEN_MINUTES * SECONDS_PER_MINUTE * MS_PER_SECOND


def test_index_price_tick_preserves_seeded_candle_range() -> None:
    seed = {
        "bucket": BUCKET_OPEN_TIME,
        "open": 80800.0,
        "high": 80850.0,
        "low": 80750.0,
        "close": 80810.0,
        "volume": 0,
        "closeTime": FIRST_EVENT_TIME,
    }

    candle, state = candle_from_index_price_event(
        {"E": SECOND_EVENT_TIME, "i": "80880.25"},
        "10m",
        seed,
    )

    assert candle is not None
    assert state is not None
    assert candle["openTime"] == BUCKET_OPEN_TIME
    assert candle["open"] == 80800.0
    assert candle["high"] == 80880.25
    assert candle["low"] == 80750.0
    assert candle["close"] == 80880.25
    assert candle["isClosed"] is False


def test_index_price_tick_starts_next_interval() -> None:
    next_bucket = BUCKET_OPEN_TIME + TEN_M_MS

    candle, _ = candle_from_index_price_event(
        {"E": next_bucket, "i": "80901.5"},
        "10m",
        {"bucket": BUCKET_OPEN_TIME, "open": 1, "high": 1, "low": 1, "close": 1},
    )

    assert candle is not None
    assert candle["openTime"] == next_bucket
    assert candle["open"] == 80901.5
    assert candle["high"] == 80901.5
    assert candle["low"] == 80901.5
