from __future__ import annotations

from app.services import market_data_bar_features as service

START_TIME = 1_700_000_000_000
TEN_MINUTES_MS = 600_000
FUNDING_RATE_A = 0.0001
FUNDING_RATE_B = 0.0002


def test_backfill_funding_features_fetches_full_kline_range(monkeypatch) -> None:
    kline_times = [START_TIME, START_TIME + TEN_MINUTES_MS, START_TIME + TEN_MINUTES_MS * 2]
    captured = {}

    def fetch_history(symbol: str, *, start_time: int, end_time: int, limit: int):
        captured["fetch"] = {
            "symbol": symbol,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit,
        }
        return [
            {"open_time": START_TIME - service.FUNDING_INTERVAL_MS, "funding_rate": FUNDING_RATE_A},
            {"open_time": START_TIME + TEN_MINUTES_MS, "funding_rate": FUNDING_RATE_B},
        ]

    monkeypatch.setattr(service, "_kline_open_times", lambda *_args: kline_times)
    monkeypatch.setattr(service, "fetch_funding_rate_history", fetch_history)
    monkeypatch.setattr(service, "upsert_funding_rows", lambda symbol, rows: captured.update({"upsert": (symbol, rows)}))

    result = service.backfill_funding_features("btcusdt", "10m")

    assert result == {"inserted": 3, "klineBars": 3}
    assert captured["fetch"] == {
        "symbol": "BTCUSDT",
        "startTime": START_TIME - service.FUNDING_INTERVAL_MS,
        "endTime": START_TIME + TEN_MINUTES_MS * 2,
        "limit": service.FUNDING_HISTORY_LIMIT,
    }
    assert captured["upsert"] == (
        "BTCUSDT",
        [
            {"open_time": START_TIME, "funding_rate": FUNDING_RATE_A},
            {"open_time": START_TIME + TEN_MINUTES_MS, "funding_rate": FUNDING_RATE_B},
            {"open_time": START_TIME + TEN_MINUTES_MS * 2, "funding_rate": FUNDING_RATE_B},
        ],
    )


def test_fetch_funding_rate_history_paginates_until_range_end(monkeypatch) -> None:
    calls = []

    def retry_get(_url: str, params: dict):
        calls.append(dict(params))
        if len(calls) == 1:
            return [
                {"fundingTime": START_TIME, "fundingRate": str(FUNDING_RATE_A)},
                {"fundingTime": START_TIME + TEN_MINUTES_MS, "fundingRate": str(FUNDING_RATE_B)},
            ]
        return [
            {"fundingTime": START_TIME + TEN_MINUTES_MS * 2, "fundingRate": str(FUNDING_RATE_A)},
        ]

    monkeypatch.setattr(service, "retry_get", retry_get)

    rows = service.fetch_funding_rate_history(
        "btcusdt",
        start_time=START_TIME,
        end_time=START_TIME + TEN_MINUTES_MS * 3,
        limit=2,
    )

    assert calls == [
        {
            "symbol": "BTCUSDT",
            "limit": 2,
            "startTime": START_TIME,
            "endTime": START_TIME + TEN_MINUTES_MS * 3,
        },
        {
            "symbol": "BTCUSDT",
            "limit": 2,
            "startTime": START_TIME + TEN_MINUTES_MS + service.FUNDING_PAGE_STEP_MS,
            "endTime": START_TIME + TEN_MINUTES_MS * 3,
        },
    ]
    assert rows == [
        {"open_time": START_TIME, "funding_rate": FUNDING_RATE_A},
        {"open_time": START_TIME + TEN_MINUTES_MS, "funding_rate": FUNDING_RATE_B},
        {"open_time": START_TIME + TEN_MINUTES_MS * 2, "funding_rate": FUNDING_RATE_A},
    ]
