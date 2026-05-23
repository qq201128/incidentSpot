from __future__ import annotations

import time

from app.services import binance_market_data as market_data


def test_premium_index_display_uses_cache(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_retry_get(url: str, params: dict, *, max_attempts: int, timeout):
        calls["count"] += 1
        return {
            "symbol": params["symbol"],
            "markPrice": "100.0",
            "indexPrice": "101.0",
            "lastFundingRate": "0.0001",
            "nextFundingTime": 1,
            "time": 1_700_000_000_000,
        }

    monkeypatch.setattr(market_data, "retry_get", fake_retry_get)
    monkeypatch.setattr(market_data, "_maybe_persist_premium_index", lambda *_args: None)
    market_data.LAST_PREMIUM_INDEX.clear()

    first = market_data.get_premium_index_display("BTCUSDT")
    second = market_data.get_premium_index_display("BTCUSDT")

    assert first["indexPrice"] == 101.0
    assert second["indexPrice"] == 101.0
    assert calls["count"] == 1


def test_premium_index_display_returns_stale_on_fetch_error(monkeypatch) -> None:
    market_data.LAST_PREMIUM_INDEX.clear()
    market_data.LAST_PREMIUM_INDEX["BTCUSDT"] = (
        time.monotonic() - 5.0,
        {
            "symbol": "BTCUSDT",
            "markPrice": 100.0,
            "indexPrice": 99.5,
            "lastFundingRate": 0.0,
            "nextFundingTime": 0,
            "time": 1,
        },
    )

    def fail_retry_get(*_args, **_kwargs):
        raise TimeoutError("network timeout")

    monkeypatch.setattr(market_data, "retry_get", fail_retry_get)

    row = market_data.get_premium_index_display("BTCUSDT")

    assert row["indexPrice"] == 99.5
    assert row["stale"] is True
