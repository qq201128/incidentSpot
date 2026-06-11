from __future__ import annotations

import pytest

from app.services import positioning_feature_backfill as backfill


def test_paginated_positioning_rows_merges_batches(monkeypatch) -> None:
    calls: list[int | None] = []

    def fake_open_interest(_symbol: str, **_kwargs) -> list[dict]:
        calls.append(_kwargs.get("end_time"))
        if _kwargs.get("end_time") is None:
            return [{"open_time": 1_000_000, "open_interest": 10.0, "open_interest_value": 20.0}]
        return [{"open_time": 500_000, "open_interest": 5.0, "open_interest_value": 8.0}]

    monkeypatch.setattr(backfill, "fetch_open_interest_statistics", fake_open_interest)
    monkeypatch.setattr(backfill, "fetch_global_long_short_ratio", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(backfill, "fetch_taker_buy_sell_volume", lambda *_args, **_kwargs: [])

    rows = backfill._paginated_positioning_rows("ETHUSDT", lookback_start_ms=600_000, now_ms=1_200_000)

    assert calls == [None, 1_000_000 - backfill.POSITIONING_STEP_MS]
    assert [row["open_time"] for row in rows] == [500_000, 1_000_000]


def test_paginated_positioning_rows_stops_at_retention_window(monkeypatch) -> None:
    calls: list[int | None] = []
    retention_start = 10_000_000
    now_ms = retention_start + backfill.POSITIONING_RETENTION_DAYS * backfill.MS_PER_DAY

    def fake_open_interest(_symbol: str, **_kwargs) -> list[dict]:
        end_time = _kwargs.get("end_time")
        calls.append(end_time)
        if end_time is None:
            return [{"open_time": retention_start + 2 * backfill.POSITIONING_STEP_MS}]
        return [{"open_time": retention_start}]

    monkeypatch.setattr(backfill, "fetch_open_interest_statistics", fake_open_interest)
    monkeypatch.setattr(backfill, "fetch_global_long_short_ratio", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(backfill, "fetch_taker_buy_sell_volume", lambda *_args, **_kwargs: [])

    rows = backfill._paginated_positioning_rows("BTCUSDT", lookback_start_ms=0, now_ms=now_ms)

    assert calls == [None, retention_start + backfill.POSITIONING_STEP_MS]
    assert [row["open_time"] for row in rows] == [
        retention_start,
        retention_start + 2 * backfill.POSITIONING_STEP_MS,
    ]


def test_refresh_positioning_features_for_lookback_upserts(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        backfill,
        "_paginated_positioning_rows",
        lambda *_args, **_kwargs: [{"open_time": 1, "taker_buy_vol": 1.0, "taker_sell_vol": 2.0}],
    )
    monkeypatch.setattr(
        backfill,
        "upsert_positioning_rows",
        lambda symbol, rows: captured.update({"symbol": symbol, "rows": rows}),
    )

    total = backfill.refresh_positioning_features_for_lookback("ethusdt", 0)

    assert total == 1
    assert captured["symbol"] == "ETHUSDT"


def test_refresh_positioning_features_for_lookback_raises_when_empty(monkeypatch) -> None:
    monkeypatch.setattr(backfill, "_paginated_positioning_rows", lambda *_args, **_kwargs: [])

    with pytest.raises(ValueError, match="no positioning features returned"):
        backfill.refresh_positioning_features_for_lookback("ETHUSDT", 0)
