from __future__ import annotations

from app.services import factor_combination_data_dependencies as dependencies


def test_refresh_dependencies_backfills_bar_aligned_features(monkeypatch) -> None:
    calls = []

    def refresh_klines(symbol: str, duration: str) -> dict:
        calls.append(("klines", symbol, duration))
        return {"after": 42}

    monkeypatch.setattr(
        dependencies,
        "ingest_market_context_data",
        lambda symbol, *, durations: calls.append(("context", symbol, durations)) or {"ok": True},
    )
    monkeypatch.setattr(
        dependencies,
        "backfill_funding_features",
        lambda symbol, duration: calls.append(("funding", symbol, duration)) or {"inserted": 1},
    )
    monkeypatch.setattr(
        dependencies,
        "backfill_orderbook_features",
        lambda symbol, duration: calls.append(("orderbook", symbol, duration)) or {"inserted": 2},
    )

    report = dependencies.refresh_factor_combination_data_dependencies(
        "ethusdt",
        "1d",
        refresh_duration_klines=refresh_klines,
    )

    assert calls == [
        ("context", "ETHUSDT", ("1d",)),
        ("klines", "ETHUSDT", "1d"),
        ("funding", "ETHUSDT", "1d"),
        ("orderbook", "ETHUSDT", "1d"),
    ]
    assert report["featureFill"] == {
        "funding": {"inserted": 1},
        "orderbook": {"inserted": 2},
    }
    assert report["durationKlines"] == {"after": 42}
