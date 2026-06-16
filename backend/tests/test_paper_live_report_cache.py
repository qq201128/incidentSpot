from __future__ import annotations

from app.services import paper_live_report_cache as cache


def test_paper_live_report_cache_hit(monkeypatch) -> None:
    cache.clear_paper_live_report_cache()
    calls: list[tuple[str, str]] = []

    def build(symbol: str, duration: str) -> dict:
        calls.append((symbol, duration))
        return {"symbol": symbol, "duration": duration, "ok": True}

    first = cache.get_cached_paper_live_report("btcusdt", "10m", build=build)
    second = cache.get_cached_paper_live_report("BTCUSDT", "10m", build=build)

    assert first["cache"]["hit"] is False
    assert second["cache"]["hit"] is True
    assert calls == [("BTCUSDT", "10m")]


def test_paper_live_report_cache_returns_expired_entry_while_refreshing(monkeypatch) -> None:
    cache.clear_paper_live_report_cache()
    scheduled: list[tuple[str, str]] = []

    monkeypatch.setenv("PAPER_LIVE_REPORT_CACHE_TTL_SECONDS", "5")
    monkeypatch.setattr(cache.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(cache, "_schedule_refresh", lambda key, _build: scheduled.append(key))
    cache.store_paper_live_report_cache("BTCUSDT", "10m", {"symbol": "BTCUSDT", "duration": "10m"})

    monkeypatch.setattr(cache.time, "monotonic", lambda: 110.0)
    result = cache.get_cached_paper_live_report(
        "BTCUSDT",
        "10m",
        build=lambda _symbol, _duration: {"unexpected": True},
    )

    assert result["symbol"] == "BTCUSDT"
    assert result["cache"] == {
        "hit": True,
        "stale": True,
        "warming": True,
        "ageSeconds": 10.0,
    }
    assert scheduled == [("BTCUSDT", "10m")]
