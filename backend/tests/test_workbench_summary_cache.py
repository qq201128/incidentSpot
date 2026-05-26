from __future__ import annotations

import time

from app.services import workbench_summary_cache as cache


def test_serves_stale_summary_without_blocking_rebuild(monkeypatch) -> None:
    cache.clear_workbench_summary_cache()
    builds: list[tuple[str, str]] = []

    def build(symbol: str, duration: str) -> dict:
        builds.append((symbol, duration))
        return {"symbol": symbol, "duration": duration, "n": len(builds)}

    monkeypatch.setattr(cache, "SUMMARY_CACHE_TTL_SECONDS", 0.01)
    monkeypatch.setattr(cache, "SUMMARY_STALE_SERVE_SECONDS", 10.0)
    monkeypatch.setattr(cache, "_schedule_background_refresh", lambda *_args, **_kwargs: None)

    first = cache.get_cached_workbench_summary("BTCUSDT", "10m", build=build)
    assert first["n"] == 1
    assert first["cache"]["cached"] is False

    time.sleep(0.02)
    second = cache.get_cached_workbench_summary("BTCUSDT", "10m", build=build)
    assert second["n"] == 1
    assert second["cache"]["stale"] is True
    assert len(builds) == 1
