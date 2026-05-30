from __future__ import annotations

import pytest

from app.services import factor_combination_background as combo_background
from app.services.background_loop_status import background_loop_statuses, reset_background_loop_statuses

STALE_BY_DURATION = {"10m": True, "30m": False, "60m": True, "1d": False}


def test_startup_stale_refresh_records_success_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_background_loop_statuses()
    refreshed = []

    monkeypatch.setattr(combo_background, "_sync_default_simulation_slots", lambda: None)
    monkeypatch.setattr(combo_background, "factor_ranking_precomputed_symbols", lambda: ["BTCUSDT"])
    monkeypatch.setattr(combo_background, "_ranking_cache_needs_refresh", _is_stale_duration)
    monkeypatch.setattr(
        combo_background,
        "refresh_combination_ranking_for_symbol_duration",
        lambda symbol, duration, config, **_kwargs: refreshed.append((symbol, duration, config)),
    )

    combo_background.refresh_stale_configured_combination_rankings("cfg")

    details = background_loop_statuses()["factor_combo_daily"]["lastSuccessDetails"]
    assert refreshed == [("BTCUSDT", "10m", "cfg"), ("BTCUSDT", "60m", "cfg")]
    assert details["stage"] == "stale_configured"
    assert details["refreshedItems"] == [
        {"symbol": "BTCUSDT", "duration": "10m"},
        {"symbol": "BTCUSDT", "duration": "60m"},
    ]
    assert details["failedItems"] == []
    assert details["skippedItems"] == [
        {"symbol": "BTCUSDT", "duration": "30m"},
        {"symbol": "BTCUSDT", "duration": "1d"},
    ]


def test_startup_stale_refresh_records_failed_items(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_background_loop_statuses()
    refreshed = []

    def refresh(symbol: str, duration: str, config: object, **_kwargs) -> None:
        if duration == "60m":
            raise RuntimeError("refresh boom")
        refreshed.append((symbol, duration, config))

    monkeypatch.setattr(combo_background, "_sync_default_simulation_slots", lambda: None)
    monkeypatch.setattr(combo_background, "factor_ranking_precomputed_symbols", lambda: ["BTCUSDT"])
    monkeypatch.setattr(combo_background, "_ranking_cache_needs_refresh", _is_stale_duration)
    monkeypatch.setattr(combo_background, "refresh_combination_ranking_for_symbol_duration", refresh)

    combo_background.refresh_stale_configured_combination_rankings("cfg")

    status = background_loop_statuses()["factor_combo_daily"]
    assert refreshed == [("BTCUSDT", "10m", "cfg")]
    assert status["status"] == "failed"
    assert status["lastError"] == "stale factor combo ranking failed for items"
    assert status["lastFailureDetails"]["failedItems"] == [
        {"symbol": "BTCUSDT", "duration": "60m", "error": "refresh boom", "exceptionType": "RuntimeError"}
    ]


def _is_stale_duration(_symbol: str, duration: str) -> bool:
    return STALE_BY_DURATION[duration]
