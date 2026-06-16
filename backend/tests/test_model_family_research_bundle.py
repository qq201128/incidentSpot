from __future__ import annotations

from app.services import model_family_research_bundle as bundle
from app.services.model_family_config import MODEL_FAMILIES


def setup_function() -> None:
    bundle.clear_model_family_research_bundle_cache()


def teardown_function() -> None:
    bundle.clear_model_family_research_bundle_cache()


def test_research_bundle_returns_all_families(monkeypatch) -> None:
    monkeypatch.setattr(
        bundle,
        "model_family_research_status",
        lambda family, symbol, duration, current_combo_snapshot=None: {
            "modelFamily": family,
            "strategyKey": f"factor_{family}_shadow_{duration}",
            "cleanEventFeatures": family == "lstm",
            "regimeValidation": {"trend_up:normal_vol": {"winRate": 0.6, "sampleCount": 3}},
            "shadowPredictionBlockedReason": "passed" if family == "lstm" else "clean_event_retrain_required",
        },
    )
    payload = bundle.model_family_research_bundle("BTCUSDT", "10m")
    assert payload["symbol"] == "BTCUSDT"
    assert len(payload["models"]) == len(MODEL_FAMILIES)
    lstm = next(row for row in payload["models"] if row["modelFamily"] == "lstm")
    assert lstm["cleanEventFeatures"] is True
    assert lstm["regimeValidation"]["trend_up:normal_vol"]["winRate"] == 0.6


def test_research_bundle_returns_expired_cache_while_refreshing(monkeypatch) -> None:
    scheduled: list[tuple[str, str]] = []
    cached = {"symbol": "BTCUSDT", "duration": "10m", "models": [{"modelFamily": "lstm"}]}

    monkeypatch.setenv("MODEL_RESEARCH_BUNDLE_CACHE_TTL_SECONDS", "5")
    monkeypatch.setattr(bundle.time, "monotonic", lambda: 100.0)
    with bundle._cache_lock:
        bundle._cache[("BTCUSDT", "10m")] = (100.0, cached)
    monkeypatch.setattr(bundle.time, "monotonic", lambda: 110.0)
    monkeypatch.setattr(bundle, "_schedule_refresh", lambda key: scheduled.append(key))

    payload = bundle.model_family_research_bundle("BTCUSDT", "10m")

    assert payload["models"] == [{"modelFamily": "lstm"}]
    assert payload["cache"] == {
        "hit": True,
        "stale": True,
        "warming": True,
        "ageSeconds": 10.0,
    }
    assert scheduled == [("BTCUSDT", "10m")]
