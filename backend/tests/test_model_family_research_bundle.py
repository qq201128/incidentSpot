from __future__ import annotations

from app.services import model_family_research_bundle as bundle
from app.services.model_family_config import MODEL_FAMILIES


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
