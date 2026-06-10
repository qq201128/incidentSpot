from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.services.lstm_combo_snapshot import current_combo_snapshot
from app.services.model_family_config import MODEL_FAMILIES
from app.services.model_family_status_service import model_family_status


def model_family_research_bundle(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    shared_combo = current_combo_snapshot(sym, duration)

    def load(family: str) -> dict[str, Any]:
        try:
            status = model_family_status(
                family,
                sym,
                duration,
                current_combo_snapshot=shared_combo,
            )
            return _slim_research_row(status)
        except Exception as exc:
            return _failed_row(family, sym, duration, str(exc))

    workers = min(6, len(MODEL_FAMILIES))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        models = list(pool.map(load, MODEL_FAMILIES))
    return {"symbol": sym, "duration": duration, "models": models}


def _slim_research_row(status: dict[str, Any]) -> dict[str, Any]:
    admission = status.get("paperLiveAdmission") or {}
    sample_counts = status.get("sampleCounts") or {}
    return {
        "modelFamily": status.get("modelFamily"),
        "strategyKey": status.get("strategyKey"),
        "modelVersion": status.get("modelVersion"),
        "featureWindow": status.get("featureWindow"),
        "validationWinRate": admission.get("validationWinRate") or status.get("validationWinRate"),
        "validationSampleCount": sample_counts.get("validation"),
        "testWinRate": status.get("testWinRate"),
        "paperLiveAdmission": admission,
        "paperLiveStatus": status.get("paperLiveStatus") or admission.get("status"),
        "cleanEventFeatures": status.get("cleanEventFeatures"),
        "regimeValidation": status.get("regimeValidation"),
        "shadowPredictionBlockedReason": status.get("shadowPredictionBlockedReason"),
        "validationFailureReason": status.get("validationFailureReason"),
    }


def _failed_row(family: str, symbol: str, duration: str, reason: str) -> dict[str, Any]:
    return {
        "modelFamily": family,
        "strategyKey": None,
        "modelVersion": None,
        "paperLiveStatus": "model_status_failed",
        "shadowPredictionBlockedReason": reason,
        "validationFailureReason": reason,
    }
