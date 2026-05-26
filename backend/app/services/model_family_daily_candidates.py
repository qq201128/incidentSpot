from __future__ import annotations

from typing import Any, Callable

from app.services.model_family_config import MODEL_FAMILIES
from app.services.model_family_status_service import model_family_status


def model_family_daily_candidate_report(
    symbol: str,
    duration: str,
    *,
    families: tuple[str, ...] = MODEL_FAMILIES,
    status_loader: Callable[[str, str, str], dict[str, Any]] = model_family_status,
) -> dict[str, Any]:
    rows = [_family_status_payload(family, symbol, duration, status_loader) for family in families]
    failures = [row for row in rows if row["status"] == "failed"]
    return {
        "policy": "model_validation_gate_is_paper_live_prefilter_only",
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "familyCount": len(rows),
        "paperLiveReadyCount": sum(1 for row in rows if row.get("paperLiveAdmissionAllowed")),
        "failures": failures,
        "models": rows,
        "realTradingEnabled": False,
    }


def _family_status_payload(family: str, symbol: str, duration: str, status_loader) -> dict[str, Any]:
    try:
        status = status_loader(family, symbol.strip().upper(), duration)
    except Exception as exc:
        return _failure_payload(family, symbol, duration, exc)
    admission = status.get("paperLiveAdmission") or {}
    return {
        "modelFamily": family,
        "modelVersion": status.get("modelVersion"),
        "featureWindow": status.get("featureWindow"),
        "minConfidence": admission.get("minConfidence") or status.get("selectedConfidenceThreshold"),
        "validationWinRate": admission.get("validationWinRate") or status.get("validationWinRate"),
        "paperLiveWinRate": status.get("paperLiveWinRate"),
        "paperLiveSampleCount": status.get("paperLiveSampleCount"),
        "paperLiveStatus": status.get("paperLiveStatus"),
        "paperLiveAdmissionAllowed": admission.get("allowed") is True,
        "activeModelStatus": status.get("activeModelStatus") or status.get("status"),
        "reason": admission.get("reason") or status.get("shadowPredictionBlockedReason"),
        "status": "passed",
        "realTradingEnabled": False,
    }


def _failure_payload(family: str, symbol: str, duration: str, exc: Exception) -> dict[str, Any]:
    return {
        "modelFamily": family,
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "status": "failed",
        "reason": str(exc),
        "exceptionType": type(exc).__name__,
        "realTradingEnabled": False,
    }
