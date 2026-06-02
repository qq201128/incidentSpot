from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.services.model_family_config import MODEL_FAMILIES
from app.services.model_family_status_service import model_family_status
from app.services.paper_live_candidate_service import paper_live_candidate_report


@dataclass(frozen=True)
class _FamilyRequest:
    family: str
    symbol: str
    duration: str


def model_family_daily_candidate_report(
    symbol: str,
    duration: str,
    *,
    families: tuple[str, ...] = MODEL_FAMILIES,
    status_loader: Callable[[str, str, str], dict[str, Any]] = model_family_status,
    lifecycle_loader: Callable[[str, str], dict[str, Any]] = paper_live_candidate_report,
) -> dict[str, Any]:
    lifecycle = _model_lifecycle_by_family(symbol, duration, lifecycle_loader)
    rows = [
        _family_status_payload(_FamilyRequest(family, symbol.strip().upper(), duration), status_loader, lifecycle)
        for family in families
    ]
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


def _family_status_payload(
    request: _FamilyRequest,
    status_loader,
    lifecycle: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    try:
        status = status_loader(request.family, request.symbol, request.duration)
    except Exception as exc:
        return _failure_payload(request, exc)
    admission = status.get("paperLiveAdmission") or {}
    paper_live = lifecycle.get(request.family) or {}
    return {
        "modelFamily": request.family,
        "modelVersion": status.get("modelVersion"),
        "featureWindow": status.get("featureWindow"),
        "minConfidence": _first_present(admission, status, ("minConfidence", "selectedConfidenceThreshold")),
        "validationWinRate": _first_present(admission, status, ("validationWinRate", "validationWinRate")),
        "candidateLibrary": status.get("candidateLibrary") or {},
        "paperLiveWinRate": paper_live.get("paperLiveWinRate"),
        "paperLiveSampleCount": paper_live.get("paperLiveSampleCount", 0),
        "paperLiveStatus": _first_present(paper_live, admission, ("paperLiveStatus", "status")),
        "paperLiveReason": _first_present(paper_live, admission, ("reason", "reason")),
        "paperLiveAdmissionAllowed": admission.get("allowed") is True,
        "activeModelStatus": _first_present(status, status, ("activeModelStatus", "status")),
        "reason": _first_present(admission, status, ("reason", "shadowPredictionBlockedReason")),
        "status": "passed",
        "realTradingEnabled": False,
    }


def _first_present(primary: dict[str, Any], fallback: dict[str, Any], keys: tuple[str, str]) -> Any:
    primary_key, fallback_key = keys
    value = primary.get(primary_key)
    return fallback.get(fallback_key) if value is None else value


def _model_lifecycle_by_family(
    symbol: str,
    duration: str,
    lifecycle_loader: Callable[[str, str], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    report = lifecycle_loader(symbol.strip().upper(), duration)
    candidates = report.get("allCandidates") or []
    return {
        str(row["modelFamily"]): row
        for row in candidates
        if isinstance(row, dict) and row.get("candidateType") == "model" and row.get("modelFamily")
    }


def _failure_payload(request: _FamilyRequest, exc: Exception) -> dict[str, Any]:
    return {
        "modelFamily": request.family,
        "symbol": request.symbol,
        "duration": request.duration,
        "status": "failed",
        "reason": str(exc),
        "exceptionType": type(exc).__name__,
        "realTradingEnabled": False,
    }
