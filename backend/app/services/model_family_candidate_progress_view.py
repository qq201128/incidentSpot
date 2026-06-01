from __future__ import annotations

from typing import Any

from app.services.model_family_candidates import (
    model_search_space_size,
    read_model_candidate_library,
    read_model_candidate_progress,
)
from app.services.model_family_config import normalize_model_family

RECENT_LIMIT = 8
ACTIVE_PROGRESS_STATUSES = {"queued", "running"}


def read_model_candidate_progress_view(family: str, symbol: str, duration: str, *, artifact_root=None) -> dict[str, Any]:
    selected = normalize_model_family(family)
    progress = read_model_candidate_progress(selected, symbol, duration, artifact_root=artifact_root)
    if _progress_has_runtime_state(progress):
        return progress
    records = list(read_model_candidate_library(selected, symbol, duration, artifact_root=artifact_root)["records"])
    if not records:
        return progress
    return _progress_from_library(selected, symbol, duration, records)


def _progress_has_runtime_state(progress: dict[str, Any]) -> bool:
    status = str(progress.get("status") or "")
    if status in ACTIVE_PROGRESS_STATUSES:
        return True
    if int(progress.get("completed") or 0) > 0:
        return True
    return bool(progress.get("lastFailure") or progress.get("failureReason"))


def _progress_from_library(family: str, symbol: str, duration: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = len(records)
    total = max(model_search_space_size(family), completed)
    latest = records[-1]
    return {
        "status": _library_status(records),
        "modelFamily": family,
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "profile": latest.get("profile"),
        "updatedAt": latest.get("recordedAt"),
        "total": total,
        "completed": completed,
        "searchSpaceTotal": total,
        "percent": round(min(completed / total, 1.0), 4) if total else 0.0,
        "counts": _counts_from_records(records),
        "latestCompleted": _progress_record(latest),
        "recent": [_progress_record(row) for row in records[-RECENT_LIMIT:]][::-1],
        "source": "candidate_library",
    }


def _library_status(records: list[dict[str, Any]]) -> str:
    statuses = {str(row.get("status") or "") for row in records}
    if "trade_active" in statuses or "trained" in statuses:
        return "trade_active"
    if "shadow_active" in statuses:
        return "shadow_active"
    return str(records[-1].get("status") or "failed")


def _counts_from_records(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "tradeActive": 0,
        "shadowActive": 0,
        "initialBaseline": 0,
        "validationFailed": 0,
        "insufficientSamples": 0,
        "failed": 0,
    }
    for record in records:
        counts[_count_key(record.get("status"))] += 1
    return counts


def _count_key(status: Any) -> str:
    labels = {
        "trade_active": "tradeActive",
        "trained": "tradeActive",
        "shadow_active": "shadowActive",
        "initial_baseline": "initialBaseline",
        "validation_failed": "validationFailed",
        "insufficient_samples": "insufficientSamples",
    }
    return labels.get(str(status or "failed"), "failed")


def _progress_record(record: dict[str, Any]) -> dict[str, Any]:
    return {**record, "completedAt": record.get("recordedAt")}
