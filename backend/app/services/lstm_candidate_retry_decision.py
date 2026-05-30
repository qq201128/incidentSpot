from __future__ import annotations

from typing import Any

RETRY_TRAIN_STATUSES = {"untrained", "validation_failed", "failed", "insufficient_samples"}


def retry_decision(status: dict[str, Any], *, manual_trigger: bool = False) -> dict[str, Any]:
    if manual_trigger:
        return {"shouldTrain": True, "reason": "manual_candidate_search"}
    if str(status.get("lastAttemptStatus") or "") == "training":
        return {"shouldTrain": False, "reason": "training_in_progress"}
    reason = _retry_reason(status)
    if reason:
        return {"shouldTrain": True, "reason": reason}
    if _active_model_ready(status):
        return {"shouldTrain": False, "reason": "active_model_ready"}
    return {"shouldTrain": False, "reason": "not_retryable"}


def _retry_reason(status: dict[str, Any]) -> str | None:
    active_status = str(status.get("activeModelStatus") or status.get("status") or "")
    if active_status == "shadow_active":
        return "shadow_active_candidate_search"
    if active_status in RETRY_TRAIN_STATUSES:
        return f"model_status_{active_status}"
    if not bool(status.get("comboSnapshotMatches")):
        return "combo_snapshot_mismatch"
    if not bool(status.get("artifactsReady")):
        return "artifacts_incomplete"
    return None


def _active_model_ready(status: dict[str, Any]) -> bool:
    return bool(status.get("shadowPredictionReady")) and bool(status.get("comboSnapshotMatches"))
