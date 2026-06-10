from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any

from app.services.background_loop_status import record_loop_failure, record_loop_start, record_loop_stopped, record_loop_success

STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_STOPPED = "stopped"
BACKGROUND_LOOP_NAME = "auto_predict"

_LOCK = Lock()
_STATE: dict[str, Any] = {
    "status": STATUS_IDLE,
    "startedAt": None,
    "updatedAt": None,
    "error": None,
    "exceptionType": None,
    "targetCount": 0,
    "nextWaitSeconds": None,
    "pollSeconds": None,
    "readyDueCount": 0,
    "collectionTargetCount": 0,
    "activeTargetCount": 0,
    "skippedTargetCount": 0,
    "skippedTargets": [],
    "phase": None,
}


def auto_predict_loop_status() -> dict[str, Any]:
    with _LOCK:
        return dict(_STATE)


def record_auto_predict_loop_start(*, initial_delay: float, poll_seconds: int) -> None:
    details = {"initialDelaySeconds": float(initial_delay), "pollSeconds": int(poll_seconds)}
    record_loop_start(BACKGROUND_LOOP_NAME, details)
    _replace_state(
        status=STATUS_RUNNING,
        startedAt=_utc_now(),
        updatedAt=_utc_now(),
        error=None,
        exceptionType=None,
        failureDetails=None,
        initialDelaySeconds=float(initial_delay),
        pollSeconds=int(poll_seconds),
        readyDueCount=0,
        collectionTargetCount=0,
        activeTargetCount=0,
        skippedTargetCount=0,
        skippedTargets=[],
        phase="initial_delay" if initial_delay > 0 else "starting",
        stoppedAt=None,
        stopReason=None,
    )


def record_auto_predict_cycle_progress(
    target_count: int,
    *,
    cycle_details: dict[str, Any],
) -> None:
    details = {"targetCount": int(target_count), **cycle_details}
    skipped_targets = list(details.get("skippedTargets") or [])
    _replace_state(
        status=STATUS_RUNNING,
        updatedAt=_utc_now(),
        error=None,
        exceptionType=None,
        failureDetails=None,
        targetCount=int(target_count),
        readyDueCount=int(details.get("readyDueCount") or 0),
        collectionTargetCount=int(details.get("collectionTargetCount") or 0),
        activeTargetCount=int(details.get("activeTargetCount") or 0),
        skippedTargetCount=len(skipped_targets),
        skippedTargets=skipped_targets,
        phase=details.get("phase"),
        stoppedAt=None,
        stopReason=None,
    )


def record_auto_predict_cycle_success(
    target_count: int,
    next_wait_seconds: float,
    *,
    cycle_details: dict[str, Any] | None = None,
) -> None:
    details = {"targetCount": int(target_count), "nextWaitSeconds": float(next_wait_seconds)}
    if cycle_details:
        details.update(cycle_details)
    record_loop_success(BACKGROUND_LOOP_NAME, details)
    skipped_targets = list(details.get("skippedTargets") or [])
    _replace_state(
        status=STATUS_PASSED,
        updatedAt=_utc_now(),
        error=None,
        exceptionType=None,
        failureDetails=None,
        targetCount=int(target_count),
        nextWaitSeconds=float(next_wait_seconds),
        readyDueCount=int(details.get("readyDueCount") or 0),
        collectionTargetCount=int(details.get("collectionTargetCount") or 0),
        activeTargetCount=int(details.get("activeTargetCount") or 0),
        skippedTargetCount=len(skipped_targets),
        skippedTargets=skipped_targets,
        phase="completed",
        stoppedAt=None,
        stopReason=None,
    )


def record_auto_predict_cycle_failure(exc: Exception, next_wait_seconds: float) -> None:
    details = {
        "nextWaitSeconds": float(next_wait_seconds),
        "failureDetails": getattr(exc, "details", None),
    }
    record_loop_failure(BACKGROUND_LOOP_NAME, exc, details)
    _replace_state(
        status=STATUS_FAILED,
        updatedAt=_utc_now(),
        error=str(exc),
        exceptionType=type(exc).__name__,
        failureDetails=getattr(exc, "details", None),
        nextWaitSeconds=float(next_wait_seconds),
        readyDueCount=0,
        collectionTargetCount=0,
        activeTargetCount=0,
        skippedTargetCount=0,
        skippedTargets=[],
        phase=None,
        stoppedAt=None,
        stopReason=None,
    )


def record_auto_predict_loop_stopped(reason: str) -> None:
    record_loop_stopped(BACKGROUND_LOOP_NAME, reason)
    stopped_at = _utc_now()
    changes = {"updatedAt": stopped_at, "stoppedAt": stopped_at, "stopReason": reason}
    with _LOCK:
        failed = _STATE.get("status") == STATUS_FAILED
    if not failed:
        changes["status"] = STATUS_STOPPED
    _replace_state(**changes)


def _replace_state(**changes: Any) -> None:
    with _LOCK:
        _STATE.update(changes)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
