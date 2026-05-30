from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any

from app.services.background_loop_status import record_loop_failure, record_loop_start, record_loop_success

STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
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
        initialDelaySeconds=float(initial_delay),
        pollSeconds=int(poll_seconds),
    )


def record_auto_predict_cycle_success(target_count: int, next_wait_seconds: float) -> None:
    details = {"targetCount": int(target_count), "nextWaitSeconds": float(next_wait_seconds)}
    record_loop_success(BACKGROUND_LOOP_NAME, details)
    _replace_state(
        status=STATUS_PASSED,
        updatedAt=_utc_now(),
        error=None,
        exceptionType=None,
        targetCount=int(target_count),
        nextWaitSeconds=float(next_wait_seconds),
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
    )


def _replace_state(**changes: Any) -> None:
    with _LOCK:
        _STATE.update(changes)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
