from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any

STATUS_RUNNING = "running"
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_STOPPED = "stopped"

_LOCK = Lock()
_STATE: dict[str, dict[str, Any]] = {}


def background_loop_statuses() -> dict[str, dict[str, Any]]:
    with _LOCK:
        return {name: dict(payload) for name, payload in _STATE.items()}


def record_loop_start(name: str, details: dict[str, Any] | None = None) -> None:
    _update(name, {"status": STATUS_RUNNING, "startedAt": _utc_now(), "details": details or {}})


def record_loop_success(name: str, details: dict[str, Any] | None = None) -> None:
    _update(name, {"status": STATUS_PASSED, "lastSuccessAt": _utc_now(), "lastSuccessDetails": details or {}})


def record_loop_failure(
    name: str,
    exc: Exception,
    details: dict[str, Any] | None = None,
) -> None:
    _update(
        name,
        {
            "status": STATUS_FAILED,
            "lastFailureAt": _utc_now(),
            "lastError": str(exc),
            "lastExceptionType": type(exc).__name__,
            "lastFailureDetails": details or {},
        },
    )


def record_loop_stopped(name: str, reason: str) -> None:
    with _LOCK:
        current = _STATE.setdefault(name, {"status": STATUS_RUNNING})
        status = current.get("status")
    changes = {"stoppedAt": _utc_now(), "stopReason": reason}
    if status != STATUS_FAILED:
        changes["status"] = STATUS_STOPPED
    _update(name, changes)


def reset_background_loop_statuses() -> None:
    with _LOCK:
        _STATE.clear()


def _update(name: str, changes: dict[str, Any]) -> None:
    with _LOCK:
        current = _STATE.setdefault(name, {"status": STATUS_RUNNING})
        current.update({"updatedAt": _utc_now(), **changes})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
