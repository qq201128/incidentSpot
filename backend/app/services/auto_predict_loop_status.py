from __future__ import annotations

from copy import deepcopy
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
    "currentTask": None,
    "currentTasks": [],
    "lastTaskTiming": None,
}


def auto_predict_loop_status() -> dict[str, Any]:
    with _LOCK:
        return _status_snapshot(_STATE)


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
        currentTask=None,
        currentTasks=[],
        lastTaskTiming=None,
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
        currentTask=None,
        currentTasks=[],
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
        currentTask=None,
        currentTasks=[],
        stoppedAt=None,
        stopReason=None,
    )


def record_auto_predict_current_task(task: dict[str, Any]) -> str:
    started_at = _utc_now()
    task_id = _task_id(task)
    current = {**task, "taskId": task_id, "startedAt": started_at, "elapsedSeconds": 0.0}
    with _LOCK:
        _STATE["status"] = STATUS_RUNNING
        _STATE["updatedAt"] = started_at
        _STATE["currentTasks"] = _upsert_task(_current_tasks(), current)
        _STATE["currentTask"] = current
    return task_id


def record_auto_predict_current_task_progress(task_id: str | None = None, **changes: Any) -> None:
    updated_at = _utc_now()
    with _LOCK:
        current = _matching_current_task(task_id)
        if isinstance(current, dict):
            updated = {**current, **changes, "updatedAt": updated_at}
            _STATE["currentTasks"] = _upsert_task(_current_tasks(), updated)
            _STATE["currentTask"] = updated
            _STATE["updatedAt"] = updated_at


def record_auto_predict_current_task_done(timings: dict[str, Any], task_id: str | None = None) -> None:
    finished_at = _utc_now()
    with _LOCK:
        current = _matching_current_task(task_id)
        if isinstance(current, dict):
            _STATE["lastTaskTiming"] = {
                **current,
                "finishedAt": finished_at,
                "elapsedSeconds": _elapsed_seconds(str(current["startedAt"]), finished_at),
                "timings": dict(timings),
            }
            _STATE["currentTasks"] = _remove_task(_current_tasks(), str(current["taskId"]))
        remaining = _current_tasks()
        _STATE["currentTask"] = remaining[-1] if remaining else None
        _STATE["updatedAt"] = finished_at


def record_auto_predict_loop_stopped(reason: str) -> None:
    record_loop_stopped(BACKGROUND_LOOP_NAME, reason)
    stopped_at = _utc_now()
    changes = {
        "updatedAt": stopped_at,
        "currentTask": None,
        "currentTasks": [],
        "stoppedAt": stopped_at,
        "stopReason": reason,
    }
    with _LOCK:
        failed = _STATE.get("status") == STATUS_FAILED
    if not failed:
        changes["status"] = STATUS_STOPPED
    _replace_state(**changes)


def _replace_state(**changes: Any) -> None:
    with _LOCK:
        _STATE.update(changes)


def _status_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(state)
    current = snapshot.get("currentTask")
    if isinstance(current, dict) and current.get("startedAt"):
        current["elapsedSeconds"] = _elapsed_seconds(str(current["startedAt"]), _utc_now())
    for task in snapshot.get("currentTasks") or []:
        if isinstance(task, dict) and task.get("startedAt"):
            task["elapsedSeconds"] = _elapsed_seconds(str(task["startedAt"]), _utc_now())
    return snapshot


def _task_id(task: dict[str, Any]) -> str:
    parts = [
        task.get("currentStage"),
        task.get("currentFamily"),
        task.get("symbol"),
        task.get("duration"),
        task.get("entryOpenTime"),
    ]
    return ":".join(str(part) for part in parts)


def _current_tasks() -> list[dict[str, Any]]:
    tasks = _STATE.get("currentTasks")
    return list(tasks) if isinstance(tasks, list) else []


def _matching_current_task(task_id: str | None) -> dict[str, Any] | None:
    if task_id is None:
        current = _STATE.get("currentTask")
        return current if isinstance(current, dict) else None
    for task in _current_tasks():
        if task.get("taskId") == task_id:
            return task
    return None


def _upsert_task(tasks: list[dict[str, Any]], task: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in tasks if item.get("taskId") != task.get("taskId")] + [task]


def _remove_task(tasks: list[dict[str, Any]], task_id: str) -> list[dict[str, Any]]:
    return [item for item in tasks if item.get("taskId") != task_id]


def _elapsed_seconds(started_at: str, ended_at: str) -> float:
    start = datetime.fromisoformat(started_at)
    end = datetime.fromisoformat(ended_at)
    return round(max((end - start).total_seconds(), 0.0), 6)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
