from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.services.background_loop_status import record_loop_failure, record_loop_start, record_loop_success
from app.services.model_search_job_defaults import DEFAULT_CANDIDATE_BUDGET, DEFAULT_CANDIDATES_PER_JOB
from app.services.model_search_job_types import DEFAULT_STALE_AFTER_SECONDS

LOOP_NAME = "model_search_api_worker"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKER_SCRIPT = PROJECT_ROOT / "backend" / "scripts" / "run_model_search_worker.py"
API_WORKER_LOG_DIR = PROJECT_ROOT / "runtime" / "model-search-api-worker"
JOB_LOG_DIR = PROJECT_ROOT / "runtime" / "model-search-jobs"


@dataclass
class _ApiWorkerState:
    process: Any | None = None
    command: tuple[str, ...] = field(default_factory=tuple)
    log_path: str | None = None
    started_at: str | None = None
    last_exit_code: int | None = None
    last_failure_reason: str | None = None


_LOCK = threading.Lock()
_STATE = _ApiWorkerState()


def ensure_api_model_search_worker(
    resource: dict[str, Any],
    *,
    launcher: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    command = _worker_command(resource)
    with _LOCK:
        _refresh_exited_worker_locked()
        if _is_running_locked():
            return {**_status_locked(), "started": False}
        log_path = _launch_log_path()
        try:
            process = _launch_process(command, log_path, launcher)
        except OSError as exc:
            _record_start_failure_locked(command, log_path, exc)
            raise RuntimeError(f"model search API worker startup failed: {exc}") from exc
        _store_started_worker_locked(process, command, log_path)
        record_loop_start(LOOP_NAME, {"command": _display_command(command), "logPath": str(log_path)})
        threading.Thread(target=_monitor_process, args=(process,), name=LOOP_NAME, daemon=True).start()
        return {**_status_locked(), "started": True}


def api_model_search_worker_status() -> dict[str, Any]:
    with _LOCK:
        _refresh_exited_worker_locked()
        return _status_locked()


def reset_api_model_search_worker_state() -> None:
    with _LOCK:
        _STATE.process = None
        _STATE.command = ()
        _STATE.log_path = None
        _STATE.started_at = None
        _STATE.last_exit_code = None
        _STATE.last_failure_reason = None


def _launch_process(command: tuple[str, ...], log_path: Path, launcher: Callable[..., Any]) -> Any:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        return launcher(
            list(command),
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=_creation_flags(),
        )


def _worker_command(resource: dict[str, Any]) -> tuple[str, ...]:
    return (
        sys.executable,
        str(WORKER_SCRIPT),
        "--run-until-empty",
        "--max-running-jobs",
        "0",
        "--internal-threads",
        str(_resource_int(resource, "internalThreads")),
        "--parallel-workers",
        str(_resource_int(resource, "parallelWorkers")),
        "--xgboost-process-workers",
        str(_resource_int(resource, "xgboostProcessWorkers")),
        "--torch-jobs",
        str(_resource_int(resource, "torchJobs")),
        "--resource-profile",
        str(resource.get("resourceProfile") or "local_safe"),
        "--log-dir",
        str(JOB_LOG_DIR),
        "--stale-after-seconds",
        str(DEFAULT_STALE_AFTER_SECONDS),
        "--candidates-per-job",
        str(DEFAULT_CANDIDATES_PER_JOB),
        "--candidate-budget",
        str(DEFAULT_CANDIDATE_BUDGET),
        "--compact",
    )


def _resource_int(resource: dict[str, Any], key: str) -> int:
    value = int(resource.get(key) or 1)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _store_started_worker_locked(process: Any, command: tuple[str, ...], log_path: Path) -> None:
    _STATE.process = process
    _STATE.command = command
    _STATE.log_path = str(log_path)
    _STATE.started_at = _utc_now()
    _STATE.last_exit_code = None
    _STATE.last_failure_reason = None


def _monitor_process(process: Any) -> None:
    exit_code = int(process.wait())
    with _LOCK:
        if _STATE.process is process:
            _record_exit_locked(exit_code)


def _refresh_exited_worker_locked() -> None:
    if _STATE.process is None:
        return
    exit_code = _STATE.process.poll()
    if exit_code is not None:
        _record_exit_locked(int(exit_code))


def _record_exit_locked(exit_code: int) -> None:
    _STATE.process = None
    _STATE.last_exit_code = exit_code
    details = {"exitCode": exit_code, "command": _display_command(_STATE.command), "logPath": _STATE.log_path}
    if exit_code == 0:
        _STATE.last_failure_reason = None
        record_loop_success(LOOP_NAME, details)
        return
    reason = f"model search API worker exited with code {exit_code}"
    _STATE.last_failure_reason = reason
    record_loop_failure(LOOP_NAME, RuntimeError(reason), details)


def _record_start_failure_locked(command: tuple[str, ...], log_path: Path, exc: OSError) -> None:
    _STATE.command = command
    _STATE.log_path = str(log_path)
    _STATE.started_at = _utc_now()
    _STATE.last_exit_code = None
    _STATE.last_failure_reason = str(exc)
    record_loop_failure(LOOP_NAME, exc, {"command": _display_command(command), "logPath": str(log_path)})


def _status_locked() -> dict[str, Any]:
    running = _is_running_locked()
    return {
        "running": running,
        "managedByApi": running,
        "command": _display_command(_STATE.command) if _STATE.command else None,
        "logPath": _STATE.log_path,
        "startedAt": _STATE.started_at,
        "lastExitCode": _STATE.last_exit_code,
        "lastFailureReason": _STATE.last_failure_reason,
    }


def _is_running_locked() -> bool:
    return _STATE.process is not None and _STATE.process.poll() is None


def _launch_log_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return API_WORKER_LOG_DIR / f"{stamp}_api_worker.log"


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _display_command(command: tuple[str, ...]) -> str:
    return subprocess.list2cmdline(list(command))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
