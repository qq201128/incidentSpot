from __future__ import annotations

import json
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.lstm_artifacts import artifact_paths
from app.services.model_family_candidate_search_service import (
    ModelCandidateSearchConfig,
    run_model_candidate_search,
)
from app.services.model_search_job_store import (
    claim_next_model_search_job,
    fail_model_search_job,
    finish_model_search_job,
    heartbeat_model_search_job,
)
from app.services.model_search_resource import (
    ModelSearchResourceConfig,
    apply_model_search_resource_config,
    resource_config_from_job,
)
from app.services.model_search_status_service import model_search_queue_status
from app.services.model_search_job_types import DEFAULT_STALE_AFTER_SECONDS

DEFAULT_LOG_DIR = Path("runtime") / "model-search-jobs"
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30


@dataclass(frozen=True)
class ModelSearchWorkerConfig:
    max_running_jobs: int = 1
    resource: ModelSearchResourceConfig = ModelSearchResourceConfig()
    log_dir: Path = DEFAULT_LOG_DIR
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS
    heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS


class ModelSearchHeartbeat:
    def __init__(self, job_id: str, interval_seconds: int) -> None:
        self.job_id = job_id
        self.interval_seconds = max(int(interval_seconds), 1)
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._loop, name="model-search-heartbeat", daemon=True)

    def __enter__(self) -> "ModelSearchHeartbeat":
        heartbeat_model_search_job(self.job_id)
        self._thread.start()
        return self

    def __exit__(self, exc_type, _exc, _tb) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        if self._error is not None and exc_type is None:
            raise RuntimeError(f"model search heartbeat failed: {self._error}") from self._error

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                heartbeat_model_search_job(self.job_id)
            except BaseException as exc:
                self._error = exc
                self._stop.set()


def run_one_model_search_job(config: ModelSearchWorkerConfig | None = None) -> dict[str, Any]:
    selected = config or ModelSearchWorkerConfig()
    job = claim_next_model_search_job(
        max_running_jobs=selected.max_running_jobs,
        stale_after_seconds=selected.stale_after_seconds,
    )
    if job is None:
        return {"status": "idle", "reason": "no_pending_job", "queue": model_search_queue_status()}
    resource = apply_model_search_resource_config(resource_config_from_job(selected.resource, job))
    return _run_claimed_job(job, selected, resource)


def _run_claimed_job(
    job: dict[str, Any],
    config: ModelSearchWorkerConfig,
    resource: dict[str, Any],
) -> dict[str, Any]:
    log_path = _job_log_path(config.log_dir, job)
    try:
        with ModelSearchHeartbeat(job["job_id"], config.heartbeat_interval_seconds):
            result = _run_search_with_log(job, resource, log_path)
        artifact_path = str(artifact_paths(job["symbol"], job["duration"], family=job["model_family"]).root)
        stored = finish_model_search_job(
            job["job_id"],
            result=result,
            resource=resource,
            artifact_path=artifact_path,
            log_path=str(log_path),
        )
        return {"status": stored["status"], "job": stored, "result": result}
    except Exception as exc:
        failure = _failure_payload(job, resource, exc)
        _append_failure_log(log_path, job, failure)
        stored = fail_model_search_job(
            job["job_id"],
            failure_type=type(exc).__name__,
            failure_reason=str(exc),
            failure_context=failure,
            resource=resource,
            log_path=str(log_path),
        )
        raise RuntimeError(json.dumps({"status": "failed", "job": stored}, ensure_ascii=False)) from exc


def _run_search_with_log(job: dict[str, Any], resource: dict[str, Any], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        with redirect_stdout(handle), redirect_stderr(handle):
            _write_log_event(handle, "start", job, resource)
            result = run_model_candidate_search(_search_config(job, resource))
            _write_log_event(handle, "finish", job, {"result": result})
            return result


def _search_config(job: dict[str, Any], resource: dict[str, Any]) -> ModelCandidateSearchConfig:
    return ModelCandidateSearchConfig(
        family=job["model_family"],
        symbol=job["symbol"],
        duration=job["duration"],
        profile=job["profile"],
        parallel_workers=int(resource["parallelWorkers"]),
        reset_history=bool(job.get("resetHistory")),
    )


def _failure_payload(job: dict[str, Any], resource: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "jobId": job["job_id"],
        "symbol": job["symbol"],
        "duration": job["duration"],
        "modelFamily": job["model_family"],
        "profile": job["profile"],
        "resource": resource,
        "exceptionType": type(exc).__name__,
        "exceptionMessage": str(exc),
        "traceback": traceback.format_exc(),
    }


def _job_log_path(log_dir: Path, job: dict[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    name = f"{stamp}_{job['symbol']}_{job['duration']}_{job['model_family']}_{job['job_id'][:8]}.log"
    return log_dir / name


def _write_log_event(handle: Any, event: str, job: dict[str, Any], payload: dict[str, Any]) -> None:
    handle.write(json.dumps({"event": event, "jobId": job["job_id"], "payload": payload}, ensure_ascii=False) + "\n")
    handle.flush()


def _append_failure_log(log_path: Path, job: dict[str, Any], failure: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        _write_log_event(handle, "failure", job, failure)
