from __future__ import annotations

import json
import tempfile
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
from app.services.model_family_status_service import model_family_status
from app.services.model_search_job_store import (
    claim_next_model_search_job,
    fail_model_search_job,
    finish_model_search_job,
    heartbeat_model_search_job,
    retry_failed_model_search_job,
)
from app.services.model_search_untrained_enqueue import is_trained_model_status
from app.services.model_search_resource import (
    ModelSearchResourceConfig,
    apply_model_search_resource_config,
    resource_config_from_job,
)
from app.services.model_search_status_service import model_search_queue_status
from app.services.model_search_job_types import DEFAULT_STALE_AFTER_SECONDS

DEFAULT_LOG_DIR = Path(tempfile.gettempdir()) / "incidentSpot-model-search-jobs"
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30
DEFAULT_CANDIDATES_PER_JOB = 1


@dataclass(frozen=True)
class ModelSearchWorkerConfig:
    max_running_jobs: int = 1
    resource: ModelSearchResourceConfig = ModelSearchResourceConfig()
    log_dir: Path = DEFAULT_LOG_DIR
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS
    heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    candidates_per_job: int = DEFAULT_CANDIDATES_PER_JOB


@dataclass(frozen=True)
class SearchRunContext:
    job: dict[str, Any]
    resource: dict[str, Any]
    log_path: Path
    config: ModelSearchWorkerConfig


@dataclass(frozen=True)
class JobLogEvent:
    event: str
    job: dict[str, Any]
    payload: dict[str, Any]


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
    resource = apply_model_search_resource_config(
        resource_config_from_job(selected.resource, job),
        max_running_jobs=selected.max_running_jobs,
    )
    return _run_claimed_job(job, selected, resource)


def _run_claimed_job(
    job: dict[str, Any],
    config: ModelSearchWorkerConfig,
    resource: dict[str, Any],
) -> dict[str, Any]:
    log_path = _job_log_path(config.log_dir, job)
    try:
        trained = _already_trained_result(job)
        if trained is not None:
            _append_skip_log(log_path, job, trained)
            stored = finish_model_search_job(
                job["job_id"],
                result=trained,
                resource=resource,
                artifact_path=None,
                log_path=str(log_path),
            )
            return {"status": stored["status"], "job": stored, "result": trained}
        with ModelSearchHeartbeat(job["job_id"], config.heartbeat_interval_seconds):
            result = _run_search_with_log(SearchRunContext(job, resource, log_path, config))
        artifact_path = str(artifact_paths(job["symbol"], job["duration"], family=job["model_family"]).root)
        has_more = _has_more_candidates(result)
        stored = finish_model_search_job(
            job["job_id"],
            result=_continuation_result(result) if has_more else result,
            resource=resource,
            artifact_path=artifact_path,
            log_path=str(log_path),
        )
        if has_more:
            continued = retry_failed_model_search_job(stored["job_id"], clear_reset_history=True)
            return {"status": "partial", "job": continued, "result": result}
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


def _run_search_with_log(context: SearchRunContext) -> dict[str, Any]:
    context.log_path.parent.mkdir(parents=True, exist_ok=True)
    with context.log_path.open("a", encoding="utf-8") as handle:
        with redirect_stdout(handle), redirect_stderr(handle):
            _write_log_event(handle, JobLogEvent("start", context.job, context.resource))
            result = run_model_candidate_search(_search_config(context.job, context.resource, context.config))
            _write_log_event(handle, JobLogEvent("finish", context.job, {"result": result}))
            return result


def _search_config(
    job: dict[str, Any],
    resource: dict[str, Any],
    config: ModelSearchWorkerConfig,
) -> ModelCandidateSearchConfig:
    return ModelCandidateSearchConfig(
        family=job["model_family"],
        symbol=job["symbol"],
        duration=job["duration"],
        profile=job["profile"],
        parallel_workers=int(resource["parallelWorkers"]),
        reset_history=bool(job.get("resetHistory")),
        candidates_per_job=config.candidates_per_job,
    )


def _has_more_candidates(result: dict[str, Any]) -> bool:
    batch = result.get("jobBatch") or {}
    return bool(batch.get("hasMoreCandidates"))


def _continuation_result(result: dict[str, Any]) -> dict[str, Any]:
    return {**result, "status": "partial_batch", "batchStatus": result.get("status")}


def _already_trained_result(job: dict[str, Any]) -> dict[str, Any] | None:
    if _is_partial_continuation(job):
        return None
    if bool(job.get("resetHistory")):
        return None
    status = model_family_status(job["model_family"], job["symbol"], job["duration"])
    if not is_trained_model_status(status):
        return None
    return {
        "status": "skipped",
        "reason": "already_trained_skipped",
        "modelStatus": status.get("status"),
        "shadowPredictionReady": status.get("shadowPredictionReady"),
    }


def _is_partial_continuation(job: dict[str, Any]) -> bool:
    report = job.get("trainingReport")
    if not isinstance(report, dict):
        return False
    if report.get("status") == "partial_batch":
        return True
    batch = report.get("jobBatch") if isinstance(report.get("jobBatch"), dict) else {}
    return bool(batch.get("hasMoreCandidates"))


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


def _write_log_event(handle: Any, event: JobLogEvent) -> None:
    payload = {"event": event.event, "jobId": event.job["job_id"], "payload": event.payload}
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _append_failure_log(log_path: Path, job: dict[str, Any], failure: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        _write_log_event(handle, JobLogEvent("failure", job, failure))


def _append_skip_log(log_path: Path, job: dict[str, Any], result: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        _write_log_event(handle, JobLogEvent("skipped", job, result))
