from __future__ import annotations

import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable

from app.services.model_search_adaptive_parallelism import (
    AdaptiveParallelismConfig,
    decide_adaptive_parallelism,
    max_adaptive_jobs,
    sample_host_resources,
)
from app.services.model_search_job_runner import ModelSearchWorkerConfig, run_one_model_search_job
from app.services.model_search_job_store import list_model_search_jobs
from app.services.model_search_job_types import JOB_STATUS_PENDING

JOB_SUMMARY_KEYS = (
    "job_id",
    "symbol",
    "duration",
    "model_family",
    "profile",
    "status",
    "stage",
    "created_at",
    "started_at",
    "finished_at",
    "heartbeat_at",
    "artifact_path",
    "log_path",
    "failure_type",
    "failure_reason",
    "rejection_reason",
    "resource_profile",
    "internal_threads",
    "parallel_workers",
    "xgboost_process_workers",
    "attempt_count",
    "resetHistory",
)
RESULT_SUMMARY_KEYS = (
    "status",
    "reason",
    "validationFailureReason",
    "family",
    "modelStatus",
    "shadowPredictionReady",
    "batchStatus",
)
JOB_BATCH_SUMMARY_KEYS = (
    "selectedCandidates",
    "availableCandidatesBeforeJob",
    "remainingCandidatesAfterJob",
    "hasMoreCandidates",
)
QUEUE_SUMMARY_KEYS = ("version", "totalJobs", "counts")
OMITTED_RESULT_KEYS = ("reports", "trainingRules", "successiveHalvingStages")


class ProcessJobLauncher:
    def __init__(self, max_workers: int) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._executor = ProcessPoolExecutor(max_workers=max_workers)

    def __enter__(self) -> "ProcessJobLauncher":
        return self

    def __exit__(self, exc_type, _exc, _tb) -> None:
        self._executor.shutdown(wait=True, cancel_futures=exc_type is not None)

    def start(self, config: ModelSearchWorkerConfig) -> Future:
        return self._executor.submit(_run_one_model_search_job, config)


@dataclass(frozen=True)
class WorkerPoolRuntime:
    base_config: ModelSearchWorkerConfig
    adaptive_config: AdaptiveParallelismConfig
    poll_seconds: float
    run_until_empty: bool


@dataclass(frozen=True)
class RunningModelSearchJob:
    handle: Any
    adaptive_decision: dict[str, Any]


def run_adaptive_worker_pool(
    base_config: ModelSearchWorkerConfig,
    adaptive_config: AdaptiveParallelismConfig,
    *,
    poll_seconds: float,
    run_until_empty: bool,
    launcher_factory: Callable[[int], Any] = ProcessJobLauncher,
    pending_job_counter: Callable[[], int] | None = None,
) -> Iterable[dict[str, Any]]:
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    initial_sample = sample_host_resources(adaptive_config.cpu_sample_seconds)
    max_workers = max_adaptive_jobs(adaptive_config, initial_sample, base_config.resource)
    runtime = WorkerPoolRuntime(base_config, adaptive_config, poll_seconds, run_until_empty)
    pending_counter = pending_job_counter or pending_model_search_job_count
    with launcher_factory(max_workers) as launcher:
        yield from _run_pool_loop(runtime=runtime, launcher=launcher, pending_job_counter=pending_counter)


def _run_pool_loop(
    *,
    runtime: WorkerPoolRuntime,
    launcher: Any,
    pending_job_counter: Callable[[], int],
) -> Iterable[dict[str, Any]]:
    active: list[RunningModelSearchJob] = []
    pending_exhausted = False
    while True:
        idle_seen = False
        continuation_seen = False
        for job in _completed_jobs(active):
            active.remove(job)
            report = job.handle.result()
            idle_seen = idle_seen or _is_idle(report)
            continuation_seen = continuation_seen or _has_continuation(report)
            yield _report_with_adaptive_decision(report, job.adaptive_decision)
        if continuation_seen:
            pending_exhausted = False
        if idle_seen and not active and runtime.run_until_empty and not continuation_seen:
            return
        if idle_seen and not active:
            time.sleep(runtime.poll_seconds)
            continue
        if idle_seen and active:
            pending_exhausted = True
        if pending_exhausted and active:
            time.sleep(runtime.poll_seconds)
            continue
        decision = _decision(runtime.base_config, runtime.adaptive_config, len(active))
        decision_payload = decision.to_payload()
        if decision.target_jobs <= len(active):
            pending_jobs = 0 if active else pending_job_counter()
            if runtime.run_until_empty and not active and pending_jobs == 0:
                yield _idle_report(decision_payload)
                return
            yield from _wait_for_capacity(runtime, active, decision_payload, pending_jobs)
            continue
        handle = launcher.start(_job_config(runtime.base_config, decision.target_jobs))
        active.append(RunningModelSearchJob(handle, decision_payload))
        if not active or not _completed_jobs(active):
            time.sleep(runtime.poll_seconds)
            continue


def pending_model_search_job_count() -> int:
    return len(list_model_search_jobs({"statuses": (JOB_STATUS_PENDING,)}))


def _wait_for_capacity(
    runtime: WorkerPoolRuntime,
    active: list[RunningModelSearchJob],
    decision_payload: dict[str, Any],
    pending_jobs: int,
) -> Iterable[dict[str, Any]]:
    if active:
        time.sleep(runtime.poll_seconds)
        return
    yield _waiting_report(decision_payload, pending_jobs)
    time.sleep(runtime.poll_seconds)


def _completed_jobs(active: list[RunningModelSearchJob]) -> list[RunningModelSearchJob]:
    return [job for job in list(active) if job.handle.done()]


def _decision(
    base_config: ModelSearchWorkerConfig,
    adaptive_config: AdaptiveParallelismConfig,
    running_jobs: int,
):
    sample = sample_host_resources(adaptive_config.cpu_sample_seconds)
    return decide_adaptive_parallelism(
        adaptive_config,
        sample,
        running_jobs=running_jobs,
        resource=base_config.resource,
    )


def _job_config(base_config: ModelSearchWorkerConfig, max_running_jobs: int) -> ModelSearchWorkerConfig:
    return replace(base_config, max_running_jobs=max_running_jobs)


def _is_idle(report: dict[str, Any]) -> bool:
    return report.get("status") == "idle" and report.get("reason") == "no_pending_job"


def _has_continuation(report: dict[str, Any]) -> bool:
    return report.get("status") == "partial"


def _report_with_adaptive_decision(report: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    return {**report, "adaptiveParallelism": decision}


def _waiting_report(decision: dict[str, Any], pending_jobs: int) -> dict[str, Any]:
    return {
        "status": "waiting",
        "reason": str(decision.get("reason") or "adaptive_capacity_limited"),
        "queue": {"pending": int(pending_jobs)},
        "adaptiveParallelism": decision,
    }


def _idle_report(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "idle",
        "reason": "no_pending_job",
        "queue": {"pending": 0},
        "adaptiveParallelism": decision,
    }


def _run_one_model_search_job(config: ModelSearchWorkerConfig) -> dict[str, Any]:
    return _ipc_safe_report(run_one_model_search_job(config))


def _ipc_safe_report(report: dict[str, Any]) -> dict[str, Any]:
    payload = {"status": report.get("status"), "summaryOnly": True}
    _copy_optional_fields(payload, report, ("reason",))
    job = _dict_summary(report.get("job"), JOB_SUMMARY_KEYS)
    result = _result_summary(report.get("result"))
    queue = _queue_summary(report.get("queue"))
    if job:
        payload["job"] = job
    if result:
        payload["result"] = result
    if queue:
        payload["queue"] = queue
    return payload


def _result_summary(result: Any) -> dict[str, Any]:
    summary = _dict_summary(result, RESULT_SUMMARY_KEYS)
    if not isinstance(result, dict):
        return summary
    batch = _dict_summary(result.get("jobBatch"), JOB_BATCH_SUMMARY_KEYS)
    omitted = [key for key in OMITTED_RESULT_KEYS if key in result]
    if batch:
        summary["jobBatch"] = batch
    if omitted:
        summary["omittedKeys"] = omitted
    return summary


def _queue_summary(queue: Any) -> dict[str, Any]:
    summary = _dict_summary(queue, QUEUE_SUMMARY_KEYS)
    if not isinstance(queue, dict):
        return summary
    worker = queue.get("workerStatus") or queue.get("worker")
    worker_summary = _dict_summary(worker, ("state", "pendingJobs", "runningJobs", "failedJobs"))
    if worker_summary:
        summary["workerStatus"] = worker_summary
    return summary


def _dict_summary(payload: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {key: payload[key] for key in keys if key in payload}


def _copy_optional_fields(target: dict[str, Any], source: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in source:
            target[key] = source[key]
