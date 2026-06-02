from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from app.services.model_family_daily_candidates import model_family_daily_candidate_report
from app.services.model_family_status_service import model_family_status
from app.services.model_search_job_store import list_model_search_jobs
from app.services.model_search_job_types import JOB_STATUS_RUNNING

MODEL_SEARCH_WORKER_COMMAND = "python backend/scripts/run_model_search_worker.py --loop --adaptive-parallelism"


def model_search_queue_status(
    filters: dict[str, Any] | None = None,
    *,
    include_symbol_details: bool = True,
) -> dict[str, Any]:
    jobs = list_model_search_jobs(filters)
    grouped = _group_jobs(jobs) if include_symbol_details else {}
    worker = model_search_worker_status(jobs, active_jobs=_active_worker_jobs(filters, jobs))
    return {
        "version": "model_search_status_v1",
        "realTradingEnabled": False,
        "totalJobs": len(jobs),
        "counts": dict(Counter(str(job["status"]) for job in jobs)),
        "symbols": [_symbol_payload(symbol, rows) for symbol, rows in grouped.items()],
        "runningJobs": [job for job in jobs if job["status"] == JOB_STATUS_RUNNING],
        "failedJobs": _failure_rows(jobs),
        "rejectedJobs": _rejected_rows(jobs),
        "latestLogPath": worker["latestLogPath"],
        "latestFailureReason": worker["latestFailureReason"],
        "worker": worker,
        "workerStatus": worker,
    }


def model_search_status_with_lifecycle(filters: dict[str, Any] | None = None) -> dict[str, Any]:
    status = model_search_queue_status(filters)
    for symbol in status["symbols"]:
        for duration in symbol["durations"]:
            duration["paperLive"] = _paper_live_payload(symbol["symbol"], duration["duration"])
    return status


def model_search_worker_status(
    jobs: list[dict[str, Any]],
    active_jobs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    counts = Counter(str(job["status"]) for job in jobs)
    active_counts = Counter(str(job["status"]) for job in active_jobs or jobs)
    pending = int(counts.get("pending") or 0)
    running = int(active_counts.get("running") or 0)
    failed = _latest_relevant_failed_job(jobs)
    state = _worker_state(pending, running, failed)
    return {
        "state": state,
        "pendingJobs": pending,
        "runningJobs": running,
        "failedJobs": int(counts.get("failed") or 0),
        "latestLogPath": _latest_log_path(jobs),
        "latestFailureReason": _failure_reason(failed),
        "latestFailureType": failed.get("failure_type") if failed else None,
        "latestFailedJobId": failed.get("job_id") if failed else None,
        "workerRequiredCommand": MODEL_SEARCH_WORKER_COMMAND,
        "managedByApi": False,
    }


def _worker_state(pending: int, running: int, failed: dict[str, Any] | None) -> str:
    if pending > 0 and running > 0:
        return "queued"
    if running > 0:
        return "running"
    if pending > 0:
        return "worker_required"
    if failed is not None:
        return "failed"
    return "idle"


def _active_worker_jobs(filters: dict[str, Any] | None, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not filters:
        return jobs
    if _has_running_job(jobs):
        return jobs
    running = list_model_search_jobs({"statuses": (JOB_STATUS_RUNNING,)})
    if not running:
        return jobs
    seen = {str(job["job_id"]) for job in jobs}
    merged = list(jobs)
    for job in running:
        job_id = str(job["job_id"])
        if job_id in seen:
            continue
        seen.add(job_id)
        merged.append(job)
    return merged


def _has_running_job(jobs: list[dict[str, Any]]) -> bool:
    return any(job["status"] == JOB_STATUS_RUNNING for job in jobs)


def _group_jobs(jobs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        grouped[str(job["symbol"])].append(job)
    return dict(grouped)


def _symbol_payload(symbol: str, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    durations = defaultdict(list)
    for job in jobs:
        durations[str(job["duration"])].append(job)
    return {
        "symbol": symbol,
        "counts": dict(Counter(str(job["status"]) for job in jobs)),
        "durations": [_duration_payload(duration, rows) for duration, rows in durations.items()],
    }


def _duration_payload(duration: str, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    families = defaultdict(list)
    for job in jobs:
        families[str(job["model_family"])].append(job)
    return {
        "duration": duration,
        "counts": dict(Counter(str(job["status"]) for job in jobs)),
        "families": [_family_payload(family, rows) for family, rows in families.items()],
    }


def _family_payload(family: str, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    latest = sorted(jobs, key=lambda item: str(item.get("created_at") or ""), reverse=True)[0]
    status = _model_status(latest)
    return {
        "modelFamily": family,
        "counts": dict(Counter(str(job["status"]) for job in jobs)),
        "latestJob": latest,
        "modelStatus": status.get("status"),
        "shadowPredictionReady": status.get("shadowPredictionReady"),
        "blockedReason": status.get("shadowPredictionBlockedReason"),
        "modelStatusError": status.get("error"),
        "modelStatusExceptionType": status.get("exceptionType"),
        "candidateSearchProgress": status.get("candidateSearchProgress"),
    }


def _model_status(job: dict[str, Any]) -> dict[str, Any]:
    try:
        return model_family_status(job["model_family"], job["symbol"], job["duration"])
    except Exception as exc:
        return {
            "status": "status_failed",
            "shadowPredictionReady": False,
            "shadowPredictionBlockedReason": str(exc),
            "error": str(exc),
            "exceptionType": type(exc).__name__,
        }


def _paper_live_payload(symbol: str, duration: str) -> dict[str, Any]:
    report = model_family_daily_candidate_report(symbol, duration)
    return {
        "paperCollectingCount": sum(1 for row in report["models"] if row.get("paperLiveStatus") == "paper_collecting"),
        "paperStableCount": sum(1 for row in report["models"] if row.get("paperLiveStatus") == "paper_stable"),
        "paperFailedCount": sum(1 for row in report["models"] if row.get("paperLiveStatus") == "paper_failed"),
        "models": report["models"],
    }


def _failure_rows(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [job for job in jobs if job["status"] == "failed"]


def _rejected_rows(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [job for job in jobs if job["status"] == "rejected"]


def _latest_failed_job(jobs: list[dict[str, Any]]) -> dict[str, Any] | None:
    failed = _failure_rows(jobs)
    if not failed:
        return None
    return sorted(failed, key=_latest_timestamp, reverse=True)[0]


def _latest_successful_job(jobs: list[dict[str, Any]]) -> dict[str, Any] | None:
    succeeded = [job for job in jobs if job["status"] == "succeeded"]
    if not succeeded:
        return None
    return sorted(succeeded, key=_latest_timestamp, reverse=True)[0]


def _latest_relevant_failed_job(jobs: list[dict[str, Any]]) -> dict[str, Any] | None:
    failed = _latest_failed_job(jobs)
    succeeded = _latest_successful_job(jobs)
    if failed is None:
        return None
    if succeeded is None:
        return failed
    return failed if _latest_timestamp(failed) > _latest_timestamp(succeeded) else None


def _latest_log_path(jobs: list[dict[str, Any]]) -> str | None:
    logged = [job for job in jobs if job.get("log_path")]
    if not logged:
        return None
    return sorted(logged, key=_latest_timestamp, reverse=True)[0]["log_path"]


def _latest_timestamp(job: dict[str, Any]) -> str:
    return str(job.get("finished_at") or job.get("heartbeat_at") or job.get("started_at") or job.get("created_at") or "")


def _failure_reason(job: dict[str, Any] | None) -> str | None:
    if job is None:
        return None
    return job.get("failure_reason") or job.get("rejection_reason")
