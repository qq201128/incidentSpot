from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.model_family_candidate_progress_view import read_model_candidate_progress_view
from app.services.model_family_search_rules import model_family_training_rules
from app.services.model_search_job_store import list_model_search_jobs


def candidate_search_progress(
    family: str,
    symbol: str,
    duration: str,
    *,
    artifact_root: Path | None,
) -> dict[str, Any]:
    progress = read_model_candidate_progress_view(family, symbol, duration, artifact_root=artifact_root)
    if artifact_root is not None:
        return progress
    queued = _latest_active_search_job(family, symbol, duration)
    if queued:
        return _queued_progress_from_job(family, queued, progress)
    if str(progress.get("status") or "") in {"queued", "running"}:
        return _inactive_progress(progress)
    return progress


def _latest_active_search_job(family: str, symbol: str, duration: str) -> dict[str, Any] | None:
    jobs = list_model_search_jobs({
        "symbols": (symbol,),
        "durations": (duration,),
        "families": (family,),
        "statuses": ("pending", "running"),
    })
    if not jobs:
        return None
    return sorted(jobs, key=_active_search_job_sort_key, reverse=True)[0]


def _active_search_job_sort_key(job: dict[str, Any]) -> tuple[int, str]:
    status_rank = 1 if job.get("status") == "running" else 0
    timestamp = str(job.get("started_at") or job.get("created_at") or "")
    return status_rank, timestamp


def _queued_progress_from_job(
    family: str,
    job: dict[str, Any],
    base_progress: dict[str, Any],
) -> dict[str, Any]:
    status = "running" if job.get("status") == "running" else "queued"
    total = _search_space_total(family, base_progress)
    return {
        **base_progress,
        "status": status,
        "profile": job.get("profile"),
        "updatedAt": job.get("started_at") or job.get("created_at"),
        "searchSpaceTotal": total,
        "total": total,
        "parallelWorkers": job.get("parallel_workers"),
        "internalThreads": job.get("internal_threads"),
        "xgboostProcessWorkers": job.get("xgboost_process_workers"),
        "modelSearchJob": job,
    }


def _search_space_total(family: str, progress: dict[str, Any]) -> int:
    total = int(progress.get("searchSpaceTotal") or progress.get("total") or 0)
    if total > 0:
        return total
    return int(model_family_training_rules(family)["searchSpaceTotal"])


def _inactive_progress(progress: dict[str, Any]) -> dict[str, Any]:
    return {**progress, "status": "paused", "staleRuntimeStatus": progress.get("status")}
