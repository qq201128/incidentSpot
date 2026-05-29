from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from app.services.model_family_daily_candidates import model_family_daily_candidate_report
from app.services.model_family_status_service import model_family_status
from app.services.model_search_job_store import list_model_search_jobs


def model_search_queue_status(filters: dict[str, Any] | None = None) -> dict[str, Any]:
    jobs = list_model_search_jobs(filters)
    grouped = _group_jobs(jobs)
    return {
        "version": "model_search_status_v1",
        "realTradingEnabled": False,
        "totalJobs": len(jobs),
        "counts": dict(Counter(str(job["status"]) for job in jobs)),
        "symbols": [_symbol_payload(symbol, rows) for symbol, rows in grouped.items()],
        "runningJobs": [job for job in jobs if job["status"] == "running"],
        "failedJobs": _failure_rows(jobs),
        "rejectedJobs": _rejected_rows(jobs),
        "latestLogPath": _latest_log_path(jobs),
    }


def model_search_status_with_lifecycle(filters: dict[str, Any] | None = None) -> dict[str, Any]:
    status = model_search_queue_status(filters)
    for symbol in status["symbols"]:
        for duration in symbol["durations"]:
            duration["paperLive"] = _paper_live_payload(symbol["symbol"], duration["duration"])
    return status


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
        "candidateSearchProgress": status.get("candidateSearchProgress"),
    }


def _model_status(job: dict[str, Any]) -> dict[str, Any]:
    try:
        return model_family_status(job["model_family"], job["symbol"], job["duration"])
    except Exception as exc:
        return {"status": "status_failed", "shadowPredictionReady": False, "shadowPredictionBlockedReason": str(exc)}


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


def _latest_log_path(jobs: list[dict[str, Any]]) -> str | None:
    logged = [job for job in jobs if job.get("log_path")]
    if not logged:
        return None
    return sorted(logged, key=lambda item: str(item.get("finished_at") or item.get("started_at") or ""), reverse=True)[0]["log_path"]
