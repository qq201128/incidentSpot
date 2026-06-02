from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.model_family_status_service import model_family_status
from app.services.model_search_job_store import enqueue_model_search_jobs

TRAINED_STATUSES = frozenset({"shadow_active", "trade_active", "initial_baseline", "trained"})


@dataclass(frozen=True)
class ModelSearchTarget:
    symbol: str
    duration: str
    family: str


def enqueue_untrained_model_search_jobs(
    *,
    symbols: tuple[str, ...],
    durations: tuple[str, ...],
    families: tuple[str, ...],
    profile: str,
    reset_existing: bool = False,
    reset_history: bool = False,
    resource: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if reset_history:
        queued = enqueue_model_search_jobs(
            symbols=symbols,
            durations=durations,
            families=families,
            profile=profile,
            reset_existing=reset_existing,
            reset_history=True,
            resource=resource,
        )
        return {**queued, "trainedSkipped": [], "trainedSkippedCount": 0, "statusErrors": []}
    split = split_untrained_targets(
        symbols=symbols,
        durations=durations,
        families=families,
    )
    if not split["untrainedTargets"]:
        return _empty_enqueue_payload(split)
    queued = _enqueue_targets(
        split["untrainedTargets"],
        profile=profile,
        reset_existing=reset_existing,
        reset_history=reset_history,
        resource=resource,
    )
    return {**queued, **_split_payload(split)}


def split_untrained_targets(
    *,
    symbols: tuple[str, ...],
    durations: tuple[str, ...],
    families: tuple[str, ...],
) -> dict[str, list]:
    trained, untrained, status_errors = [], [], []
    for symbol in symbols:
        for duration in durations:
            for family in families:
                target = ModelSearchTarget(symbol.strip().upper(), duration, family)
                try:
                    status = model_family_status(target.family, target.symbol, target.duration)
                except Exception as exc:
                    status_errors.append(_target_payload(target, status="status_error", reason=str(exc)))
                    untrained.append(target)
                    continue
                if is_trained_model_status(status):
                    trained.append(_target_payload(target, status=str(status.get("status") or "trained")))
                    continue
                untrained.append(target)
    return {"trainedTargets": trained, "untrainedTargets": untrained, "statusErrors": status_errors}


def is_trained_model_status(status: dict[str, Any]) -> bool:
    return bool(status.get("shadowPredictionReady")) or str(status.get("activeModelStatus") or status.get("status")) in TRAINED_STATUSES


def _empty_enqueue_payload(split: dict[str, list]) -> dict[str, Any]:
    return {
        "version": "model_search_jobs_v1",
        "realTradingEnabled": False,
        "total": 0,
        "created": 0,
        "existing": 0,
        "reset": 0,
        "jobs": [],
        **_split_payload(split),
    }


def _enqueue_targets(
    targets: list[ModelSearchTarget],
    *,
    profile: str,
    reset_existing: bool,
    reset_history: bool,
    resource: dict[str, Any] | None,
) -> dict[str, Any]:
    payloads = [
        enqueue_model_search_jobs(
            symbols=(target.symbol,),
            durations=(target.duration,),
            families=(target.family,),
            profile=profile,
            reset_existing=reset_existing,
            reset_history=reset_history,
            resource=resource,
        )
        for target in targets
    ]
    jobs = [job for payload in payloads for job in payload["jobs"]]
    return {
        "version": "model_search_jobs_v1",
        "realTradingEnabled": False,
        "total": len(jobs),
        "created": sum(int(payload.get("created") or 0) for payload in payloads),
        "existing": sum(int(payload.get("existing") or 0) for payload in payloads),
        "reset": sum(int(payload.get("reset") or 0) for payload in payloads),
        "jobs": jobs,
    }


def _split_payload(split: dict[str, list]) -> dict[str, Any]:
    return {
        "trainedSkipped": split["trainedTargets"],
        "trainedSkippedCount": len(split["trainedTargets"]),
        "statusErrors": split["statusErrors"],
    }


def _target_payload(target: ModelSearchTarget, *, status: str, reason: str | None = None) -> dict[str, Any]:
    payload = {"symbol": target.symbol, "duration": target.duration, "family": target.family, "status": status}
    return payload if reason is None else {**payload, "reason": reason}
