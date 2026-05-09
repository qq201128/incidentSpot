from __future__ import annotations

import asyncio
import importlib
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Callable
from zoneinfo import ZoneInfo

from app.services.model_metrics import compare_candidate_meta
from app.services.model_registry import (
    CANDIDATE_DIR,
    SYMBOL,
    ModelSpec,
    VersionRecord,
    activate_version,
    active_meta,
    active_snapshot,
    archive_active_model,
    list_versions,
    model_specs,
    publish_artifacts,
    read_registry,
    record_candidate_version,
    record_run,
    training_params,
    utc_now_iso,
)


logger = logging.getLogger("uvicorn.error")
LOCAL_TZ = ZoneInfo("Asia/Shanghai")

_RUN_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "running": False,
    "currentTrigger": None,
    "startedAt": None,
    "finishedAt": None,
    "nextRunAt": None,
    "lastError": None,
}


@dataclass(frozen=True)
class RunContext:
    run_id: str
    trigger: str
    candidate_root: Path


@dataclass(frozen=True)
class CandidateResult:
    spec: ModelSpec
    candidate_dir: Path
    version_id: str
    metrics: dict[str, Any]
    trigger: str
    decision: dict[str, Any]


async def auto_train_loop(stop_event: asyncio.Event) -> None:
    logger.info("model training loop: scheduled daily at 00:00 Asia/Shanghai")
    while not stop_event.is_set():
        next_run = next_midnight()
        _set_state(nextRunAt=next_run.isoformat())
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_seconds_until(next_run))
            continue
        except asyncio.TimeoutError:
            pass
        if stop_event.is_set():
            break
        try:
            await asyncio.to_thread(run_model_update, "scheduled")
        except Exception:
            logger.exception("scheduled model training failed")


def model_dashboard() -> dict[str, Any]:
    registry = read_registry()
    return {
        "schedule": training_state(),
        "models": [active_snapshot(spec) for spec in model_specs()],
        "versions": list_versions(),
        "lastRun": registry.get("lastRun"),
    }


def training_state() -> dict[str, Any]:
    with _STATE_LOCK:
        return dict(_STATE)


def start_training_task() -> dict[str, Any]:
    if training_state()["running"]:
        raise RuntimeError("model training is already running")
    _set_state(running=True, currentTrigger="manual", startedAt=utc_now_iso(), lastError=None)
    loop = asyncio.get_running_loop()
    loop.create_task(_run_training_task("manual"))
    return training_state()


def activate_model_version(model_key: str, version_id: str) -> dict[str, Any]:
    snapshot = activate_version(model_key, version_id)
    _reload_prediction_cache(model_key)
    return snapshot


def run_model_update(trigger: str) -> dict[str, Any]:
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("model training is already running")
    run_id = _run_id()
    _set_state(running=True, currentTrigger=trigger, startedAt=utc_now_iso(), lastError=None)
    try:
        run = _execute_run(run_id, trigger)
        record_run(run)
        _set_state(running=False, finishedAt=run["finishedAt"], currentTrigger=None)
        return run
    except Exception as exc:
        _record_fatal_run(run_id, trigger, exc)
        raise
    finally:
        _RUN_LOCK.release()


def next_midnight(now: datetime | None = None) -> datetime:
    current = now or datetime.now(LOCAL_TZ)
    target = current.replace(hour=0, minute=0, second=0, microsecond=0)
    if current >= target:
        target += timedelta(days=1)
    return target


async def _run_training_task(trigger: str) -> None:
    try:
        await asyncio.to_thread(run_model_update, trigger)
    except Exception as exc:
        _set_state(running=False, finishedAt=utc_now_iso(), currentTrigger=None, lastError=str(exc))
        logger.exception("model training task failed")


def _execute_run(run_id: str, trigger: str) -> dict[str, Any]:
    context = RunContext(run_id, trigger, CANDIDATE_DIR / run_id)
    results = [_update_single_model(spec, context) for spec in model_specs()]
    return {
        "runId": run_id,
        "trigger": trigger,
        "startedAt": _STATE["startedAt"],
        "finishedAt": utc_now_iso(),
        "status": _run_status(results),
        "results": results,
    }


def _update_single_model(spec: ModelSpec, context: RunContext) -> dict[str, Any]:
    try:
        candidate_dir = context.candidate_root / spec.key
        metrics = _train_candidate(spec, candidate_dir)
        decision = compare_candidate_meta(active_meta(spec), metrics)
        version_id = f"{context.run_id}-{spec.key}"
        result = CandidateResult(spec, candidate_dir, version_id, metrics, context.trigger, decision)
        if decision["approved"]:
            return _publish_candidate(result)
        record_candidate_version(_version_record(result, "rejected"))
        return _model_result(result, "rejected")
    except Exception as exc:
        return {"modelKey": spec.key, "status": "failed", "error": str(exc)}


def _publish_candidate(candidate: CandidateResult) -> dict[str, Any]:
    archive_active_model(candidate.spec, candidate.version_id, candidate.trigger)
    record_candidate_version(_version_record(candidate, "active"))
    publish_artifacts(candidate.candidate_dir, candidate.spec)
    from app.services.model_registry import mark_active

    mark_active(candidate.spec.key, candidate.version_id)
    _reload_prediction_cache(candidate.spec.key)
    return _model_result(candidate, "published")


def _train_candidate(spec: ModelSpec, candidate_dir: Path) -> dict[str, Any]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    params = training_params(spec)
    if spec.family == "enhanced":
        return _train_enhanced(spec, candidate_dir, params)
    return _train_legacy(spec, candidate_dir, params)


def _train_enhanced(spec: ModelSpec, out_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
    module = importlib.import_module("app.services.enhanced_pipeline")
    return _with_model_dir(
        module,
        out_dir,
        lambda: module.train_enhanced(
            SYMBOL,
            spec.duration,
            params["min_move_bps"],
            params["train_window_days"],
            params["trade_confidence_threshold"],
            params["min_trade_gap_minutes"],
        ),
    )


def _train_legacy(spec: ModelSpec, out_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
    module = importlib.import_module("train_10m")
    return _with_model_dir(
        module,
        out_dir,
        lambda: module.train_for_symbol(
            SYMBOL,
            spec.duration,
            params["min_move_bps"],
            params["train_window_days"],
        ),
    )


def _with_model_dir(module: ModuleType, out_dir: Path, action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    previous = module.MODEL_DIR
    module.MODEL_DIR = out_dir
    try:
        return action()
    finally:
        module.MODEL_DIR = previous


def _reload_prediction_cache(model_key: str) -> None:
    spec = next(item for item in model_specs() if item.key == model_key)
    if spec.family == "enhanced":
        from app.services.enhanced_predictor import clear_model_cache
    else:
        from app.services.predict_10m import clear_model_cache
    clear_model_cache(spec.duration)


def _model_result(candidate: CandidateResult, status: str) -> dict[str, Any]:
    return {
        "modelKey": candidate.spec.key,
        "status": status,
        "versionId": candidate.version_id,
        "metrics": candidate.metrics,
        "decision": candidate.decision,
    }


def _version_record(candidate: CandidateResult, status: str) -> VersionRecord:
    return VersionRecord(
        candidate.spec,
        candidate.version_id,
        candidate.candidate_dir,
        status,
        candidate.metrics,
        candidate.trigger,
        candidate.decision,
    )


def _record_fatal_run(run_id: str, trigger: str, exc: Exception) -> None:
    run = {
        "runId": run_id,
        "trigger": trigger,
        "startedAt": _STATE.get("startedAt"),
        "finishedAt": utc_now_iso(),
        "status": "failed",
        "error": str(exc),
        "results": [],
    }
    record_run(run)
    _set_state(running=False, finishedAt=run["finishedAt"], currentTrigger=None, lastError=str(exc))


def _run_status(results: list[dict[str, Any]]) -> str:
    if any(item["status"] == "failed" for item in results):
        return "completed_with_errors"
    if any(item["status"] == "rejected" for item in results):
        return "completed_with_rejections"
    return "completed"


def _set_state(**kwargs: Any) -> None:
    with _STATE_LOCK:
        _STATE.update(kwargs)


def _seconds_until(target: datetime) -> float:
    return max(0.0, (target - datetime.now(LOCAL_TZ)).total_seconds())


def _run_id() -> str:
    return datetime.now(LOCAL_TZ).strftime("%Y%m%d%H%M%S")
