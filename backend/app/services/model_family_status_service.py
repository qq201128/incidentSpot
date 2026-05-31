from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.lstm_artifacts import artifact_paths, read_json, required_artifacts_exist
from app.services.lstm_combo_snapshot import current_combo_snapshot
from app.services.lstm_lifecycle import (
    LSTM_STATUS_LEGACY_TRAINED,
    LSTM_STATUS_TRADE_ACTIVE,
    shadow_predictable_status,
    trade_active_status,
)
from app.services.lstm_status_service import validation_gate_payload, validation_threshold
from app.services.lstm_torch_backend import torch_availability
from app.services.model_family_config import (
    JOBLIB_MODEL_FAMILIES,
    TORCH_MODEL_FAMILIES,
    model_family_strategy_key,
    normalize_model_family,
)
from app.services.model_family_paper_live_policy import model_status_policy_payload
from app.services.model_family_search_rules import model_family_training_rules
from app.services.model_family_candidates import (
    model_candidate_library_summary,
    read_model_candidate_progress,
)
from app.services.model_search_job_store import list_model_search_jobs

_DEPENDENCY_STATUS_CACHE: dict[str, dict[str, Any]] = {}


def model_family_status(
    family: str,
    symbol: str,
    duration: str,
    *,
    artifact_root: Path | None = None,
    current_combo_snapshot: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected = normalize_model_family(family)
    sym = symbol.strip().upper()
    paths = artifact_paths(sym, duration, artifact_root, family=selected)
    raw_status = read_json(paths.status) or _untrained_status(selected, sym, duration)
    attempt = read_json(paths.attempt) or {}
    report = read_json(paths.report) or {}
    version = read_json(paths.version) or {}
    artifacts_ready = required_artifacts_exist(paths)
    status = _active_status(selected, sym, duration, raw_status, version, report, artifacts_ready)
    dependency = _dependency_status(selected)
    snapshot = _combo_snapshot_status(sym, duration, paths.features, current=current_combo_snapshot)
    ready_reason = _shadow_ready_reason(status, version, report, artifacts_ready, dependency["available"])
    trade_reason = _trade_ready_reason(status, version, report, artifacts_ready, dependency["available"])
    gate = validation_gate_payload(version, report)
    progress = _candidate_search_progress(selected, sym, duration, artifact_root)
    return {
        **status,
        "modelFamily": selected,
        "strategyKey": model_family_strategy_key(selected, duration),
        "modelVersion": version.get("modelVersion") or report.get("modelVersion"),
        "trainedAt": version.get("trainedAt") or report.get("trainedAt"),
        "sampleCounts": report.get("sampleCounts") or {},
        "testAccuracy": (report.get("test") or {}).get("accuracy"),
        "testWinRate": (report.get("test") or {}).get("winRate"),
        "validationWinRate": (report.get("validation") or {}).get("winRate"),
        "validationGate": gate,
        "selectedConfidenceThreshold": version.get("selectedConfidenceThreshold") or report.get("selectedConfidenceThreshold"),
        "activeModelStatus": status.get("status"),
        "lastAttemptStatus": attempt.get("status"),
        "lastTrainingAttempt": attempt,
        "candidateStatus": attempt.get("candidateStatus") or version.get("candidateStatus"),
        "promotionReason": attempt.get("promotionReason") or version.get("promotionReason"),
        "validationFailureReason": _validation_failure_reason(raw_status, attempt, report),
        "artifactsReady": artifacts_ready,
        "dependencyAvailable": dependency["available"],
        "dependencyStatus": dependency,
        "torchAvailable": dependency["available"] if selected in TORCH_MODEL_FAMILIES else True,
        "torchStatus": dependency if selected in TORCH_MODEL_FAMILIES else {"available": True},
        "comboSnapshotMatches": snapshot["matches"],
        "comboSnapshotReason": snapshot["reason"],
        "comboSnapshotCurrent": snapshot["current"],
        "comboSnapshotTrained": snapshot["trained"],
        "candidateLibrary": model_candidate_library_summary(selected, sym, duration, artifact_root=artifact_root),
        "candidateSearchProgress": progress,
        "trainingRules": model_family_training_rules(selected),
        "shadowPredictionReady": ready_reason == "passed",
        "shadowPredictionBlockedReason": ready_reason,
        "tradePredictionReady": trade_reason == "passed",
        "tradePredictionBlockedReason": trade_reason,
        **model_status_policy_payload(status.get("status"), gate),
    }


def _candidate_search_progress(
    family: str,
    symbol: str,
    duration: str,
    artifact_root: Path | None,
) -> dict[str, Any]:
    progress = read_model_candidate_progress(family, symbol, duration, artifact_root=artifact_root)
    if artifact_root is not None or progress.get("status") in {"queued", "running"}:
        return progress
    queued = _latest_active_search_job(family, symbol, duration)
    return _queued_progress_from_job(family, queued, progress) if queued else progress


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


def active_model_family_status(family: str, symbol: str, duration: str, *, artifact_root: Path | None = None) -> dict:
    selected = normalize_model_family(family)
    sym = symbol.strip().upper()
    paths = artifact_paths(sym, duration, artifact_root, family=selected)
    return _active_status(
        selected,
        sym,
        duration,
        read_json(paths.status) or _untrained_status(selected, sym, duration),
        read_json(paths.version) or {},
        read_json(paths.report) or {},
        required_artifacts_exist(paths),
    )


def model_validation_block_reason(status: dict[str, Any], version: dict[str, Any], report: dict[str, Any]) -> str:
    active_status = status.get("status")
    if active_status == "shadow_active":
        gate = validation_gate_payload(version, report)
        return str(gate.get("reason") or "validation_gate_failed")
    if not trade_active_status(active_status):
        return status.get("reason") or f"model_status_{status.get('status') or 'unknown'}"
    gate = validation_gate_payload(version, report)
    if not gate:
        return "validation_gate_missing"
    if gate.get("status") != "passed":
        return str(gate.get("reason") or "validation_gate_failed")
    if validation_threshold(version, report, gate) is None:
        return "validation_confidence_threshold_missing"
    return "passed"


def _active_status(family: str, symbol: str, duration: str, status, version, report, artifacts_ready: bool) -> dict:
    if shadow_predictable_status(status.get("status")):
        return status
    if not _active_artifacts_pass_validation(status, version, report, artifacts_ready):
        return status
    return {
        "status": LSTM_STATUS_TRADE_ACTIVE,
        "modelFamily": family,
        "symbol": status.get("symbol") or symbol,
        "duration": status.get("duration") or duration,
        "featureWindow": report.get("featureWindow"),
        "minMoveBps": version.get("minMoveBps") or report.get("minMoveBps"),
        "updatedAt": version.get("trainedAt") or report.get("trainedAt"),
    }


def _active_artifacts_pass_validation(status, version, report, artifacts_ready: bool) -> bool:
    return (
        status.get("status") != "untrained"
        and artifacts_ready
        and model_validation_block_reason({"status": LSTM_STATUS_TRADE_ACTIVE}, version, report) == "passed"
    )


def _combo_snapshot_status(
    symbol: str,
    duration: str,
    features_path: Path,
    *,
    current: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    current_list = current if current is not None else current_combo_snapshot(symbol, duration)
    features = read_json(features_path) or {}
    trained = features.get("comboSnapshot")
    trained = list(trained) if isinstance(trained, list) else []
    if not current_list:
        return _snapshot(False, current_list, trained, "current_combo_snapshot_missing")
    if not trained:
        return _snapshot(False, current_list, trained, "trained_combo_snapshot_missing")
    if current_list != trained:
        return _snapshot(False, current_list, trained, "combo_snapshot_mismatch")
    return _snapshot(True, current_list, trained, "passed")


def _shadow_ready_reason(status, version, report, artifacts_ready: bool, dependency_ready: bool) -> str:
    if not dependency_ready:
        return "dependency_unavailable"
    if not shadow_predictable_status(status.get("status")):
        return status.get("reason") or f"model_status_{status.get('status') or 'unknown'}"
    if not artifacts_ready:
        return "artifacts_incomplete"
    if status.get("status") == LSTM_STATUS_LEGACY_TRAINED:
        return model_validation_block_reason(status, version, report)
    return "passed"


def _trade_ready_reason(status, version, report, artifacts_ready: bool, dependency_ready: bool) -> str:
    if not dependency_ready:
        return "dependency_unavailable"
    if not artifacts_ready:
        return "artifacts_incomplete"
    return model_validation_block_reason(status, version, report)


def _dependency_status(family: str) -> dict[str, Any]:
    cached = _DEPENDENCY_STATUS_CACHE.get(family)
    if cached is not None:
        return cached
    if family in TORCH_MODEL_FAMILIES:
        payload = torch_availability()
    elif family == "xgboost":
        try:
            import xgboost
        except ImportError as exc:
            payload = {"available": False, "error": str(exc)}
        else:
            payload = {"available": True, "version": getattr(xgboost, "__version__", None)}
    elif family == "lightgbm":
        try:
            import lightgbm
        except ImportError as exc:
            payload = {"available": False, "error": str(exc)}
        else:
            payload = {"available": True, "version": getattr(lightgbm, "__version__", None)}
    elif family == "catboost":
        try:
            import catboost
        except ImportError as exc:
            payload = {"available": False, "error": str(exc)}
        else:
            payload = {"available": True, "version": getattr(catboost, "__version__", None)}
    else:
        payload = {"available": family in JOBLIB_MODEL_FAMILIES}
    _DEPENDENCY_STATUS_CACHE[family] = payload
    return payload


def _validation_failure_reason(status, attempt, report) -> str | None:
    for payload in (attempt, status, report):
        reason = payload.get("validationFailureReason") or payload.get("reason")
        if reason:
            return str(reason)
    gate = report.get("validationGate") or {}
    return gate.get("reason") if isinstance(gate, dict) and gate.get("status") != "passed" else None


def _untrained_status(family: str, symbol: str, duration: str) -> dict[str, Any]:
    return {"status": "untrained", "modelFamily": family, "symbol": symbol, "duration": duration, "featureWindow": None}


def _snapshot(matches: bool, current: list, trained: list, reason: str) -> dict[str, Any]:
    return {"matches": matches, "current": current, "trained": trained, "reason": reason}
