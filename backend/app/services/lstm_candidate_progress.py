from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.lstm_artifacts import artifact_paths, read_json, update_json, write_json
from app.services.lstm_config import LstmTrainingConfig

PROGRESS_FILE = "candidate_search_progress.json"
RECENT_LIMIT = 8


def candidate_progress_path(
    symbol: str,
    duration: str,
    *,
    artifact_root: Path | None = None,
) -> Path:
    return artifact_paths(symbol.strip().upper(), duration, artifact_root).root / PROGRESS_FILE


def read_lstm_candidate_progress(
    symbol: str,
    duration: str,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    path = candidate_progress_path(symbol, duration, artifact_root=artifact_root)
    return read_json(path) or _empty_progress(symbol, duration)


def start_lstm_candidate_progress(
    *,
    symbol: str,
    duration: str,
    profile: str,
    total: int,
    search_space_total: int,
    parallel_workers: int,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    payload = {
        "status": "running",
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "profile": profile,
        "startedAt": now,
        "updatedAt": now,
        "finishedAt": None,
        "total": int(total),
        "completed": 0,
        "percent": _percent(0, total),
        "searchSpaceTotal": int(search_space_total),
        "parallelWorkers": int(parallel_workers),
        "counts": _empty_counts(),
        "latestCompleted": None,
        "recent": [],
    }
    write_json(candidate_progress_path(symbol, duration, artifact_root=artifact_root), payload)
    return payload


def queue_lstm_candidate_progress(
    *,
    symbol: str,
    duration: str,
    profile: str,
    total: int,
    search_space_total: int,
    parallel_workers: int,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    payload = {
        "status": "queued",
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "profile": profile,
        "startedAt": None,
        "updatedAt": now,
        "finishedAt": None,
        "total": int(total),
        "completed": 0,
        "percent": _percent(0, total),
        "searchSpaceTotal": int(search_space_total),
        "parallelWorkers": int(parallel_workers),
        "counts": _empty_counts(),
        "latestCompleted": None,
        "recent": [],
    }
    write_json(candidate_progress_path(symbol, duration, artifact_root=artifact_root), payload)
    return payload


def complete_lstm_candidate_progress(
    *,
    config: LstmTrainingConfig,
    profile: str,
    report: dict[str, Any],
    completed: int,
    total: int,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    latest = _candidate_payload(config, profile, report)
    path = candidate_progress_path(config.symbol, config.duration, artifact_root=artifact_root)

    def _updater(payload: dict[str, Any] | None) -> dict[str, Any]:
        current = payload or _empty_progress(config.symbol, config.duration)
        recent = [latest, *list(current.get("recent") or [])][:RECENT_LIMIT]
        return {
            **current,
            "status": "running",
            "updatedAt": _utc_now(),
            "total": int(total),
            "completed": int(completed),
            "percent": _percent(completed, total),
            "counts": _updated_counts(current.get("counts") or {}, report),
            "latestCompleted": latest,
            "recent": recent,
        }

    return update_json(path, _updater)


def finish_lstm_candidate_progress(
    *,
    symbol: str,
    duration: str,
    status: str,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    path = candidate_progress_path(symbol, duration, artifact_root=artifact_root)

    def _updater(payload: dict[str, Any] | None) -> dict[str, Any]:
        current = payload or _empty_progress(symbol, duration)
        completed = int(current.get("completed") or 0)
        total = int(current.get("total") or completed)
        now = _utc_now()
        return {
            **current,
            "status": status,
            "updatedAt": now,
            "finishedAt": now,
            "completed": completed,
            "total": total,
            "percent": _percent(completed, total),
        }

    return update_json(path, _updater)


def _candidate_payload(config: LstmTrainingConfig, profile: str, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "completedAt": _utc_now(),
        "profile": profile,
        "status": report.get("status") or "failed",
        "candidateStatus": report.get("candidateStatus"),
        "modelVersion": report.get("modelVersion"),
        "validationFailureReason": report.get("validationFailureReason"),
        "config": {
            "featureWindow": config.feature_window,
            "minMoveBps": config.min_move_bps,
            "epochs": config.epochs,
            "seed": config.seed,
        },
        "validation": _metric_payload(report.get("validation") or {}),
        "test": _metric_payload(report.get("test") or {}),
    }


def _metric_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "winRate": metrics.get("winRate"),
        "profitFactor": metrics.get("profitFactor"),
        "sampleCount": metrics.get("sampleCount"),
    }


def _updated_counts(counts: dict[str, Any], report: dict[str, Any]) -> dict[str, int]:
    status = str(report.get("status") or "failed")
    selected = {**_empty_counts(), **{key: int(value or 0) for key, value in counts.items()}}
    key = _count_key(status)
    selected[key] = selected.get(key, 0) + 1
    return selected


def _count_key(status: str) -> str:
    if status in {"trade_active", "trained"}:
        return "tradeActive"
    if status == "shadow_active":
        return "shadowActive"
    if status == "validation_failed":
        return "validationFailed"
    if status == "insufficient_samples":
        return "insufficientSamples"
    return "failed"


def _empty_progress(symbol: str, duration: str) -> dict[str, Any]:
    return {
        "status": "idle",
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "total": 0,
        "completed": 0,
        "percent": 0.0,
        "counts": _empty_counts(),
        "latestCompleted": None,
        "recent": [],
    }


def _empty_counts() -> dict[str, int]:
    return {
        "tradeActive": 0,
        "shadowActive": 0,
        "validationFailed": 0,
        "insufficientSamples": 0,
        "failed": 0,
    }


def _percent(completed: int, total: int) -> float:
    return 0.0 if total <= 0 else round(min(completed / total, 1.0), 4)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
