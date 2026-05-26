from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.lstm_artifacts import artifact_paths, read_json, update_json, write_json
from app.services.model_family_config import ModelFamilyTrainingConfig, normalize_model_family
from app.services.model_family_search_rules import (
    DEFAULT_PARALLEL_WORKERS,
    model_family_search_grid,
)

RECENT_LIMIT = 8

def candidate_library_path(family: str, symbol: str, duration: str, *, artifact_root: Path | None = None) -> Path:
    selected = normalize_model_family(family)
    return artifact_paths(symbol, duration, artifact_root, family=selected).root / f"{selected}_candidate_library.json"

def candidate_progress_path(family: str, symbol: str, duration: str, *, artifact_root: Path | None = None) -> Path:
    selected = normalize_model_family(family)
    return artifact_paths(symbol, duration, artifact_root, family=selected).root / f"{selected}_candidate_search_progress.json"

def read_model_candidate_library(family: str, symbol: str, duration: str, *, artifact_root: Path | None = None) -> dict:
    path = candidate_library_path(family, symbol, duration, artifact_root=artifact_root)
    payload = read_json(path) if path.exists() else None
    if payload is None:
        return _empty_library(family, symbol, duration)
    if not isinstance(payload.get("records"), list):
        raise ValueError(f"{family} candidate library records must be a list: {path}")
    return payload

def attempted_model_search_keys(family: str, symbol: str, duration: str, *, artifact_root: Path | None = None):
    library = read_model_candidate_library(family, symbol, duration, artifact_root=artifact_root)
    return frozenset(
        str(row.get("searchKey"))
        for row in library["records"]
        if row.get("searchKey") and row.get("status") != "failed"
    )

def record_model_candidate(config: ModelFamilyTrainingConfig, profile: str, report: dict, *, artifact_root=None) -> dict:
    path = candidate_library_path(config.family, config.symbol, config.duration, artifact_root=artifact_root)
    record = _candidate_record(config, profile, report)

    def _updater(payload: dict[str, Any] | None) -> dict[str, Any]:
        library = payload or _empty_library(config.family, config.symbol, config.duration)
        records = library.get("records")
        if not isinstance(records, list):
            raise ValueError(f"{config.family} candidate library records must be a list: {path}")
        next_records = [row for row in records if row.get("searchKey") != record["searchKey"]] + [record]
        return {**library, "updatedAt": _utc_now(), "total": len(next_records), "records": next_records}

    update_json(path, _updater)
    return record

def model_candidate_library_summary(family: str, symbol: str, duration: str, *, artifact_root=None) -> dict:
    records = list(read_model_candidate_library(family, symbol, duration, artifact_root=artifact_root)["records"])
    return {
        "total": len(records),
        "latest": records[-1] if records else None,
        "bestTradeCandidate": _best_record(records, "trade_active"),
        "bestShadowCandidate": _best_record(records, "shadow_active"),
    }

def read_model_candidate_progress(family: str, symbol: str, duration: str, *, artifact_root=None) -> dict:
    path = candidate_progress_path(family, symbol, duration, artifact_root=artifact_root)
    return read_json(path) or _empty_progress(family, symbol, duration)

def queue_model_candidate_progress(
    family: str,
    *,
    symbol: str,
    duration: str,
    profile: str,
    total: int,
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
) -> dict:
    payload = _progress_payload(family, symbol, duration, profile, "queued", total, parallel_workers)
    write_json(candidate_progress_path(family, symbol, duration), payload)
    return payload

def start_model_candidate_progress(
    family: str,
    *,
    symbol: str,
    duration: str,
    profile: str,
    total: int,
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
) -> dict:
    payload = _progress_payload(family, symbol, duration, profile, "running", total, parallel_workers)
    write_json(candidate_progress_path(family, symbol, duration), payload)
    return payload

def complete_model_candidate_progress(config, profile: str, report: dict, completed: int, total: int) -> dict:
    latest = _candidate_payload(config, profile, report)
    path = candidate_progress_path(config.family, config.symbol, config.duration)

    def _updater(payload: dict[str, Any] | None) -> dict[str, Any]:
        current = payload or _empty_progress(config.family, config.symbol, config.duration)
        recent = [latest, *list(current.get("recent") or [])][:RECENT_LIMIT]
        return {
            **current,
            "status": "running",
            "updatedAt": _utc_now(),
            "completed": int(completed),
            "total": int(total),
            "percent": _percent(completed, total),
            "counts": _updated_counts(current.get("counts") or {}, report),
            "latestCompleted": latest,
            "recent": recent,
        }

    return update_json(path, _updater)

def finish_model_candidate_progress(family: str, *, symbol: str, duration: str, status: str) -> dict:
    path = candidate_progress_path(family, symbol, duration)

    def _updater(payload: dict[str, Any] | None) -> dict[str, Any]:
        current = payload or _empty_progress(family, symbol, duration)
        now = _utc_now()
        return {**current, "status": status, "updatedAt": now, "finishedAt": now}

    return update_json(path, _updater)

def finish_model_candidate_progress_from_library(family: str, *, symbol: str, duration: str, profile: str, parallel_workers: int, status: str) -> dict:
    selected = normalize_model_family(family)
    path = candidate_progress_path(selected, symbol, duration)
    records = list(read_model_candidate_library(selected, symbol, duration)["records"])
    total = max(model_search_space_size(selected), len(records))

    def _updater(payload: dict[str, Any] | None) -> dict[str, Any]:
        current = payload or _empty_progress(selected, symbol, duration)
        now = _utc_now()
        recent = [_progress_record(row) for row in records[-RECENT_LIMIT:]][::-1]
        completed = len(records)
        return {
            **current,
            "status": status, "profile": profile, "updatedAt": now, "finishedAt": now,
            "completed": completed, "total": total, "searchSpaceTotal": total,
            "percent": _percent(completed, total), "parallelWorkers": int(parallel_workers),
            "counts": _counts_from_records(records),
            "latestCompleted": _progress_record(records[-1]) if records else None,
            "recent": recent,
        }

    return update_json(path, _updater)

def next_model_candidate_configs(config: ModelFamilyTrainingConfig, profile: str, attempted: frozenset[str]) -> list:
    candidates = [_candidate_config(config, overrides) for overrides in model_family_search_grid(config.family)]
    return [item for item in candidates if model_search_key(item, profile) not in attempted]

def model_search_space_size(family: str) -> int:
    return len(model_family_search_grid(normalize_model_family(family)))

def model_search_key(config: ModelFamilyTrainingConfig, profile: str) -> str:
    params = ",".join(f"{key}={config.params[key]}" for key in sorted(config.params))
    return (
        f"{config.family}|{profile}|{config.symbol}|{config.duration}|w={config.feature_window}|"
        f"m={config.min_move_bps:g}|e={config.epochs}|s={config.seed}|{params}"
    )

def _candidate_config(config: ModelFamilyTrainingConfig, overrides: dict) -> ModelFamilyTrainingConfig:
    params = dict(overrides.get("params") or config.params)
    values = {key: value for key, value in overrides.items() if key != "params"}
    return replace(config, **values, params=params)

def _candidate_record(config, profile: str, report: dict) -> dict:
    return {
        "recordedAt": _utc_now(),
        "searchKey": model_search_key(config, profile),
        "profile": profile,
        "config": _config_payload(config),
        "status": report.get("status") or "failed",
        "candidateStatus": report.get("candidateStatus"),
        "searchStage": report.get("searchStage"),
        "advancedToNextStage": report.get("advancedToNextStage"),
        "eliminationReason": report.get("eliminationReason"),
        "modelVersion": report.get("modelVersion"),
        "validationFailureReason": report.get("validationFailureReason"),
        "validation": _metric_summary(report.get("validation") or {}),
        "test": _metric_summary(report.get("test") or {}),
    }

def _candidate_payload(config, profile: str, report: dict) -> dict:
    return {**_candidate_record(config, profile, report), "completedAt": _utc_now()}

def _config_payload(config) -> dict:
    return {
        "family": config.family,
        "symbol": config.symbol,
        "duration": config.duration,
        "featureWindow": config.feature_window,
        "minMoveBps": config.min_move_bps,
        "epochs": config.epochs,
        "seed": config.seed,
        "params": config.params,
    }

def _metric_summary(metrics: dict) -> dict:
    return {
        "winRate": metrics.get("winRate"),
        "profitFactor": metrics.get("profitFactor"),
        "sampleCount": metrics.get("sampleCount"),
    }

def _best_record(records: list[dict], status: str) -> dict | None:
    selected = [row for row in records if row.get("status") == status]
    return max(selected, key=_candidate_score) if selected else None

def _candidate_score(record: dict) -> tuple[float, float, int]:
    validation = record.get("validation") or {}
    test = record.get("test") or {}
    return (
        min(float(validation.get("winRate") or 0.0), float(test.get("winRate") or 0.0)),
        min(float(validation.get("profitFactor") or 0.0), float(test.get("profitFactor") or 0.0)),
        int(test.get("sampleCount") or 0),
    )

def _progress_payload(
    family: str,
    symbol: str,
    duration: str,
    profile: str,
    status: str,
    total: int,
    parallel_workers: int,
) -> dict:
    now = _utc_now()
    return {
        **_empty_progress(family, symbol, duration),
        "status": status,
        "profile": profile,
        "startedAt": now if status == "running" else None,
        "updatedAt": now,
        "total": int(total),
        "searchSpaceTotal": int(total),
        "parallelWorkers": int(parallel_workers),
    }

def _updated_counts(counts: dict, report: dict) -> dict[str, int]:
    selected = {**_empty_counts(), **{key: int(value or 0) for key, value in counts.items()}}
    selected[_count_key(report.get("status"))] += 1
    return selected

def _counts_from_records(records: list[dict]) -> dict[str, int]:
    counts = _empty_counts()
    for record in records:
        counts[_count_key(record.get("status"))] += 1
    return counts

def _count_key(status: Any) -> str:
    labels = {"trade_active": "tradeActive", "trained": "tradeActive", "shadow_active": "shadowActive",
              "initial_baseline": "initialBaseline", "validation_failed": "validationFailed",
              "insufficient_samples": "insufficientSamples"}
    return labels.get(str(status or "failed"), "failed")

def _progress_record(record: dict) -> dict:
    return {**record, "completedAt": record.get("recordedAt")}

def _empty_library(family: str, symbol: str, duration: str) -> dict:
    return {"modelFamily": family, "symbol": symbol.strip().upper(), "duration": duration, "updatedAt": None, "total": 0, "records": []}

def _empty_progress(family: str, symbol: str, duration: str) -> dict:
    return {"status": "idle", "modelFamily": family, "symbol": symbol.strip().upper(), "duration": duration,
            "total": 0, "completed": 0, "percent": 0.0, "counts": _empty_counts(), "latestCompleted": None, "recent": []}

def _empty_counts() -> dict[str, int]:
    return {
        "tradeActive": 0,
        "shadowActive": 0,
        "initialBaseline": 0,
        "validationFailed": 0,
        "insufficientSamples": 0,
        "failed": 0,
    }

def _percent(completed: int, total: int) -> float:
    return 0.0 if total <= 0 else round(min(completed / total, 1.0), 4)

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
