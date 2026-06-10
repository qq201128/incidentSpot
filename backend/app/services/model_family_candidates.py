from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.lstm_artifacts import artifact_paths, read_json, update_json, write_json
from app.services.model_family_config import ModelFamilyTrainingConfig, normalize_model_family
from app.services.model_family_candidate_summary import model_candidate_summary
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

def attempted_model_search_keys(
    family: str,
    symbol: str,
    duration: str,
    profile: str | None = None,
    *,
    artifact_root: Path | None = None,
):
    library = read_model_candidate_library(family, symbol, duration, artifact_root=artifact_root)
    return frozenset(
        str(row.get("searchKey"))
        for row in library["records"]
        if row.get("searchKey") and row.get("status") != "failed" and _profile_matches(row, profile)
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
    return model_candidate_summary(records)

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
    search_space_total: int | None = None,
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    artifact_root: Path | None = None,
) -> dict:
    payload = _progress_payload(
        family,
        symbol,
        duration,
        profile=profile,
        status="queued",
        total=total,
        search_space_total=search_space_total,
        parallel_workers=parallel_workers,
    )
    write_json(candidate_progress_path(family, symbol, duration, artifact_root=artifact_root), payload)
    return payload

def start_model_candidate_progress(
    family: str,
    *,
    symbol: str,
    duration: str,
    profile: str,
    total: int,
    search_space_total: int | None = None,
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    artifact_root: Path | None = None,
) -> dict:
    payload = _progress_payload(
        family,
        symbol,
        duration,
        profile=profile,
        status="running",
        total=total,
        search_space_total=search_space_total,
        parallel_workers=parallel_workers,
    )
    write_json(candidate_progress_path(family, symbol, duration, artifact_root=artifact_root), payload)
    return payload

def complete_model_candidate_progress(
    config,
    *,
    profile: str,
    report: dict,
    completed: int,
    total: int,
    artifact_root: Path | None = None,
) -> dict:
    latest = _candidate_payload(config, profile, report)
    path = candidate_progress_path(config.family, config.symbol, config.duration, artifact_root=artifact_root)

    def _updater(payload: dict[str, Any] | None) -> dict[str, Any]:
        current = payload or _empty_progress(config.family, config.symbol, config.duration)
        search_total = _search_total(current, total)
        library_completed = _library_completed_count(
            config.family,
            config.symbol,
            config.duration,
            profile=profile,
            artifact_root=artifact_root,
        )
        recent = [latest, *list(current.get("recent") or [])][:RECENT_LIMIT]
        return {
            **current,
            "status": "running",
            "updatedAt": _utc_now(),
            "completed": library_completed,
            "total": search_total,
            "searchSpaceTotal": search_total,
            "percent": _percent(library_completed, search_total),
            "stageEvaluationCompleted": int(completed),
            "stageEvaluationTotal": int(max(total, completed)),
            "counts": _updated_counts(current.get("counts") or {}, report),
            "latestCompleted": latest,
            "recent": recent,
        }

    return update_json(path, _updater)

def finish_model_candidate_progress(
    family: str, *, symbol: str, duration: str, status: str, failure: dict[str, Any] | None = None
) -> dict:
    path = candidate_progress_path(family, symbol, duration)

    def _updater(payload: dict[str, Any] | None) -> dict[str, Any]:
        current = payload or _empty_progress(family, symbol, duration)
        now = _utc_now()
        next_payload = {**current, "status": status, "updatedAt": now, "finishedAt": now}
        if failure is not None:
            return {**next_payload, "lastFailure": failure}
        if status != "failed":
            return {**next_payload, "lastFailure": None}
        return next_payload

    return update_json(path, _updater)

def finish_model_candidate_progress_from_library(
    family: str,
    *,
    symbol: str,
    duration: str,
    profile: str,
    parallel_workers: int,
    status: str,
    artifact_root: Path | None = None,
) -> dict:
    selected = normalize_model_family(family)
    path = candidate_progress_path(selected, symbol, duration, artifact_root=artifact_root)
    records = _records_for_profile(
        read_model_candidate_library(selected, symbol, duration, artifact_root=artifact_root)["records"],
        profile,
    )
    search_space_total = model_search_space_size(selected)
    total = max(search_space_total, len(records))

    def _updater(payload: dict[str, Any] | None) -> dict[str, Any]:
        current = payload or _empty_progress(selected, symbol, duration)
        now = _utc_now()
        recent = [_progress_record(row) for row in records[-RECENT_LIMIT:]][::-1]
        completed = len(records)
        return {
            **current,
            "status": status, "profile": profile, "updatedAt": now, "finishedAt": now,
            "completed": completed, "total": total, "searchSpaceTotal": search_space_total,
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
        "failure": report.get("failure"),
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

def _progress_payload(
    family: str,
    symbol: str,
    duration: str,
    *,
    profile: str,
    status: str,
    total: int,
    search_space_total: int | None,
    parallel_workers: int,
) -> dict:
    now = _utc_now()
    search_total = int(search_space_total) if search_space_total is not None else int(total)
    return {
        **_empty_progress(family, symbol, duration),
        "status": status,
        "profile": profile,
        "startedAt": now if status == "running" else None,
        "updatedAt": now,
        "total": search_total,
        "searchSpaceTotal": search_total,
        "parallelWorkers": int(parallel_workers),
        "stageEvaluationCompleted": 0,
        "stageEvaluationTotal": int(total),
    }

def _search_total(progress: dict[str, Any], fallback_total: int) -> int:
    raw = progress.get("searchSpaceTotal") or progress.get("total") or fallback_total
    return int(raw)

def _library_completed_count(
    family: str,
    symbol: str,
    duration: str,
    *,
    profile: str | None = None,
    artifact_root: Path | None = None,
) -> int:
    records = read_model_candidate_library(family, symbol, duration, artifact_root=artifact_root)["records"]
    return len(_records_for_profile(records, profile))

def _records_for_profile(records: list[dict], profile: str | None) -> list[dict]:
    return [record for record in records if _profile_matches(record, profile)]

def _profile_matches(record: dict, profile: str | None) -> bool:
    return profile is None or record.get("profile") == profile

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
