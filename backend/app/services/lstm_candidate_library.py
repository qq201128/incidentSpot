from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.lstm_artifacts import artifact_paths, read_json, write_json
from app.services.lstm_candidate_keys import search_key_for_config
from app.services.lstm_config import LstmTrainingConfig

CANDIDATE_LIBRARY_FILE = "candidate_library.json"


def candidate_library_path(
    symbol: str,
    duration: str,
    *,
    artifact_root: Path | None = None,
) -> Path:
    return artifact_paths(symbol.strip().upper(), duration, artifact_root).root / CANDIDATE_LIBRARY_FILE


def read_lstm_candidate_library(
    symbol: str,
    duration: str,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    path = candidate_library_path(symbol, duration, artifact_root=artifact_root)
    payload = read_json(path) if path.exists() else None
    if payload is None:
        return _empty_library(symbol, duration)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"LSTM candidate library records must be a list: {path}")
    return payload


def attempted_search_keys(
    symbol: str,
    duration: str,
    *,
    artifact_root: Path | None = None,
) -> frozenset[str]:
    library = read_lstm_candidate_library(symbol, duration, artifact_root=artifact_root)
    return frozenset(str(row.get("searchKey")) for row in library["records"] if row.get("searchKey"))


def record_lstm_candidate(
    config: LstmTrainingConfig,
    profile: str,
    report: dict[str, Any],
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    library = read_lstm_candidate_library(config.symbol, config.duration, artifact_root=artifact_root)
    record = _candidate_record(config, profile, report)
    records = _replace_record(library["records"], record)
    payload = {
        "symbol": config.symbol.strip().upper(),
        "duration": config.duration,
        "updatedAt": _utc_now(),
        "total": len(records),
        "records": records,
    }
    write_json(candidate_library_path(config.symbol, config.duration, artifact_root=artifact_root), payload)
    return record


def lstm_candidate_library_summary(
    symbol: str,
    duration: str,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    library = read_lstm_candidate_library(symbol, duration, artifact_root=artifact_root)
    records = list(library["records"])
    return {
        "total": len(records),
        "latest": records[-1] if records else None,
        "bestTradeCandidate": _best_record(records, "trade_active"),
        "bestShadowCandidate": _best_record(records, "shadow_active"),
    }


def _candidate_record(
    config: LstmTrainingConfig,
    profile: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    status = str(report.get("status") or "failed")
    return {
        "recordedAt": _utc_now(),
        "searchKey": search_key_for_config(config, profile),
        "profile": profile,
        "config": _config_payload(config),
        "status": status,
        "candidateStatus": report.get("candidateStatus"),
        "promotionReason": report.get("promotionReason"),
        "modelVersion": report.get("modelVersion"),
        "trainedAt": report.get("trainedAt"),
        "sampleCounts": report.get("sampleCounts") or {},
        "selectedConfidenceThreshold": report.get("selectedConfidenceThreshold"),
        "validationFailureReason": report.get("validationFailureReason"),
        "validationGate": report.get("validationGate") or {},
        "validation": _metric_summary(report.get("validation") or {}),
        "test": _metric_summary(report.get("test") or {}),
    }


def _config_payload(config: LstmTrainingConfig) -> dict[str, Any]:
    return {
        "symbol": config.symbol.strip().upper(),
        "duration": config.duration,
        "featureWindow": config.feature_window,
        "epochs": config.epochs,
        "minMoveBps": config.min_move_bps,
        "seed": config.seed,
        "horizonMinutes": config.horizon_minutes,
    }


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "winRate": metrics.get("winRate"),
        "profitFactor": metrics.get("profitFactor"),
        "avgReturn": metrics.get("avgReturn"),
        "sampleCount": metrics.get("sampleCount"),
        "confidenceThresholds": metrics.get("confidenceThresholds") or [],
    }


def _replace_record(records: list[dict[str, Any]], record: dict[str, Any]) -> list[dict[str, Any]]:
    key = record["searchKey"]
    return [row for row in records if row.get("searchKey") != key] + [record]


def _best_record(records: list[dict[str, Any]], status: str) -> dict[str, Any] | None:
    selected = [row for row in records if row.get("status") == status]
    if not selected:
        return None
    return max(selected, key=_candidate_score)


def _candidate_score(record: dict[str, Any]) -> tuple[float, float, int]:
    validation = record.get("validation") or {}
    test = record.get("test") or {}
    win_rate = min(float(validation.get("winRate") or 0.0), float(test.get("winRate") or 0.0))
    profit = min(float(validation.get("profitFactor") or 0.0), float(test.get("profitFactor") or 0.0))
    samples = int((record.get("sampleCounts") or {}).get("test") or 0)
    return win_rate, profit, samples


def _empty_library(symbol: str, duration: str) -> dict[str, Any]:
    return {"symbol": symbol.strip().upper(), "duration": duration, "updatedAt": None, "total": 0, "records": []}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
