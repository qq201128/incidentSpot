from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.lstm_artifacts import artifact_paths, read_json, required_artifacts_exist
from app.services.lstm_combo_snapshot import combo_snapshot_status
from app.services.lstm_config import lstm_shadow_strategy_key
from app.services.lstm_torch_backend import torch_availability


def lstm_model_status(symbol: str, duration: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
    sym = symbol.strip().upper()
    paths = artifact_paths(sym, duration, artifact_root)
    raw_status = read_json(paths.status) or _untrained_status(sym, duration)
    attempt = read_json(paths.attempt) or {}
    report = read_json(paths.report) or {}
    version = read_json(paths.version) or {}
    snapshot = combo_snapshot_status(sym, duration, artifact_root=artifact_root)
    artifacts_ready = required_artifacts_exist(paths)
    status = _active_status(sym, duration, raw_status, version, report, artifacts_ready)
    torch_status = torch_availability()
    torch_available = bool(torch_status["available"])
    ready_reason = _shadow_prediction_ready_reason(
        status,
        version,
        report,
        artifacts_ready,
        torch_available,
    )
    return {
        **status,
        "strategyKey": lstm_shadow_strategy_key(duration),
        "modelVersion": version.get("modelVersion") or report.get("modelVersion"),
        "trainedAt": version.get("trainedAt") or report.get("trainedAt"),
        "sampleCounts": report.get("sampleCounts") or {},
        "testAccuracy": (report.get("test") or {}).get("accuracy"),
        "testWinRate": (report.get("test") or {}).get("winRate"),
        "validationGate": version.get("validationGate") or report.get("validationGate"),
        "selectedConfidenceThreshold": (
            version.get("selectedConfidenceThreshold")
            or report.get("selectedConfidenceThreshold")
        ),
        "activeModelStatus": status.get("status"),
        "lastAttemptStatus": attempt.get("status"),
        "lastTrainingAttempt": attempt,
        "validationFailureReason": _validation_failure_reason(raw_status, attempt, report),
        "artifactsReady": artifacts_ready,
        "torchAvailable": torch_available,
        "torchStatus": torch_status,
        "comboSnapshotMatches": snapshot["matches"],
        "comboSnapshotReason": snapshot["reason"],
        "comboSnapshotCurrent": snapshot["current"],
        "comboSnapshotTrained": snapshot["trained"],
        "shadowPredictionReady": ready_reason == "passed",
        "shadowPredictionBlockedReason": ready_reason,
    }


def is_lstm_shadow_ready(symbol: str, duration: str, *, artifact_root: Path | None = None) -> bool:
    status = lstm_model_status(symbol, duration, artifact_root=artifact_root)
    return bool(status.get("shadowPredictionReady"))


def active_lstm_status(
    symbol: str,
    duration: str,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    sym = symbol.strip().upper()
    paths = artifact_paths(sym, duration, artifact_root)
    return _active_status(
        sym,
        duration,
        read_json(paths.status) or _untrained_status(sym, duration),
        read_json(paths.version) or {},
        read_json(paths.report) or {},
        required_artifacts_exist(paths),
    )


def lstm_validation_block_reason(
    status: dict[str, Any],
    version: dict[str, Any],
    report: dict[str, Any],
) -> str:
    if status.get("status") != "trained":
        return status.get("reason") or f"model_status_{status.get('status') or 'unknown'}"
    gate = validation_gate_payload(version, report)
    if not gate:
        return "validation_gate_missing"
    if gate.get("status") != "passed":
        return str(gate.get("reason") or "validation_gate_failed")
    threshold = validation_threshold(version, report, gate)
    if threshold is None:
        return "validation_confidence_threshold_missing"
    return "passed"


def validation_gate_payload(version: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    gate = version.get("validationGate")
    if isinstance(gate, dict):
        return gate
    gate = report.get("validationGate")
    return gate if isinstance(gate, dict) else {}


def validation_threshold(
    version: dict[str, Any],
    report: dict[str, Any],
    gate: dict[str, Any],
) -> Any:
    threshold = version.get("selectedConfidenceThreshold")
    if threshold is not None:
        return threshold
    threshold = report.get("selectedConfidenceThreshold")
    if threshold is not None:
        return threshold
    return gate.get("minConfidence")


def _untrained_status(symbol: str, duration: str) -> dict[str, Any]:
    return {
        "status": "untrained",
        "symbol": symbol,
        "duration": duration,
        "featureWindow": None,
        "updatedAt": None,
    }


def _active_status(
    symbol: str,
    duration: str,
    status: dict[str, Any],
    version: dict[str, Any],
    report: dict[str, Any],
    artifacts_ready: bool,
) -> dict[str, Any]:
    if status.get("status") == "trained":
        return status
    if not _active_artifacts_pass_validation(status, version, report, artifacts_ready):
        return status
    return {
        "status": "trained",
        "symbol": status.get("symbol") or symbol.strip().upper(),
        "duration": status.get("duration") or duration,
        "featureWindow": report.get("featureWindow"),
        "minMoveBps": version.get("minMoveBps") or report.get("minMoveBps"),
        "updatedAt": version.get("trainedAt") or report.get("trainedAt"),
    }


def _active_artifacts_pass_validation(
    status: dict[str, Any],
    version: dict[str, Any],
    report: dict[str, Any],
    artifacts_ready: bool,
) -> bool:
    return (
        status.get("status") != "untrained"
        and artifacts_ready
        and lstm_validation_block_reason({"status": "trained"}, version, report) == "passed"
    )


def _validation_failure_reason(
    status: dict[str, Any],
    attempt: dict[str, Any],
    report: dict[str, Any],
) -> str | None:
    for payload in (attempt, status, report):
        reason = payload.get("validationFailureReason") or payload.get("reason")
        if reason:
            return str(reason)
    gate = report.get("validationGate") or {}
    if isinstance(gate, dict) and gate.get("status") != "passed":
        return gate.get("reason")
    return None


def _shadow_prediction_ready_reason(
    status: dict[str, Any],
    version: dict[str, Any],
    report: dict[str, Any],
    artifacts_ready: bool,
    torch_available: bool,
) -> str:
    if not torch_available:
        return "torch_unavailable"
    if status.get("status") != "trained":
        return status.get("reason") or f"model_status_{status.get('status') or 'unknown'}"
    if not artifacts_ready:
        return "artifacts_incomplete"
    validation_reason = lstm_validation_block_reason(status, version, report)
    if validation_reason != "passed":
        return validation_reason
    return "passed"
