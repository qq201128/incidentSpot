from __future__ import annotations

from dataclasses import dataclass
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
from app.services.model_family_config import (
    TORCH_MODEL_FAMILIES,
    model_family_strategy_key,
    normalize_model_family,
)
from app.services.model_family_dependency_status import dependency_status
from app.services.model_family_paper_live_policy import model_status_policy_payload
from app.services.model_family_search_rules import model_family_training_rules
from app.services.model_family_candidates import (
    model_candidate_library_summary,
)
from app.services.model_family_status_progress import candidate_search_progress


@dataclass(frozen=True)
class _StatusInputs:
    family: str
    symbol: str
    duration: str
    artifact_root: Path | None
    paths: Any
    raw_status: dict[str, Any]
    attempt: dict[str, Any]
    report: dict[str, Any]
    version: dict[str, Any]
    artifacts_ready: bool


@dataclass(frozen=True)
class _ReadinessContext:
    status: dict[str, Any]
    version: dict[str, Any]
    report: dict[str, Any]
    artifacts_ready: bool
    dependency_ready: bool
    clean_event_features: bool


@dataclass(frozen=True)
class _StatusPayloadContext:
    inputs: _StatusInputs
    status: dict[str, Any]
    dependency: dict[str, Any]
    snapshot: dict[str, Any]
    progress: dict[str, Any]
    readiness: _ReadinessContext


def model_family_status(
    family: str,
    symbol: str,
    duration: str,
    *,
    artifact_root: Path | None = None,
    current_combo_snapshot: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    inputs = _status_inputs(family, symbol, duration, artifact_root=artifact_root)
    status = _active_status(inputs)
    dependency = dependency_status(inputs.family)
    readiness = _ReadinessContext(status, inputs.version, inputs.report, inputs.artifacts_ready, dependency["available"], _has_clean_event_features(inputs.paths.features))
    snapshot = _combo_snapshot_status(inputs.symbol, duration, inputs.paths.features, current=current_combo_snapshot)
    progress = candidate_search_progress(inputs.family, inputs.symbol, duration, artifact_root=artifact_root)
    context = _StatusPayloadContext(inputs, status, dependency, snapshot, progress, readiness)
    return _status_payload(context)


def _status_inputs(
    family: str,
    symbol: str,
    duration: str,
    *,
    artifact_root: Path | None,
) -> _StatusInputs:
    selected = normalize_model_family(family)
    sym = symbol.strip().upper()
    paths = artifact_paths(sym, duration, artifact_root, family=selected)
    raw_status = read_json(paths.status) or _untrained_status(selected, sym, duration)
    return _StatusInputs(
        family=selected,
        symbol=sym,
        duration=duration,
        artifact_root=artifact_root,
        paths=paths,
        raw_status=raw_status,
        attempt=read_json(paths.attempt) or {},
        report=read_json(paths.report) or {},
        version=read_json(paths.version) or {},
        artifacts_ready=required_artifacts_exist(paths),
    )


def _status_payload(context: _StatusPayloadContext) -> dict[str, Any]:
    inputs = context.inputs
    gate = validation_gate_payload(inputs.version, inputs.report)
    library = model_candidate_library_summary(
        inputs.family,
        inputs.symbol,
        inputs.duration,
        artifact_root=inputs.artifact_root,
    )
    metadata = _model_metadata_payload(inputs, context.status, gate)
    shadow_reason = _shadow_ready_reason(context.readiness)
    trade_reason = _trade_ready_reason(context.readiness)
    report = inputs.report
    return {
        **context.status,
        **_display_metadata_payload(metadata, context.progress, library),
        **_dependency_payload(inputs.family, context.dependency),
        **_snapshot_payload(context.snapshot),
        "candidateLibrary": library,
        "candidateSearchProgress": context.progress,
        "trainingRules": model_family_training_rules(inputs.family),
        "shadowPredictionReady": shadow_reason == "passed",
        "shadowPredictionBlockedReason": shadow_reason,
        "tradePredictionReady": trade_reason == "passed",
        "tradePredictionBlockedReason": trade_reason,
        "cleanEventFeatures": context.readiness.clean_event_features,
        "regimeValidation": report.get("regimeValidation") if isinstance(report, dict) else None,
        **model_status_policy_payload(context.status.get("status"), gate),
    }


def _model_metadata_payload(inputs: _StatusInputs, status: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    report = inputs.report
    version = inputs.version
    attempt = inputs.attempt
    return {
        "modelFamily": inputs.family,
        "strategyKey": model_family_strategy_key(inputs.family, inputs.duration),
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
        "activeValidationFailureReason": _validation_failure_reason(inputs.raw_status, attempt, report),
        "validationFailureReason": _validation_failure_reason(inputs.raw_status, attempt, report),
        "artifactsReady": inputs.artifacts_ready,
    }


def _display_metadata_payload(metadata: dict[str, Any], progress: dict[str, Any], library: dict[str, Any]) -> dict[str, Any]:
    if _has_successful_candidate(progress, library):
        return {**metadata, "validationFailureReason": None}
    return metadata


def _has_successful_candidate(progress: dict[str, Any], library: dict[str, Any]) -> bool:
    if str(progress.get("status") or "") in {"trade_active", "trained", "shadow_active"}:
        return True
    return bool(library.get("bestTradeCandidate") or library.get("bestShadowCandidate"))


def _dependency_payload(family: str, dependency: dict[str, Any]) -> dict[str, Any]:
    return {
        "dependencyAvailable": dependency["available"],
        "dependencyStatus": dependency,
        "torchAvailable": dependency["available"] if family in TORCH_MODEL_FAMILIES else True,
        "torchStatus": dependency if family in TORCH_MODEL_FAMILIES else {"available": True},
    }


def _snapshot_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "comboSnapshotMatches": snapshot["matches"],
        "comboSnapshotReason": snapshot["reason"],
        "comboSnapshotCurrent": snapshot["current"],
        "comboSnapshotTrained": snapshot["trained"],
    }


def active_model_family_status(family: str, symbol: str, duration: str, *, artifact_root: Path | None = None) -> dict:
    return _active_status(_status_inputs(family, symbol, duration, artifact_root=artifact_root))


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


def _active_status(inputs: _StatusInputs) -> dict[str, Any]:
    status = inputs.raw_status
    if shadow_predictable_status(status.get("status")):
        return status
    if not _active_artifacts_pass_validation(inputs):
        return status
    return {
        "status": LSTM_STATUS_TRADE_ACTIVE,
        "modelFamily": inputs.family,
        "symbol": status.get("symbol") or inputs.symbol,
        "duration": status.get("duration") or inputs.duration,
        "featureWindow": inputs.report.get("featureWindow"),
        "minMoveBps": inputs.version.get("minMoveBps") or inputs.report.get("minMoveBps"),
        "updatedAt": inputs.version.get("trainedAt") or inputs.report.get("trainedAt"),
    }


def _active_artifacts_pass_validation(inputs: _StatusInputs) -> bool:
    return (
        inputs.raw_status.get("status") != "untrained"
        and inputs.artifacts_ready
        and _has_clean_event_features(inputs.paths.features)
        and model_validation_block_reason({"status": LSTM_STATUS_TRADE_ACTIVE}, inputs.version, inputs.report) == "passed"
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
        return _snapshot(False, current_list, trained, reason="current_combo_snapshot_missing")
    if not trained:
        return _snapshot(False, current_list, trained, reason="trained_combo_snapshot_missing")
    if current_list != trained:
        return _snapshot(False, current_list, trained, reason="combo_snapshot_mismatch")
    return _snapshot(True, current_list, trained, reason="passed")


def _shadow_ready_reason(context: _ReadinessContext) -> str:
    if not context.dependency_ready:
        return "dependency_unavailable"
    if not shadow_predictable_status(context.status.get("status")):
        return context.status.get("reason") or f"model_status_{context.status.get('status') or 'unknown'}"
    if not context.artifacts_ready:
        return "artifacts_incomplete"
    if not context.clean_event_features:
        return "clean_event_retrain_required"
    if context.status.get("status") == LSTM_STATUS_LEGACY_TRAINED:
        return model_validation_block_reason(context.status, context.version, context.report)
    return "passed"


def _trade_ready_reason(context: _ReadinessContext) -> str:
    if not context.dependency_ready:
        return "dependency_unavailable"
    if not context.artifacts_ready:
        return "artifacts_incomplete"
    if not context.clean_event_features:
        return "clean_event_retrain_required"
    return model_validation_block_reason(context.status, context.version, context.report)

def _has_clean_event_features(features_path: Path) -> bool:
    features = read_json(features_path) or {}
    return any(str(column).startswith("regime_") for column in (features.get("columns") or []))


def _validation_failure_reason(status, attempt, report) -> str | None:
    for payload in (attempt, status, report):
        reason = payload.get("validationFailureReason") or payload.get("reason")
        if reason:
            return str(reason)
    gate = report.get("validationGate") or {}
    return gate.get("reason") if isinstance(gate, dict) and gate.get("status") != "passed" else None


def _untrained_status(family: str, symbol: str, duration: str) -> dict[str, Any]:
    return {"status": "untrained", "modelFamily": family, "symbol": symbol, "duration": duration, "featureWindow": None}


def _snapshot(matches: bool, current: list, trained: list, *, reason: str) -> dict[str, Any]:
    return {"matches": matches, "current": current, "trained": trained, "reason": reason}
