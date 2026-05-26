from __future__ import annotations

from typing import Any

from app.services.model_family_config import ModelFamilyTrainingConfig
from app.services.model_family_status_service import model_family_status
from app.services.model_family_training_service import train_model_family


def publish_best_model_candidate(trainings: list[Any]) -> dict[str, Any] | None:
    selected = _best_trade_candidate(trainings)
    if selected is not None:
        return train_model_family(selected.config)
    selected = _best_shadow_candidate(trainings)
    if selected is not None and _needs_seed_model(selected.config):
        return train_model_family(selected.config)
    selected = _best_initial_baseline_candidate(trainings)
    if selected is not None and _needs_seed_model(selected.config):
        return train_model_family(selected.config, publish_initial_baseline=True)
    return None


def _best_trade_candidate(trainings: list[Any]) -> Any | None:
    eligible = [
        item
        for item in trainings
        if str(item.report.get("status") or "") in {"trade_active", "trained"}
    ]
    return max(eligible, key=lambda item: _candidate_score(item.report)) if eligible else None


def _best_shadow_candidate(trainings: list[Any]) -> Any | None:
    eligible = [item for item in trainings if str(item.report.get("status") or "") == "shadow_active"]
    return max(eligible, key=lambda item: _candidate_score(item.report)) if eligible else None


def _best_initial_baseline_candidate(trainings: list[Any]) -> Any | None:
    eligible = [item for item in trainings if str(item.report.get("status") or "") == "validation_failed"]
    return max(eligible, key=lambda item: _candidate_score(item.report)) if eligible else None


def _needs_seed_model(config: ModelFamilyTrainingConfig) -> bool:
    status = model_family_status(config.family, config.symbol, config.duration)
    return str(status.get("activeModelStatus") or status.get("status") or "") in {
        "untrained",
        "validation_failed",
        "failed",
        "insufficient_samples",
    }


def _candidate_score(report: dict[str, Any]) -> tuple[float, float, int]:
    validation = report.get("validation") or {}
    return (
        float(validation.get("winRate") or 0.0),
        float(validation.get("profitFactor") or 0.0),
        int((report.get("sampleCounts") or {}).get("validation") or validation.get("sampleCount") or 0),
    )
