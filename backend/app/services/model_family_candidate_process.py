from __future__ import annotations

from typing import Any

from app.services.model_family_candidates import model_search_key
from app.services.model_family_config import ModelFamilyTrainingConfig
from app.services.model_family_training_service import train_model_family


def train_candidate_in_process(config: ModelFamilyTrainingConfig, profile: str) -> dict[str, Any]:
    try:
        report = train_model_family(
            config,
            publish_shadow_active=False,
            publish_trade_active=False,
            write_attempt=False,
            persist_artifacts=False,
        )
    except Exception as exc:
        report = _failed_report(config, profile, exc)
    return {**report, "searchKey": model_search_key(config, profile)}


def _failed_report(config: ModelFamilyTrainingConfig, profile: str, exc: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "candidateStatus": "failed",
        "modelFamily": config.family,
        "symbol": config.symbol,
        "duration": config.duration,
        "modelVersion": None,
        "searchKey": model_search_key(config, profile),
        "validationFailureReason": str(exc),
        "failure": {
            "stage": "candidate_training",
            "exceptionType": type(exc).__name__,
            "error": str(exc),
        },
    }
