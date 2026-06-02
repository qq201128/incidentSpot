from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Any

from app.services.model_family_config import ModelFamilyTrainingConfig, TORCH_MODEL_FAMILIES
from app.services.model_family_search_rules import SUCCESSIVE_HALVING_SURVIVAL_RATE

COARSE_EPOCH_RATIO = 0.5
MIN_COARSE_EPOCHS = 1
NON_ADVANCING_STATUSES = frozenset({"failed", "insufficient_samples"})
FULL_STAGE_NON_ADVANCING_STATUSES = NON_ADVANCING_STATUSES | frozenset({"validation_failed"})


@dataclass(frozen=True)
class StageClosure:
    reports: list[Any]
    survivors: list[Any]
    payload: dict[str, Any]


def coarse_candidate_config(config: ModelFamilyTrainingConfig) -> ModelFamilyTrainingConfig:
    if config.family not in TORCH_MODEL_FAMILIES:
        return config
    epochs = max(MIN_COARSE_EPOCHS, min(config.epochs, int(config.epochs * COARSE_EPOCH_RATIO)))
    return replace(config, epochs=epochs)


def close_halving_stage(results: list[Any], stage: str) -> StageClosure:
    survivors = _survivors(results, stage)
    survivor_keys = {str(item.report.get("searchKey")) for item in survivors}
    reports = [_annotated_result(item, stage, str(item.report.get("searchKey")) in survivor_keys) for item in results]
    return StageClosure(
        reports=reports,
        survivors=survivors,
        payload=_stage_payload(stage, results, survivors),
    )


def walk_forward_stage_payload(finalists: list[Any]) -> dict[str, Any]:
    return {
        "stage": "walk_forward",
        "evaluated": len(finalists),
        "advanced": len(finalists),
        "survivalRate": 1.0,
        "candidateKeys": _search_keys(finalists),
        "advancedKeys": _search_keys(finalists),
    }


def candidate_score(report: dict[str, Any]) -> tuple[float, float, int]:
    validation = report.get("validation") or {}
    samples = report.get("sampleCounts") or {}
    return (
        float(validation.get("winRate") or 0.0),
        float(validation.get("profitFactor") or 0.0),
        int(samples.get("validation") or validation.get("sampleCount") or 0),
    )


def _survivors(results: list[Any], stage: str) -> list[Any]:
    eligible = [item for item in results if _can_advance(item.report, stage)]
    limit = ceil(len(results) * SUCCESSIVE_HALVING_SURVIVAL_RATE)
    selected = sorted(eligible, key=lambda item: candidate_score(item.report), reverse=True)
    return selected[:limit]


def _annotated_result(item: Any, stage: str, advanced: bool) -> Any:
    reason = None if advanced else _elimination_reason(stage, item.report)
    report = {**item.report, "advancedToNextStage": advanced, "eliminationReason": reason}
    return type(item)(item.config, report)


def _can_advance(report: dict[str, Any], stage: str) -> bool:
    blocked = FULL_STAGE_NON_ADVANCING_STATUSES if stage == "full" else NON_ADVANCING_STATUSES
    return str(report.get("status") or "failed") not in blocked


def _elimination_reason(stage: str, report: dict[str, Any]) -> str:
    if not _can_advance(report, stage):
        return f"{stage}_training_failed"
    return f"{stage}_rank_below_survival_cutoff"


def _stage_payload(stage: str, results: list[Any], survivors: list[Any]) -> dict[str, Any]:
    return {
        "stage": stage,
        "evaluated": len(results),
        "advanced": len(survivors),
        "survivalRate": SUCCESSIVE_HALVING_SURVIVAL_RATE,
        "candidateKeys": _search_keys(results),
        "advancedKeys": _search_keys(survivors),
    }


def _search_keys(results: list[Any]) -> list[str]:
    return [str(item.report.get("searchKey")) for item in results if item.report.get("searchKey")]
