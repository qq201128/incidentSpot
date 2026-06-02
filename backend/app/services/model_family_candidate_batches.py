from __future__ import annotations

from typing import Any

from app.services.model_family_config import ModelFamilyTrainingConfig


def candidate_job_batch(
    candidates: list[ModelFamilyTrainingConfig],
    candidates_per_job: int | None,
) -> list[ModelFamilyTrainingConfig]:
    if candidates_per_job is None:
        return candidates
    if candidates_per_job <= 0:
        raise ValueError("candidates_per_job must be positive")
    return candidates[:candidates_per_job]


def job_batch_payload(
    candidates: list[ModelFamilyTrainingConfig],
    available_count: int,
) -> dict[str, Any]:
    remaining = max(int(available_count) - len(candidates), 0)
    return {
        "selectedCandidates": len(candidates),
        "availableCandidatesBeforeJob": int(available_count),
        "remainingCandidatesAfterJob": remaining,
        "hasMoreCandidates": remaining > 0,
    }
