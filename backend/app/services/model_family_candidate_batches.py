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


def budgeted_candidates(
    candidates: list[ModelFamilyTrainingConfig],
    *,
    candidate_budget: int | None,
    attempted_count: int,
) -> list[ModelFamilyTrainingConfig]:
    if candidate_budget is None:
        return candidates
    remaining_budget = max(int(candidate_budget) - int(attempted_count), 0)
    return candidates[:remaining_budget]


def job_batch_payload(
    candidates: list[ModelFamilyTrainingConfig],
    available_count: int,
    *,
    full_available_count: int | None = None,
    candidate_budget: int | None = None,
    attempted_count: int = 0,
) -> dict[str, Any]:
    remaining = max(int(available_count) - len(candidates), 0)
    full_available = int(available_count if full_available_count is None else full_available_count)
    payload = {
        "selectedCandidates": len(candidates),
        "availableCandidatesBeforeJob": int(available_count),
        "remainingCandidatesAfterJob": remaining,
        "hasMoreCandidates": remaining > 0,
    }
    if candidate_budget is None:
        return payload
    unsearched = max(full_available - len(candidates) - remaining, 0)
    return {
        **payload,
        "candidateBudget": int(candidate_budget),
        "attemptedCandidatesBeforeJob": int(attempted_count),
        "budgetExhausted": remaining <= 0 and unsearched > 0,
        "unsearchedCandidatesAfterBudget": unsearched,
    }
