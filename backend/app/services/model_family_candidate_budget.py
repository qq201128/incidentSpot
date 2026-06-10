from __future__ import annotations

from typing import Any

from app.services.model_family_candidate_batches import job_batch_payload
from app.services.model_family_candidates import finish_model_candidate_progress_from_library


def candidate_budget_exhausted_result(
    cfg: Any,
    *,
    status: str,
    full_available_count: int,
    attempted_count: int,
) -> dict[str, Any]:
    finish_model_candidate_progress_from_library(
        cfg.family,
        symbol=cfg.symbol,
        duration=cfg.duration,
        profile=cfg.profile,
        parallel_workers=cfg.parallel_workers,
        status=status,
    )
    return {
        "status": status,
        "reason": "candidate_budget_exhausted",
        "family": cfg.family,
        "jobBatch": job_batch_payload(
            [],
            0,
            full_available_count=full_available_count,
            candidate_budget=cfg.candidate_budget,
            attempted_count=attempted_count,
        ),
    }
