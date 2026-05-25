from __future__ import annotations

from math import comb
from typing import Any

from app.services.factor_combination_walk_forward import (
    RECENT_MIN_WIN_RATE,
    VALIDATION_MIN_AVG_RETURN,
    VALIDATION_MIN_PROFIT_FACTOR,
    VALIDATION_MIN_SAMPLE_COUNT,
    VALIDATION_MIN_WIN_RATE,
)


def combination_search_diagnostics(
    config: Any,
    candidates: list[Any],
    plan: list[Any],
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    generated_count: int,
    stages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "mode": "staged_layered_pairwise_diversity_v1",
        "baseCandidateCount": len(candidates),
        "fullCombinationEstimate": _full_combination_estimate(len(candidates), config.combo_sizes),
        "generatedCombinationCount": generated_count,
        "evaluatedCombinationCount": len(plan),
        "prefilteredCombinationCount": max(generated_count - len(plan), 0),
        "searchStages": stages or [],
        "prefilterLimit": config.prefilter_limit,
        "beamWidth": config.beam_width,
        "parallelWorkers": config.parallel_workers,
        "walkForwardPassedCount": sum(1 for row in rows if row["walkForwardPassed"]),
        "failureReasonCounts": _failure_reason_counts(failures),
        "targetCriteria": _target_criteria(),
    }


def _full_combination_estimate(candidate_count: int, sizes: tuple[int, ...]) -> int:
    return sum(comb(candidate_count, size) for size in sizes if candidate_count >= size)


def _failure_reason_counts(failures: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        reason = str(failure.get("error") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def _target_criteria() -> dict[str, Any]:
    return {
        "validationMinSampleCount": VALIDATION_MIN_SAMPLE_COUNT,
        "validationMinWinRate": VALIDATION_MIN_WIN_RATE,
        "validationMinProfitFactor": VALIDATION_MIN_PROFIT_FACTOR,
        "validationMinAvgReturn": VALIDATION_MIN_AVG_RETURN,
        "recentMinWinRate": RECENT_MIN_WIN_RATE,
    }
