from __future__ import annotations

from typing import Any

from app.services.factor_learning_controls import (
    learning_mining_excluded_factor_names,
    learning_risk_blocked_factor_names,
)

RETRIEVAL_LIMIT = 8


def build_factor_learning_retrieval(memory: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(memory, dict):
        return _empty_retrieval()
    factor_mining = memory.get("factorMining") or {}
    loss_memory = memory.get("lossMemory") or {}
    weights = memory.get("weights") or {}
    success_patterns = _limited_items(factor_mining.get("successPatterns") or [])
    forbidden_regions = _limited_items(factor_mining.get("forbiddenRegions") or [])
    loss_patterns = _limited_items(loss_memory.get("patterns") or [])
    return {
        "blockedFactorNames": sorted(learning_risk_blocked_factor_names(memory)),
        "miningExcludedFactorNames": sorted(learning_mining_excluded_factor_names(memory)),
        "successPatterns": success_patterns,
        "forbiddenRegions": forbidden_regions,
        "lossPatterns": loss_patterns,
        "topWeights": _top_weights(weights),
        "summary": {
            "successPatternCount": len(factor_mining.get("successPatterns") or []),
            "forbiddenRegionCount": len(factor_mining.get("forbiddenRegions") or []),
            "lossPatternCount": len(loss_memory.get("patterns") or []),
            "weightCount": len(weights),
        },
    }


def _empty_retrieval() -> dict[str, Any]:
    return {
        "blockedFactorNames": [],
        "miningExcludedFactorNames": [],
        "successPatterns": [],
        "forbiddenRegions": [],
        "lossPatterns": [],
        "topWeights": [],
        "summary": {
            "successPatternCount": 0,
            "forbiddenRegionCount": 0,
            "lossPatternCount": 0,
            "weightCount": 0,
        },
    }


def _limited_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in items[:RETRIEVAL_LIMIT] if isinstance(item, dict)]


def _top_weights(weights: dict[str, Any]) -> list[dict[str, Any]]:
    scored = []
    for name, value in weights.items():
        weight = _finite_float(value)
        if weight is None:
            continue
        scored.append({"name": str(name), "weight": round(weight, 6)})
    scored.sort(key=lambda item: item["weight"], reverse=True)
    return scored[:RETRIEVAL_LIMIT]


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None
