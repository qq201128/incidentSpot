from __future__ import annotations

from typing import Any

from app.services.factor_learning_memory_store import load_factor_learning_memory


def load_factor_learning_memory_for(symbol: str, duration: str) -> dict[str, Any] | None:
    return load_factor_learning_memory(symbol, duration)


def learning_blocked_factor_names(memory: dict[str, Any] | None) -> set[str]:
    if not isinstance(memory, dict):
        return set()
    names = set()
    names.update(_forbidden_region_members(memory.get("factorMining") or {}))
    names.update(_loss_pattern_features(memory.get("lossMemory") or {}))
    return names


def learning_weight(memory: dict[str, Any] | None, name: str) -> float:
    if not isinstance(memory, dict):
        return 0.0
    return _finite_float((memory.get("weights") or {}).get(name)) or 0.0


def _forbidden_region_members(factor_mining: dict[str, Any]) -> set[str]:
    names = set()
    for region in factor_mining.get("forbiddenRegions") or []:
        if not isinstance(region, dict):
            continue
        for member in region.get("members") or []:
            member_name = str(member or "").strip()
            if member_name:
                names.add(member_name)
    return names


def _loss_pattern_features(loss_memory: dict[str, Any]) -> set[str]:
    names = set()
    for pattern in loss_memory.get("patterns") or []:
        if not isinstance(pattern, dict):
            continue
        feature = str(pattern.get("feature") or "").strip()
        if feature:
            names.add(feature)
    return names


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None
