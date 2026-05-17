from __future__ import annotations

from math import isfinite
from typing import Any

import pandas as pd

from app.services.factor_combo_scoring import oriented_zscore
from app.services.factor_learning_controls import learning_blocked_factor_names
from app.services.factor_learning_memory_store import load_factor_learning_memory

SCORE_DECIMALS = 6
PROBABILITY_DECIMALS = 4
FILTERED_QUALITY_SCORE_MAX = 0.49
MEMORY_NOT_PROVIDED = object()


def enrich_signal_with_factor_learning(
    payload: dict[str, Any],
    frame: pd.DataFrame,
    index: Any,
    *,
    symbol: str,
    duration: str,
    memory: dict[str, Any] | None | object = MEMORY_NOT_PROVIDED,
    zscore_cache: dict[tuple[str, int], pd.Series] | None = None,
) -> dict[str, Any]:
    resolved_memory = _resolve_memory(memory, symbol, duration)
    if resolved_memory is None:
        return _with_missing_memory(payload)
    return apply_factor_learning_memory(payload, frame, index, resolved_memory, zscore_cache=zscore_cache)


def apply_factor_learning_memory(
    payload: dict[str, Any],
    frame: pd.DataFrame,
    index: Any,
    memory: dict[str, Any],
    *,
    zscore_cache: dict[tuple[str, int], pd.Series] | None = None,
) -> dict[str, Any]:
    members = _members(payload)
    blocked_members = _blocked_member_matches(members, memory)
    weighted = _weighted_member_score(frame, index, members, memory.get("weights") or {}, zscore_cache)
    enriched = _apply_weighted_score(dict(payload), weighted)
    loss_matches = _matched_loss_patterns(frame, index, memory)
    confirmations = _confirmation_count(frame, index, members, enriched["direction"], zscore_cache)
    filter_passed = _filter_passed(
        memory.get("filters") or {},
        confirmations,
        len(members),
        loss_matches,
        blocked_members,
    )
    enriched["qualityPassed"] = bool(payload["qualityPassed"] and filter_passed)
    if payload["qualityPassed"] and not filter_passed:
        enriched["qualityGateReason"] = "factor_learning_filter_blocked"
    enriched["factorLearning"] = _learning_payload(
        memory,
        payload,
        weighted,
        confirmations,
        loss_matches,
        blocked_members,
        filter_passed,
    )
    return enriched


def _with_missing_memory(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "factorLearning": {
            "state": "missing_memory",
            "filterPassed": None,
            "baseQualityPassed": bool(payload["qualityPassed"]),
            "qualityScore": payload["confidence"],
        },
    }


def _resolve_memory(
    memory: dict[str, Any] | None | object,
    symbol: str,
    duration: str,
) -> dict[str, Any] | None:
    if memory is MEMORY_NOT_PROVIDED:
        return load_factor_learning_memory(symbol, duration)
    return memory if isinstance(memory, dict) else None


def _apply_weighted_score(payload: dict[str, Any], weighted: dict[str, Any] | None) -> dict[str, Any]:
    if weighted is None:
        return payload
    score = float(weighted["score"])
    direction = "up" if score >= 0 else "down"
    payload["score"] = round(score, SCORE_DECIMALS)
    payload["direction"] = direction
    payload["probabilityUp"] = _probability_up(float(payload["confidence"]), direction)
    return payload


def _learning_payload(
    memory: dict[str, Any],
    base_payload: dict[str, Any],
    weighted: dict[str, Any] | None,
    confirmations: int,
    loss_matches: list[dict[str, Any]],
    blocked_members: list[dict[str, Any]],
    filter_passed: bool,
) -> dict[str, Any]:
    quality_score = _quality_score(float(base_payload["confidence"]), filter_passed)
    return {
        "state": "active",
        "memoryUpdatedAt": memory.get("updatedAt"),
        "baseQualityPassed": bool(base_payload["qualityPassed"]),
        "filterPassed": filter_passed,
        "qualityScore": quality_score,
        "rawScore": base_payload["score"],
        "learnedScore": None if weighted is None else round(float(weighted["score"]), SCORE_DECIMALS),
        "weightCoverage": 0.0 if weighted is None else weighted["coverage"],
        "confirmationCount": confirmations,
        "requiredConfirmations": _required_confirmations(
            memory.get("filters") or {},
            len(_members(base_payload)),
        ),
        "lossPatternMatches": loss_matches,
        "blockedMembers": blocked_members,
    }


def _weighted_member_score(
    frame: pd.DataFrame,
    index: Any,
    members: list[dict[str, Any]],
    weights: dict[str, Any],
    zscore_cache: dict[tuple[str, int], pd.Series] | None,
) -> dict[str, Any] | None:
    weighted_scores = []
    total_weight = 0.0
    for member in members:
        name = str(member["name"])
        weight = _finite_float(weights.get(name))
        score = _member_score_at(frame, index, member, zscore_cache)
        if weight is None or score is None or weight <= 0:
            continue
        weighted_scores.append(score * weight)
        total_weight += weight
    if total_weight <= 0:
        return None
    return {"score": sum(weighted_scores) / total_weight, "coverage": round(total_weight, 6)}


def _confirmation_count(
    frame: pd.DataFrame,
    index: Any,
    members: list[dict[str, Any]],
    direction: str,
    zscore_cache: dict[tuple[str, int], pd.Series] | None,
) -> int:
    target = 1 if direction == "up" else -1
    scores = [_member_score_at(frame, index, member, zscore_cache) for member in members]
    return sum(1 for score in scores if score is not None and score * target > 0)


def _member_score_at(
    frame: pd.DataFrame,
    index: Any,
    member: dict[str, Any],
    zscore_cache: dict[tuple[str, int], pd.Series] | None,
) -> float | None:
    name = str(member["name"])
    if name not in frame.columns:
        return None
    values = _cached_oriented_zscore(frame[name], name, int(member.get("orientation") or 1), zscore_cache)
    if index not in values.index:
        return None
    score = _finite_float(values.at[index])
    return None if score is None else float(score)


def _cached_oriented_zscore(
    series: pd.Series,
    name: str,
    orientation: int,
    zscore_cache: dict[tuple[str, int], pd.Series] | None,
) -> pd.Series:
    if zscore_cache is None:
        return oriented_zscore(series, orientation)
    key = (name, orientation)
    if key not in zscore_cache:
        zscore_cache[key] = oriented_zscore(series, orientation)
    return zscore_cache[key]


def _matched_loss_patterns(frame: pd.DataFrame, index: Any, memory: dict[str, Any]) -> list[dict[str, Any]]:
    patterns = (memory.get("lossMemory") or {}).get("patterns") or []
    matches = []
    for pattern in patterns:
        match = _loss_pattern_match(frame, index, pattern)
        if match is not None:
            matches.append(match)
    return matches


def _loss_pattern_match(frame: pd.DataFrame, index: Any, pattern: dict[str, Any]) -> dict[str, Any] | None:
    feature = str(pattern.get("feature") or "")
    if not feature or feature not in frame.columns:
        return None
    value = _finite_float(frame.at[index, feature])
    threshold = _finite_float(pattern.get("threshold"))
    if value is None or threshold is None:
        return None
    direction = str(pattern.get("direction"))
    matched = value >= threshold if direction == "high" else value <= threshold
    if not matched:
        return None
    return {
        "feature": feature,
        "direction": direction,
        "threshold": threshold,
        "value": round(value, SCORE_DECIMALS),
        "lossRate": pattern.get("lossRate"),
        "support": pattern.get("support"),
    }


def _filter_passed(
    config: dict[str, Any],
    confirmations: int,
    member_count: int,
    loss_matches: list[dict[str, Any]],
    blocked_members: list[dict[str, Any]],
) -> bool:
    if blocked_members:
        return False
    max_loss_matches = config.get("lossPatternMaxMatches")
    loss_passed = max_loss_matches is None or len(loss_matches) <= int(max_loss_matches)
    return confirmations >= _required_confirmations(config, member_count) and loss_passed


def _required_confirmations(config: dict[str, Any], member_count: int) -> int:
    configured = int(config.get("minConfirmations") or 1)
    return max(1, min(configured, member_count))


def _quality_score(confidence: float, filter_passed: bool) -> float:
    score = confidence if filter_passed else min(confidence, FILTERED_QUALITY_SCORE_MAX)
    return round(score, PROBABILITY_DECIMALS)


def _probability_up(confidence: float, direction: str) -> float:
    value = confidence if direction == "up" else 1.0 - confidence
    return round(value, PROBABILITY_DECIMALS)


def _members(payload: dict[str, Any]) -> list[dict[str, Any]]:
    members = payload.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("factor combo signal missing members")
    return [dict(member) for member in members]


def _blocked_member_matches(members: list[dict[str, Any]], memory: dict[str, Any]) -> list[dict[str, Any]]:
    blocked_names = learning_blocked_factor_names(memory)
    matches = []
    for member in members:
        name = str(member.get("name") or "").strip()
        if name and name in blocked_names:
            matches.append({"feature": name})
    return matches


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None
