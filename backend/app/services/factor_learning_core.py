from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.factor_adaptive_learning import adaptive_learning_summary
from app.services.factor_learning_common import utc_now
from app.services.factor_learning_loss import loss_memory
from app.services.factor_learning_memory_store import FACTOR_LEARNING_VERSION
from app.services.factor_learning_patterns import (
    candidate_loss_columns,
    factor_rows,
    factor_weights,
    filter_config,
    forbidden_regions,
    success_patterns,
)
from app.services.factor_learning_retrieval import build_factor_learning_retrieval
from app.services.factor_operator_library import factor_operator_summary


def build_factor_learning_memory(
    frame: pd.DataFrame,
    ranking_report: dict[str, Any],
    settled_predictions: list[dict[str, Any]],
    *,
    symbol: str,
    duration: str,
    settlement_sweep: dict[str, int] | None = None,
    mined_frame_failures: list[dict[str, Any]] | None = None,
    mined_library: dict[str, Any] | None = None,
    agent_mined_library: dict[str, Any] | None = None,
    monitoring_report: dict[str, Any] | None = None,
    lstm_shadow: dict[str, Any] | None = None,
    previous_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = factor_rows(ranking_report)
    loss_columns = candidate_loss_columns(rows, frame)
    learned_losses = loss_memory(frame, settled_predictions, loss_columns)
    adaptive_learning = adaptive_learning_summary(
        rows,
        settled_predictions,
        duration=duration,
        loss_patterns=learned_losses["patterns"],
        monitoring_report=monitoring_report,
        lstm_shadow=lstm_shadow,
    )
    payload = _memory_payload(
        symbol=symbol,
        duration=duration,
        updated_at=utc_now(),
        rows=rows,
        frame=frame,
        ranking_report=ranking_report,
        settled_predictions=settled_predictions,
        learned_losses=learned_losses,
        adaptive_learning=adaptive_learning,
        settlement_sweep=settlement_sweep,
        mined_frame_failures=mined_frame_failures,
        mined_library=mined_library,
        agent_mined_library=agent_mined_library,
        monitoring_report=monitoring_report,
        lstm_shadow=lstm_shadow,
        previous_memory=previous_memory,
    )
    payload["retrieval"] = build_factor_learning_retrieval(payload)
    return payload


def _memory_payload(
    *,
    symbol: str,
    duration: str,
    updated_at: str,
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    ranking_report: dict[str, Any],
    settled_predictions: list[dict[str, Any]],
    learned_losses: dict[str, Any],
    adaptive_learning: dict[str, Any],
    settlement_sweep: dict[str, int] | None,
    mined_frame_failures: list[dict[str, Any]] | None,
    mined_library: dict[str, Any] | None,
    agent_mined_library: dict[str, Any] | None,
    monitoring_report: dict[str, Any] | None,
    lstm_shadow: dict[str, Any] | None,
    previous_memory: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "version": FACTOR_LEARNING_VERSION,
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "updatedAt": updated_at,
        "source": _source_payload(
            ranking_report=ranking_report,
            settled_predictions=settled_predictions,
            learned_losses=learned_losses,
            settlement_sweep=settlement_sweep,
            mined_frame_failures=mined_frame_failures,
        ),
        "factorMining": _factor_mining_payload(
            previous_memory=previous_memory,
            current_success=success_patterns(rows),
            current_forbidden=forbidden_regions(frame, rows),
            now=updated_at,
        ),
        "lossMemory": learned_losses,
        "filters": filter_config(learned_losses),
        "weights": factor_weights(rows, learned_losses["patterns"]),
        "adaptiveLearning": adaptive_learning,
        "lstmShadow": lstm_shadow or {},
        "minedFactorLibrary": mined_library or {},
        "agentMinedFactorLibrary": agent_mined_library or {},
        "monitoring": monitoring_report or {},
    }
    return payload


def _factor_mining_payload(
    *,
    previous_memory: dict[str, Any] | None,
    current_success: list[dict[str, Any]],
    current_forbidden: list[dict[str, Any]],
    now: str,
) -> dict[str, Any]:
    return {
        "operatorLibrary": factor_operator_summary(),
        "successPatterns": _merge_success_patterns(previous_memory, current_success, now),
        "forbiddenRegions": _merge_forbidden_regions(previous_memory, current_forbidden, now),
    }


def _merge_success_patterns(
    previous_memory: dict[str, Any] | None,
    current: list[dict[str, Any]],
    now: str,
) -> list[dict[str, Any]]:
    previous = _previous_factor_mining_items(previous_memory, "successPatterns")
    by_key = {_success_pattern_key(item): dict(item) for item in previous if _success_pattern_key(item)}
    for item in current:
        key = _success_pattern_key(item)
        if not key:
            continue
        merged = _merge_success_pattern(by_key.get(key), item, now)
        by_key[key] = merged
    rows = list(by_key.values())
    rows.sort(key=lambda item: (float(item.get("score") or 0.0), int(item.get("support") or 0)), reverse=True)
    return rows


def _merge_success_pattern(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    now: str,
) -> dict[str, Any]:
    if previous is None:
        return {**current, "firstSeenAt": now, "lastSeenAt": now}
    previous_support = int(previous.get("support") or 0)
    current_support = int(current.get("support") or 0)
    support = previous_support + current_support
    return {
        **previous,
        **current,
        "support": support,
        "score": _weighted_average(
            previous_value=previous.get("score"),
            previous_weight=previous_support,
            current_value=current.get("score"),
            current_weight=current_support,
        ),
        "factors": _merged_list(previous.get("factors"), current.get("factors")),
        "firstSeenAt": previous.get("firstSeenAt") or now,
        "lastSeenAt": now,
    }


def _merge_forbidden_regions(
    previous_memory: dict[str, Any] | None,
    current: list[dict[str, Any]],
    now: str,
) -> list[dict[str, Any]]:
    previous = _previous_factor_mining_items(previous_memory, "forbiddenRegions")
    by_key = {_forbidden_region_key(item): dict(item) for item in previous if _forbidden_region_key(item)}
    for item in current:
        key = _forbidden_region_key(item)
        if not key:
            continue
        by_key[key] = _merge_forbidden_region(by_key.get(key), item, now)
    rows = list(by_key.values())
    rows.sort(
        key=lambda item: (float(item.get("avgAbsCorrelation") or 0.0), int(item.get("support") or 0)),
        reverse=True,
    )
    return rows


def _merge_forbidden_region(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    now: str,
) -> dict[str, Any]:
    if previous is None:
        return {**current, "firstSeenAt": now, "lastSeenAt": now}
    previous_support = int(previous.get("support") or 0)
    current_support = int(current.get("support") or 0)
    support = previous_support + current_support
    return {
        **previous,
        **current,
        "support": support,
        "avgAbsCorrelation": _weighted_average(
            previous_value=previous.get("avgAbsCorrelation"),
            previous_weight=previous_support,
            current_value=current.get("avgAbsCorrelation"),
            current_weight=current_support,
        ),
        "members": _merged_list(previous.get("members"), current.get("members")),
        "firstSeenAt": previous.get("firstSeenAt") or now,
        "lastSeenAt": now,
    }


def _previous_factor_mining_items(previous_memory: dict[str, Any] | None, key: str) -> list[dict[str, Any]]:
    if not isinstance(previous_memory, dict):
        return []
    factor_mining = previous_memory.get("factorMining")
    if not isinstance(factor_mining, dict):
        return []
    items = factor_mining.get(key)
    return [dict(item) for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _success_pattern_key(item: dict[str, Any]) -> str:
    return str(item.get("pattern") or "")


def _forbidden_region_key(item: dict[str, Any]) -> str:
    return str(item.get("region") or "")


def _weighted_average(
    *,
    previous_value: Any,
    previous_weight: int,
    current_value: Any,
    current_weight: int,
) -> float:
    total = previous_weight + current_weight
    if total <= 0:
        return 0.0
    return round(
        (float(previous_value or 0.0) * previous_weight + float(current_value or 0.0) * current_weight) / total,
        4,
    )


def _merged_list(previous: Any, current: Any) -> list[Any]:
    values = []
    for item in [*(previous or []), *(current or [])]:
        if item not in values:
            values.append(item)
    return values


def _source_payload(
    *,
    ranking_report: dict[str, Any],
    settled_predictions: list[dict[str, Any]],
    learned_losses: dict[str, Any],
    settlement_sweep: dict[str, int] | None,
    mined_frame_failures: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    return {
        "rankingTotal": int(ranking_report.get("total") or 0),
        "baseFactorCount": int(ranking_report.get("baseFactorCount") or 0),
        "rankingRefreshSource": str(ranking_report.get("learningRefreshSource") or "provided"),
        "minedFactorSourceCount": int(ranking_report.get("minedFactorSourceCount") or 0),
        "minedFactorUsedCount": int(ranking_report.get("minedFactorUsedCount") or 0),
        "settledPredictionCount": len(settled_predictions),
        "lossPatternCount": len(learned_losses["patterns"]),
        "lossMemoryStatus": learned_losses["status"],
        "settlementSweep": settlement_sweep or {},
        "minedFrameFailureCount": len(mined_frame_failures or []),
        "minedFrameFailures": (mined_frame_failures or [])[:20],
    }
