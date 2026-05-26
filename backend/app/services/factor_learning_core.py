from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.factor_adaptive_learning import adaptive_learning_summary
from app.services.factor_learning_common import utc_now
from app.services.factor_learning_loss import loss_memory
from app.services.factor_learning_memory_store import FACTOR_LEARNING_VERSION
from app.services.factor_learning_mining_merge import factor_mining_payload
from app.services.factor_learning_patterns import (
    candidate_loss_columns,
    factor_rows,
    factor_weights,
    filter_config,
    forbidden_regions,
    success_patterns,
)
from app.services.factor_learning_retrieval import build_factor_learning_retrieval

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
        "factorMining": factor_mining_payload(
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
