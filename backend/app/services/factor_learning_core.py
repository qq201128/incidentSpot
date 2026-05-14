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
    return {
        "version": FACTOR_LEARNING_VERSION,
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "updatedAt": utc_now(),
        "source": _source_payload(
            ranking_report,
            settled_predictions,
            learned_losses,
            settlement_sweep,
            mined_frame_failures,
        ),
        "factorMining": {
            "operatorLibrary": factor_operator_summary(),
            "successPatterns": success_patterns(rows),
            "forbiddenRegions": forbidden_regions(frame, rows),
        },
        "lossMemory": learned_losses,
        "filters": filter_config(learned_losses),
        "weights": factor_weights(rows, learned_losses["patterns"]),
        "adaptiveLearning": adaptive_learning,
        "lstmShadow": lstm_shadow or {},
        "minedFactorLibrary": mined_library or {},
        "agentMinedFactorLibrary": agent_mined_library or {},
        "monitoring": monitoring_report or {},
    }


def _source_payload(
    ranking_report: dict[str, Any],
    settled_predictions: list[dict[str, Any]],
    learned_losses: dict[str, Any],
    settlement_sweep: dict[str, int] | None,
    mined_frame_failures: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    return {
        "rankingTotal": int(ranking_report.get("total") or 0),
        "baseFactorCount": int(ranking_report.get("baseFactorCount") or 0),
        "minedFactorSourceCount": int(ranking_report.get("minedFactorSourceCount") or 0),
        "minedFactorUsedCount": int(ranking_report.get("minedFactorUsedCount") or 0),
        "settledPredictionCount": len(settled_predictions),
        "lossPatternCount": len(learned_losses["patterns"]),
        "lossMemoryStatus": learned_losses["status"],
        "settlementSweep": settlement_sweep or {},
        "minedFrameFailureCount": len(mined_frame_failures or []),
        "minedFrameFailures": (mined_frame_failures or [])[:20],
    }
