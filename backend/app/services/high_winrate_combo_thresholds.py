from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.services.factor_learning_common import SUCCESS_PROFIT_FACTOR_MIN
from app.services.high_winrate_combo_goal_types import ComboHit, OrientedScore
from app.services.trading_costs import roundtrip_cost_rate


@dataclass(frozen=True)
class ThresholdFrame:
    index: pd.Index
    fwd_ret: np.ndarray
    scores: dict[str, np.ndarray]


def threshold_frame(frame: pd.DataFrame, scores: dict[str, OrientedScore]) -> ThresholdFrame:
    return ThresholdFrame(
        index=frame.index,
        fwd_ret=frame["fwd_ret"].to_numpy(dtype=float),
        scores={name: score.score.reindex(frame.index).to_numpy(dtype=float) for name, score in scores.items()},
    )


def combo_score_array(members: tuple[str, ...], frame: ThresholdFrame) -> np.ndarray:
    values = np.zeros(len(frame.index), dtype=float)
    for member in members:
        values += frame.scores[member]
    return values / len(members)


def best_threshold_hit(
    members: tuple[str, ...],
    orientations: tuple[int, ...],
    score_values: np.ndarray,
    score_index: pd.Index,
    frame: ThresholdFrame,
    thresholds: tuple[float, ...],
    *,
    min_win_rate: float,
    min_trades: int,
) -> ComboHit | None:
    hits = [
        _combo_hit(members, orientations, score_values, score_index, frame, threshold, min_win_rate, min_trades)[0]
        for threshold in thresholds
    ]
    valid = [hit for hit in hits if hit is not None]
    return max(valid, key=lambda row: (row.win_rate, row.profit_factor, row.avg_return, row.trades), default=None)


def best_threshold_hit_with_diagnostics(
    members: tuple[str, ...],
    orientations: tuple[int, ...],
    score_values: np.ndarray,
    score_index: pd.Index,
    frame: ThresholdFrame,
    thresholds: tuple[float, ...],
    diagnostics: dict[str, Any],
    *,
    min_win_rate: float,
    min_trades: int,
) -> ComboHit | None:
    candidates = []
    for threshold in thresholds:
        hit, rejected = _combo_hit(
            members,
            orientations,
            score_values,
            score_index,
            frame,
            threshold,
            min_win_rate,
            min_trades,
        )
        diagnostics["testedThresholdEvaluations"] += 1
        _record_combo_gate_result(diagnostics, hit, rejected)
        if hit is not None:
            candidates.append(hit)
    return max(candidates, key=lambda row: (row.win_rate, row.profit_factor, row.avg_return, row.trades), default=None)


def threshold_hit_result(
    members: tuple[str, ...],
    orientations: tuple[int, ...],
    score_values: np.ndarray,
    score_index: pd.Index,
    frame: ThresholdFrame,
    threshold: float,
    *,
    min_win_rate: float,
    min_trades: int,
) -> tuple[ComboHit | None, dict[str, Any] | None]:
    return _combo_hit(members, orientations, score_values, score_index, frame, threshold, min_win_rate, min_trades)


def _combo_hit(
    members: tuple[str, ...],
    orientations: tuple[int, ...],
    score_values: np.ndarray,
    score_index: pd.Index,
    frame: ThresholdFrame,
    threshold: float,
    min_win_rate: float,
    min_trades: int,
) -> tuple[ComboHit | None, dict[str, Any] | None]:
    returns = _threshold_returns(score_values, frame.fwd_ret, threshold)
    metrics = _return_metrics(returns)
    if metrics["trades"] < int(min_trades):
        return None, _combo_rejection(members, threshold, "min_trades_below_min", metrics)
    if metrics["winRate"] < min_win_rate:
        return None, _combo_rejection(members, threshold, "win_rate_below_min", metrics)
    if metrics["profitFactor"] < SUCCESS_PROFIT_FACTOR_MIN:
        return None, _combo_rejection(members, threshold, "profit_factor_below_min", metrics)
    hit = ComboHit(
        members,
        orientations,
        threshold,
        metrics["winRate"],
        metrics["profitFactor"],
        metrics["trades"],
        metrics["avgReturn"],
        pd.Series(score_values, index=score_index),
    )
    return hit, None


def _threshold_returns(score: np.ndarray, fwd_ret: np.ndarray, threshold: float) -> np.ndarray:
    signal = np.where(score >= threshold, 1.0, np.where(score <= -threshold, -1.0, np.nan))
    returns = signal * fwd_ret - roundtrip_cost_rate()
    return returns[np.isfinite(returns)]


def _return_metrics(returns: np.ndarray) -> dict[str, Any]:
    if returns.size == 0:
        return {"trades": 0, "winRate": 0.0, "profitFactor": 0.0, "avgReturn": 0.0}
    gains = float(returns[returns > 0].sum())
    losses = abs(float(returns[returns < 0].sum()))
    profit_factor = gains / losses if losses > 0 else math.inf
    return {
        "trades": int(returns.size),
        "winRate": float((returns > 0).mean()),
        "profitFactor": profit_factor,
        "avgReturn": float(returns.mean()),
    }


def _combo_rejection(
    members: tuple[str, ...],
    threshold: float,
    reason: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {"members": members, "threshold": threshold, "reason": reason, **metrics}


def _record_combo_gate_result(diagnostics: dict[str, Any], hit: ComboHit | None, rejected: dict[str, Any] | None) -> None:
    from app.services.high_winrate_combo_goal_diagnostics import record_combo_gate_result

    record_combo_gate_result(diagnostics, hit, rejected)
