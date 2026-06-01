from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.services.high_winrate_combo_goal_config import OFFLINE_CANDIDATE_MIN_PROFIT_FACTOR
from app.services.high_winrate_combo_goal_types import ComboHit, OrientedScore
from app.services.trading_costs import roundtrip_cost_rate


@dataclass(frozen=True)
class ThresholdFrame:
    index: pd.Index
    fwd_ret: np.ndarray
    scores: dict[str, np.ndarray]


@dataclass(frozen=True)
class ThresholdHitContext:
    members: tuple[str, ...]
    orientations: tuple[int, ...]
    score_values: np.ndarray
    score_index: pd.Index
    frame: ThresholdFrame
    min_win_rate: float
    min_trades: int


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
    *,
    score_index: pd.Index,
    frame: ThresholdFrame,
    thresholds: tuple[float, ...],
    min_win_rate: float,
    min_trades: int,
) -> ComboHit | None:
    context = _hit_context(
        members,
        orientations,
        score_values,
        score_index=score_index,
        frame=frame,
        min_win_rate=min_win_rate,
        min_trades=min_trades,
    )
    hits = [
        _combo_hit(context, threshold)[0]
        for threshold in thresholds
    ]
    valid = [hit for hit in hits if hit is not None]
    return max(valid, key=lambda row: (row.win_rate, row.profit_factor, row.avg_return, row.trades), default=None)


def best_threshold_hit_with_diagnostics(
    members: tuple[str, ...],
    orientations: tuple[int, ...],
    score_values: np.ndarray,
    *,
    score_index: pd.Index,
    frame: ThresholdFrame,
    thresholds: tuple[float, ...],
    diagnostics: dict[str, Any],
    min_win_rate: float,
    min_trades: int,
) -> ComboHit | None:
    context = _hit_context(
        members,
        orientations,
        score_values,
        score_index=score_index,
        frame=frame,
        min_win_rate=min_win_rate,
        min_trades=min_trades,
    )
    candidates = []
    for threshold in thresholds:
        hit, rejected = _combo_hit(context, threshold)
        diagnostics["testedThresholdEvaluations"] += 1
        _record_combo_gate_result(diagnostics, hit, rejected)
        if hit is not None:
            candidates.append(hit)
    return max(candidates, key=lambda row: (row.win_rate, row.profit_factor, row.avg_return, row.trades), default=None)


def threshold_hit_result(
    members: tuple[str, ...],
    orientations: tuple[int, ...],
    score_values: np.ndarray,
    *,
    score_index: pd.Index,
    frame: ThresholdFrame,
    threshold: float,
    min_win_rate: float,
    min_trades: int,
) -> tuple[ComboHit | None, dict[str, Any] | None]:
    context = _hit_context(
        members,
        orientations,
        score_values,
        score_index=score_index,
        frame=frame,
        min_win_rate=min_win_rate,
        min_trades=min_trades,
    )
    return _combo_hit(context, threshold)


def _hit_context(
    members: tuple[str, ...],
    orientations: tuple[int, ...],
    score_values: np.ndarray,
    *,
    score_index: pd.Index,
    frame: ThresholdFrame,
    min_win_rate: float,
    min_trades: int,
) -> ThresholdHitContext:
    return ThresholdHitContext(members, orientations, score_values, score_index, frame, min_win_rate, min_trades)


def _combo_hit(
    context: ThresholdHitContext,
    threshold: float,
) -> tuple[ComboHit | None, dict[str, Any] | None]:
    returns = _threshold_returns(context.score_values, context.frame.fwd_ret, threshold)
    metrics = _return_metrics(returns)
    if metrics["trades"] < int(context.min_trades):
        return None, _combo_rejection(context, threshold=threshold, reason="min_trades_below_min", metrics=metrics)
    if metrics["winRate"] < context.min_win_rate:
        return None, _combo_rejection(context, threshold=threshold, reason="win_rate_below_min", metrics=metrics)
    if metrics["profitFactor"] < OFFLINE_CANDIDATE_MIN_PROFIT_FACTOR:
        return None, _combo_rejection(context, threshold=threshold, reason="profit_factor_below_min", metrics=metrics)
    hit = ComboHit(
        context.members,
        context.orientations,
        threshold,
        metrics["winRate"],
        metrics["profitFactor"],
        metrics["trades"],
        metrics["avgReturn"],
        pd.Series(context.score_values, index=context.score_index),
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
    context: ThresholdHitContext,
    *,
    threshold: float,
    reason: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {"members": context.members, "threshold": threshold, "reason": reason, **metrics}


def _record_combo_gate_result(diagnostics: dict[str, Any], hit: ComboHit | None, rejected: dict[str, Any] | None) -> None:
    from app.services.high_winrate_combo_goal_diagnostics import record_combo_gate_result

    record_combo_gate_result(diagnostics, hit, rejected)
