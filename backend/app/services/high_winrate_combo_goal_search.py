from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from app.services.factor_duration_alignment import duration_entry_rows
from app.services.factor_learning_common import SUCCESS_PROFIT_FACTOR_MIN
from app.services.factor_performance_metrics import BACKTEST_MIN_PERIODS
from app.services.high_winrate_combo_goal_config import (
    GoalSearchConfig,
    SEARCH_CANDIDATE_LIMIT,
    SIGNAL_THRESHOLDS,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
    signal_thresholds,
    validated_search_config as _validated_search_config,
)
from app.services import high_winrate_combo_goal_diagnostics as diag

TARGET_WIN_RATE = 0.70
TARGET_COUNT = 5
TARGET_MIN_TRADES = BACKTEST_MIN_PERIODS
NEXT_ENTRY_HORIZON_BARS = 1
ZSCORE_CLIP = 4.0
COMBO_SIZES = (2, 3, 4)
EXCLUDED_COLUMNS = frozenset({"open_time", "open", "high", "low", "close", "volume", "fwd_ret"})


@dataclass(frozen=True)
class OrientedScore:
    score: pd.Series
    orientation: int


@dataclass(frozen=True)
class ComboHit:
    members: tuple[str, ...]
    orientations: tuple[int, ...]
    threshold: float
    win_rate: float
    profit_factor: float
    trades: int
    avg_return: float
    score: pd.Series


@dataclass(frozen=True)
class ScoreSearch:
    scores: dict[str, OrientedScore]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RankedSearch:
    hits: list[ComboHit]
    diagnostics: dict[str, Any]


def set_target_min_trades(value: int) -> None:
    global TARGET_MIN_TRADES
    TARGET_MIN_TRADES = int(value)


def search_frame(frame: pd.DataFrame, duration: str) -> pd.DataFrame:
    out = duration_entry_rows(frame.copy(), duration)
    out["fwd_ret"] = out["close"].shift(-NEXT_ENTRY_HORIZON_BARS) / out["close"] - 1.0
    return out


def oriented_scores(frame: pd.DataFrame) -> dict[str, OrientedScore]:
    return oriented_score_search(frame).scores


def oriented_score_search(frame: pd.DataFrame) -> ScoreSearch:
    scores: dict[str, OrientedScore] = {}
    rejected: list[tuple[str, int]] = []
    max_valid_pairs = 0
    for name in numeric_factor_columns(frame):
        series = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        usable_pairs = usable_pair_count(series, frame["fwd_ret"])
        max_valid_pairs = max(max_valid_pairs, usable_pairs)
        if usable_pairs < BACKTEST_MIN_PERIODS:
            rejected.append((name, usable_pairs))
            continue
        orientation = orientation_for_series(series, frame["fwd_ret"])
        scores[name] = OrientedScore(expanding_zscore(series) * orientation, orientation)
    numeric_columns = numeric_factor_columns(frame)
    payload = diag.candidate_diagnostics(len(numeric_columns), len(scores), rejected, max_valid_pairs)
    return ScoreSearch(scores, payload)


def numeric_factor_columns(frame: pd.DataFrame) -> list[str]:
    return [
        name
        for name in frame.columns
        if name not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(frame[name])
    ]


def usable_pair_count(series: pd.Series, fwd_ret: pd.Series) -> int:
    return len(pd.concat([series, fwd_ret], axis=1).dropna())


def orientation_for_series(series: pd.Series, fwd_ret: pd.Series) -> int:
    valid = pd.concat([series, fwd_ret], axis=1).dropna()
    if valid.iloc[:, 0].nunique(dropna=True) < 2 or valid.iloc[:, 1].nunique(dropna=True) < 2:
        return 1
    corr = valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman")
    return 1 if corr is not None and math.isfinite(float(corr)) and corr >= 0 else -1


def expanding_zscore(series: pd.Series) -> pd.Series:
    mean = series.expanding(min_periods=BACKTEST_MIN_PERIODS).mean().shift(1)
    std = series.expanding(min_periods=BACKTEST_MIN_PERIODS).std().shift(1)
    return ((series - mean) / std.replace(0.0, np.nan)).clip(-ZSCORE_CLIP, ZSCORE_CLIP)


def ranked_hits(
    frame: pd.DataFrame,
    scores: dict[str, OrientedScore],
    config: GoalSearchConfig | None = None,
) -> list[ComboHit]:
    return ranked_hit_search(frame, scores, config).hits


def ranked_hit_search(
    frame: pd.DataFrame,
    scores: dict[str, OrientedScore],
    config: GoalSearchConfig | None = None,
) -> RankedSearch:
    hits: dict[tuple[str, ...], ComboHit] = {}
    cfg = validated_search_config(config)
    names = search_candidate_names(frame, scores, cfg)
    payload = diag.ranked_search_diagnostics(names, cfg.candidate_limit)
    for size in COMBO_SIZES:
        for members in combinations(names, size):
            best = best_combo_hit_with_diagnostics(frame, members, scores, payload, cfg)
            if best is not None:
                hits[best.members] = best
    rows = list(hits.values())
    rows.sort(key=lambda row: (row.win_rate, row.profit_factor, row.avg_return, row.trades), reverse=True)
    payload["hitCount"] = len(rows)
    payload["failureReason"] = diag.ranked_failure_reason(
        bool(scores),
        len(rows),
        payload["selectedCandidateFactors"],
        min(COMBO_SIZES),
        payload,
    )
    return RankedSearch(rows, payload)


def search_candidate_names(
    frame: pd.DataFrame,
    scores: dict[str, OrientedScore],
    config: GoalSearchConfig | None = None,
) -> list[str]:
    cfg = validated_search_config(config)
    hits = [
        row for row in (best_combo_hit(frame, (name,), scores, min_win_rate=0.0, config=cfg) for name in scores)
        if row is not None
    ]
    hits.sort(key=lambda row: (row.win_rate, row.profit_factor, row.avg_return, row.trades), reverse=True)
    return [row.members[0] for row in hits[: cfg.candidate_limit]]


def best_combo_hit(
    frame: pd.DataFrame,
    members: tuple[str, ...],
    scores: dict[str, OrientedScore],
    *,
    min_win_rate: float = TARGET_WIN_RATE,
    config: GoalSearchConfig | None = None,
) -> ComboHit | None:
    cfg = validated_search_config(config)
    combo_score = sum(scores[name].score for name in members) / len(members)
    orientations = tuple(scores[name].orientation for name in members)
    candidates = [
        combo_hit(frame, members, orientations, combo_score, threshold, min_win_rate=min_win_rate, min_trades=cfg.min_trades)
        for threshold in cfg.signal_thresholds
    ]
    valid = [row for row in candidates if row is not None]
    return max(valid, key=lambda row: (row.win_rate, row.profit_factor, row.avg_return, row.trades), default=None)


def best_combo_hit_with_diagnostics(
    frame: pd.DataFrame,
    members: tuple[str, ...],
    scores: dict[str, OrientedScore],
    diagnostics: dict[str, Any],
    config: GoalSearchConfig | None = None,
) -> ComboHit | None:
    cfg = validated_search_config(config)
    diagnostics["testedCombinations"] += 1
    combo_score = sum(scores[name].score for name in members) / len(members)
    orientations = tuple(scores[name].orientation for name in members)
    candidates = []
    for threshold in cfg.signal_thresholds:
        hit, rejected = combo_hit_result(frame, members, orientations, combo_score, threshold, cfg)
        diagnostics["testedThresholdEvaluations"] += 1
        diag.record_combo_gate_result(diagnostics, hit, rejected)
        if hit is not None:
            candidates.append(hit)
    return max(candidates, key=lambda row: (row.win_rate, row.profit_factor, row.avg_return, row.trades), default=None)


def combo_hit_result(
    frame: pd.DataFrame,
    members: tuple[str, ...],
    orientations: tuple[int, ...],
    score: pd.Series,
    threshold: float,
    config: GoalSearchConfig | None = None,
) -> tuple[ComboHit | None, dict[str, Any] | None]:
    cfg = validated_search_config(config)
    signal = pd.Series(np.where(score >= threshold, 1.0, np.where(score <= -threshold, -1.0, np.nan)), index=frame.index)
    returns = (signal * frame["fwd_ret"]).replace([np.inf, -np.inf], np.nan).dropna()
    metrics = combo_return_metrics(returns)
    if metrics["trades"] < cfg.min_trades:
        return None, combo_rejection(members, threshold, "min_trades_below_min", metrics)
    if metrics["winRate"] < TARGET_WIN_RATE:
        return None, combo_rejection(members, threshold, "win_rate_below_min", metrics)
    if metrics["profitFactor"] < SUCCESS_PROFIT_FACTOR_MIN:
        return None, combo_rejection(members, threshold, "profit_factor_below_min", metrics)
    hit = ComboHit(
        members,
        orientations,
        threshold,
        metrics["winRate"],
        metrics["profitFactor"],
        metrics["trades"],
        metrics["avgReturn"],
        score,
    )
    return hit, None


def combo_hit(
    frame: pd.DataFrame,
    members: tuple[str, ...],
    orientations: tuple[int, ...],
    score: pd.Series,
    threshold: float,
    *,
    min_win_rate: float = TARGET_WIN_RATE,
    min_trades: int = TARGET_MIN_TRADES,
) -> ComboHit | None:
    signal = pd.Series(np.where(score >= threshold, 1.0, np.where(score <= -threshold, -1.0, np.nan)), index=frame.index)
    returns = (signal * frame["fwd_ret"]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < int(min_trades):
        return None
    win_rate = float((returns > 0).mean())
    factor = profit_factor(returns)
    if win_rate < min_win_rate or factor < SUCCESS_PROFIT_FACTOR_MIN:
        return None
    return ComboHit(members, orientations, threshold, win_rate, factor, len(returns), float(returns.mean()), score)


def combo_return_metrics(returns: pd.Series) -> dict[str, Any]:
    if returns.empty:
        return {"trades": 0, "winRate": 0.0, "profitFactor": 0.0, "avgReturn": 0.0}
    return {
        "trades": len(returns),
        "winRate": float((returns > 0).mean()),
        "profitFactor": profit_factor(returns),
        "avgReturn": float(returns.mean()),
    }


def combo_rejection(
    members: tuple[str, ...],
    threshold: float,
    reason: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {"members": members, "threshold": threshold, "reason": reason, **metrics}


def profit_factor(returns: pd.Series) -> float:
    gains = float(returns[returns > 0].sum())
    losses = abs(float(returns[returns < 0].sum()))
    return gains / losses if losses > 0 else math.inf


def validated_search_config(config: GoalSearchConfig | None = None) -> GoalSearchConfig:
    return _validated_search_config(config, TARGET_MIN_TRADES)


def selected_hits(hits: list[ComboHit], target_count: int) -> list[ComboHit]:
    return hits[:target_count]
