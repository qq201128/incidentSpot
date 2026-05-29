from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from app.services.factor_duration_alignment import duration_entry_rows
from app.services.factor_learning_common import SUCCESS_PROFIT_FACTOR_MIN
from app.services.factor_performance_metrics import BACKTEST_MIN_PERIODS
from app.services.trading_costs import roundtrip_cost_rate
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
from app.services.high_winrate_combo_goal_utils import (
    combo_rejection,
    combo_return_metrics,
    nested_frames,
    nested_split_payload,
    profit_factor,
    selected_hits,
)
from app.services.high_winrate_combo_goal_types import ComboHit, OrientedScore, RankedSearch, ScoreSearch
from app.services import high_winrate_combo_goal_diagnostics as diag
from app.services.high_winrate_combo_thresholds import (
    best_threshold_hit,
    best_threshold_hit_with_diagnostics,
    combo_score_array,
    threshold_frame,
    threshold_hit_result,
)

TARGET_WIN_RATE = 0.62
TARGET_COUNT = 5
TARGET_MIN_TRADES = BACKTEST_MIN_PERIODS
NEXT_ENTRY_HORIZON_BARS = 1
ZSCORE_CLIP = 4.0
COMBO_SIZES = (2, 3, 4)
EXCLUDED_COLUMNS = frozenset({"open_time", "open", "high", "low", "close", "volume", "fwd_ret"})
TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
NESTED_VALIDATION_MIN_TRADES = 20
NESTED_TRAIN_MIN_TRADES = 20


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
    training_frame = _nested_frames(frame)["train"]
    for name in numeric_factor_columns(frame):
        series = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        train_series = series.reindex(training_frame.index)
        usable_pairs = usable_pair_count(train_series, training_frame["fwd_ret"])
        max_valid_pairs = max(max_valid_pairs, usable_pairs)
        if usable_pairs < BACKTEST_MIN_PERIODS:
            rejected.append((name, usable_pairs))
            continue
        orientation = orientation_for_series(train_series, training_frame["fwd_ret"])
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
    split = _nested_frames(frame)
    train_min_trades = min(cfg.min_trades, NESTED_TRAIN_MIN_TRADES)
    names = search_candidate_names(split["train"], scores, cfg, train_min_trades)
    payload = diag.ranked_search_diagnostics(names, cfg.candidate_limit)
    payload["selectionMode"] = "train_threshold_validation_combo_v1"
    payload["nestedSplit"] = _nested_split_payload(frame, split)
    payload["testedValidationEvaluations"] = 0
    train_threshold_frame = threshold_frame(split["train"], scores)
    validation_threshold_frame = threshold_frame(split["validation"], scores)
    for size in COMBO_SIZES:
        for members in combinations(names, size):
            payload["testedCombinations"] += 1
            orientations = tuple(scores[name].orientation for name in members)
            train_score = combo_score_array(members, train_threshold_frame)
            train_best = best_threshold_hit_with_diagnostics(
                members,
                orientations,
                train_score,
                train_threshold_frame.index,
                train_threshold_frame,
                cfg.signal_thresholds,
                payload,
                min_win_rate=TARGET_WIN_RATE,
                min_trades=train_min_trades,
            )
            if train_best is None:
                continue
            validation_score = combo_score_array(members, validation_threshold_frame)
            validation_hit, rejected = threshold_hit_result(
                members,
                orientations,
                validation_score,
                validation_threshold_frame.index,
                validation_threshold_frame,
                train_best.threshold,
                min_trades=min(cfg.min_trades, NESTED_VALIDATION_MIN_TRADES),
                min_win_rate=TARGET_WIN_RATE,
            )
            payload["testedValidationEvaluations"] += 1
            diag.record_combo_gate_result(payload, validation_hit, rejected)
            if validation_hit is not None:
                hits[validation_hit.members] = validation_hit
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
    min_trades: int | None = None,
) -> list[str]:
    cfg = validated_search_config(config)
    search_data = threshold_frame(frame, scores)
    hits = [
        row for row in (
            best_threshold_hit(
                (name,),
                (scores[name].orientation,),
                search_data.scores[name],
                search_data.index,
                search_data,
                cfg.signal_thresholds,
                min_win_rate=0.0,
                min_trades=min_trades or cfg.min_trades,
            )
            for name in scores
        )
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
    min_trades: int | None = None,
    config: GoalSearchConfig | None = None,
) -> ComboHit | None:
    cfg = validated_search_config(config)
    combo_score = sum(scores[name].score for name in members) / len(members)
    orientations = tuple(scores[name].orientation for name in members)
    required_trades = int(min_trades if min_trades is not None else cfg.min_trades)
    candidates = [
        combo_hit(frame, members, orientations, combo_score, threshold, min_win_rate=min_win_rate, min_trades=required_trades)
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
    min_trades: int | None = None,
) -> ComboHit | None:
    cfg = validated_search_config(config)
    diagnostics["testedCombinations"] += 1
    combo_score = sum(scores[name].score for name in members) / len(members)
    orientations = tuple(scores[name].orientation for name in members)
    candidates = []
    for threshold in cfg.signal_thresholds:
        hit, rejected = combo_hit_result(frame, members, orientations, combo_score, threshold, cfg, min_trades=min_trades)
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
    min_trades: int | None = None,
    min_win_rate: float = TARGET_WIN_RATE,
) -> tuple[ComboHit | None, dict[str, Any] | None]:
    cfg = validated_search_config(config)
    aligned_score = score.reindex(frame.index)
    signal = pd.Series(
        np.where(aligned_score >= threshold, 1.0, np.where(aligned_score <= -threshold, -1.0, np.nan)),
        index=frame.index,
    )
    returns = (signal * frame["fwd_ret"] - roundtrip_cost_rate()).replace([np.inf, -np.inf], np.nan).dropna()
    metrics = combo_return_metrics(returns)
    required_trades = int(min_trades if min_trades is not None else cfg.min_trades)
    if metrics["trades"] < required_trades:
        return None, combo_rejection(members, threshold, "min_trades_below_min", metrics)
    if metrics["winRate"] < min_win_rate:
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
    aligned_score = score.reindex(frame.index)
    signal = pd.Series(
        np.where(aligned_score >= threshold, 1.0, np.where(aligned_score <= -threshold, -1.0, np.nan)),
        index=frame.index,
    )
    returns = (signal * frame["fwd_ret"] - roundtrip_cost_rate()).replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < int(min_trades):
        return None
    win_rate = float((returns > 0).mean())
    factor = profit_factor(returns)
    if win_rate < min_win_rate or factor < SUCCESS_PROFIT_FACTOR_MIN:
        return None
    return ComboHit(members, orientations, threshold, win_rate, factor, len(returns), float(returns.mean()), score)


def validated_search_config(config: GoalSearchConfig | None = None) -> GoalSearchConfig:
    return _validated_search_config(config, TARGET_MIN_TRADES)


def _nested_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return nested_frames(frame, TRAIN_RATIO, VALIDATION_RATIO)


def _nested_split_payload(frame: pd.DataFrame, split: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return nested_split_payload(frame, split, TRAIN_RATIO, VALIDATION_RATIO)
