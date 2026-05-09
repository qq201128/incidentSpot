from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.services.trade_quality_gate import load_quality_gate, quality_score

MS_PER_DAY = 1000.0 * 60 * 60 * 24
ROUNDTRIP_COST = 0.001
CONFIDENCE_THRESHOLDS = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999)
QUALITY_THRESHOLDS = (0.50, 0.60, 0.70, 0.75, 0.78, 0.80, 0.85, 0.90, 0.95, 0.99)


@dataclass(frozen=True)
class BacktestProfileContext:
    test_frame: pd.DataFrame
    test_side: np.ndarray
    test_confidence: np.ndarray
    test_fwd_ret: np.ndarray
    min_trade_gap_minutes: int


def build_confidence_profiles(context: BacktestProfileContext) -> list[dict]:
    return [
        _build_profile(context, context.test_confidence >= threshold, "trade_confidence_threshold", threshold)
        for threshold in CONFIDENCE_THRESHOLDS
    ]


def build_quality_profiles(
    context: BacktestProfileContext,
    feature_cols: list[str],
    duration: str,
) -> list[dict]:
    config = load_quality_gate(duration)
    scores = _quality_scores(context.test_frame, context.test_side, feature_cols, config)
    confidence_min = float(config["trade_confidence_threshold"])
    confidence_mask = context.test_confidence >= confidence_min
    return [
        {
            **_build_profile(context, confidence_mask & (scores >= threshold), "trade_quality_score_threshold", threshold),
            "trade_confidence_threshold": confidence_min,
        }
        for threshold in QUALITY_THRESHOLDS
    ]


def build_selected_confidence_profile(context: BacktestProfileContext, threshold: float) -> dict:
    return _build_profile(
        context,
        context.test_confidence >= float(threshold),
        "trade_confidence_threshold",
        float(threshold),
    )


def _quality_scores(
    test_frame: pd.DataFrame,
    test_side: np.ndarray,
    feature_cols: list[str],
    config: dict,
) -> np.ndarray:
    rows = test_frame[feature_cols].to_dict(orient="records")
    directions = np.where(test_side > 0, "up", "down")
    return np.array(
        [quality_score(row, direction, config) for row, direction in zip(rows, directions)],
        dtype=float,
    )


def _build_profile(
    context: BacktestProfileContext,
    candidate: np.ndarray,
    threshold_key: str,
    threshold: float,
) -> dict:
    tradable = _apply_trade_gap(candidate, context.test_frame["open_time"], context.min_trade_gap_minutes)
    pnl = np.zeros_like(context.test_fwd_ret, dtype=float)
    pnl[tradable] = context.test_side[tradable] * context.test_fwd_ret[tradable] - ROUNDTRIP_COST
    traded_pnl = pnl[tradable]
    hit = _direction_hits(context, tradable)
    trades = int(tradable.sum())
    return {
        threshold_key: float(threshold),
        "min_trade_gap_minutes": int(context.min_trade_gap_minutes),
        "test_rows": int(len(pnl)),
        "test_trades": trades,
        "trades_per_day": float(trades / _span_days(context.test_frame["open_time"])),
        "roundtrip_cost_rate": ROUNDTRIP_COST,
        "strategy_return": _strategy_return(pnl),
        "buy_hold_return": _buy_hold_return(context.test_fwd_ret),
        "win_rate": float((traded_pnl > 0).mean()) if len(traded_pnl) else 0.0,
        "direction_hit_rate": float(hit.mean()) if len(hit) else 0.0,
        "avg_trade_return": float(traded_pnl.mean()) if len(traded_pnl) else 0.0,
    }


def _apply_trade_gap(candidate: np.ndarray, open_time: pd.Series, gap_minutes: int) -> np.ndarray:
    if int(gap_minutes) <= 0:
        return candidate.copy()
    tradable = np.zeros_like(candidate, dtype=bool)
    min_gap_ms = int(gap_minutes) * 60 * 1000
    last_trade_time = None
    for index, ok in enumerate(candidate):
        if not ok:
            continue
        current_time = float(open_time.iloc[index])
        if last_trade_time is None or (current_time - last_trade_time) >= min_gap_ms:
            tradable[index] = True
            last_trade_time = current_time
    return tradable


def _direction_hits(context: BacktestProfileContext, tradable: np.ndarray) -> np.ndarray:
    if int(tradable.sum()) == 0:
        return np.array([])
    return (context.test_side[tradable] > 0) == (context.test_fwd_ret[tradable] > 0)


def _span_days(open_time: pd.Series) -> float:
    if len(open_time) < 2:
        return 1e-9
    span = (float(open_time.iloc[-1]) - float(open_time.iloc[0])) / MS_PER_DAY
    return max(span, 1e-9)


def _strategy_return(pnl: np.ndarray) -> float:
    equity = np.cumprod(1.0 + pnl)
    return float(equity[-1] - 1.0) if len(equity) else 0.0


def _buy_hold_return(test_fwd_ret: np.ndarray) -> float:
    return float(np.prod(1.0 + test_fwd_ret) - 1.0) if len(test_fwd_ret) else 0.0
