from __future__ import annotations

from math import isfinite
from typing import Any

import pandas as pd

from app.services.factor_combination_service import COMBINATION_METHOD, combination_score
from app.services.factor_learning_signal_filter import enrich_signal_with_factor_learning
from app.services.kline_timing import is_within_entry_grace

LIVE_MIN_WIN_RATE = 0.5
PROBABILITY_DECIMALS = 4
SCORE_DECIMALS = 6


def build_live_signal_from_ranking(
    frame: pd.DataFrame,
    row: dict[str, Any],
    *,
    symbol: str,
    duration: str,
    entry_open_time: int | None = None,
    entry_grace_ms: int | None = None,
) -> dict[str, Any]:
    score, index = _latest_combo_score(frame, row)
    direction = "up" if score >= 0 else "down"
    confidence = _live_confidence(row)
    quality_passed = _quality_passed(confidence, entry_open_time, entry_grace_ms)
    payload = _live_signal_payload(row, symbol, duration, score, index, direction, confidence, quality_passed)
    priced = {
        **payload,
        "entryPrice": _frame_value(frame, index, "close"),
        "sourceOpenTime": _frame_value(frame, index, "open_time"),
    }
    return enrich_signal_with_factor_learning(priced, frame, index, symbol=symbol, duration=duration)


def _latest_combo_score(frame: pd.DataFrame, row: dict[str, Any]) -> tuple[float, Any]:
    members = _row_members(row)
    missing = [member["name"] for member in members if member["name"] not in frame.columns]
    if missing:
        raise ValueError(f"combination signal missing factors: {', '.join(missing)}")
    scores = combination_score(frame, members).dropna()
    if scores.empty:
        raise ValueError(f"combination signal has no finite score: {row.get('factorName')}")
    return float(scores.iloc[-1]), scores.index[-1]


def _live_signal_payload(
    row: dict[str, Any],
    symbol: str,
    duration: str,
    score: float,
    index: Any,
    direction: str,
    confidence: float,
    quality_passed: bool,
) -> dict[str, Any]:
    probability_up = confidence if direction == "up" else 1.0 - confidence
    return {
        "symbol": symbol.upper(),
        "duration": duration,
        "factorName": row.get("factorName"),
        "factorDisplayName": row.get("factorDisplayName"),
        "members": _row_members(row),
        "direction": direction,
        "probabilityUp": round(probability_up, PROBABILITY_DECIMALS),
        "confidence": round(confidence, PROBABILITY_DECIMALS),
        "score": round(score, SCORE_DECIMALS),
        "source": "factor_combination_ranking",
        "method": row.get("method") or COMBINATION_METHOD,
        "historicalWinRate": row.get("winRate"),
        "historicalSharpe": row.get("sharpe"),
        "qualityPassed": quality_passed,
        "qualityMinWinRate": LIVE_MIN_WIN_RATE,
        "frameIndex": str(index),
    }


def _quality_passed(
    confidence: float,
    entry_open_time: int | None,
    entry_grace_ms: int | None,
) -> bool:
    if confidence < LIVE_MIN_WIN_RATE:
        return False
    if entry_open_time is None or entry_grace_ms is None:
        return True
    return is_within_entry_grace(int(entry_open_time), grace_ms=int(entry_grace_ms))


def _row_members(row: dict[str, Any]) -> list[dict[str, Any]]:
    members = row.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"combination row missing members: {row.get('factorName')}")
    return [dict(member) for member in members]


def _live_confidence(row: dict[str, Any]) -> float:
    value = _finite_float(row.get("winRate"))
    if value is None:
        raise ValueError(f"combination row missing winRate: {row.get('factorName')}")
    return max(0.0, min(float(value), 0.99))


def _frame_value(frame: pd.DataFrame, index: Any, column: str) -> float | int | None:
    if column not in frame.columns:
        return None
    value = frame.at[index, column]
    if value is None:
        return None
    number = float(value)
    if not isfinite(number):
        return None
    return int(number) if column == "open_time" else number


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None
