from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.ensemble_judge_constants import (
    ENSEMBLE_MARGIN_THRESHOLD,
    ENSEMBLE_RANKER_STRATEGY_KEY,
    MIN_ENSEMBLE_CANDIDATES,
    STAGE_ENSEMBLE_READY,
)
from app.services.retired_strategy_keys import is_retired_strategy_key, retired_strategy_sql_filter


def predict_ensemble_ranker_prediction(symbol: str, duration: str, *, entry_open_time: int) -> dict[str, Any]:
    sym = symbol.strip().upper()
    conn = get_conn()
    try:
        _assert_ensemble_ready(conn, sym, duration)
        candidates = _weighted_candidates(conn, sym, duration, entry_open_time)
        if len(candidates) < MIN_ENSEMBLE_CANDIDATES:
            raise ValueError("insufficient_ensemble_candidates")
        return _prediction_payload(sym, duration, entry_open_time, candidates)
    finally:
        conn.close()


def _assert_ensemble_ready(conn: Any, symbol: str, duration: str) -> None:
    row = conn.execute(
        """
        SELECT confirmed_stage
        FROM ensemble_stage_status
        WHERE symbol = ? AND duration = ?
        """,
        (symbol, duration),
    ).fetchone()
    if row is None or row["confirmed_stage"] != STAGE_ENSEMBLE_READY:
        raise ValueError("ensemble_stage_not_confirmed")


def _weighted_candidates(conn: Any, symbol: str, duration: str, open_time: int) -> list[dict[str, Any]]:
    retired_filter, retired_params = retired_strategy_sql_filter(table_prefix="p")
    rows = conn.execute(
        f"""
        SELECT p.signal_key, p.strategy_key, p.direction, p.probability_up, p.expected_return,
               s.weight_suggestion
        FROM predictions p
        JOIN ensemble_signal_scores s
          ON s.symbol = p.symbol AND s.duration = p.duration AND s.signal_key = p.signal_key
        WHERE p.symbol = ? AND p.duration = ? AND p.open_time = ?
          AND p.signal_key != ?{retired_filter}
        ORDER BY p.id
        """,
        (symbol, duration, int(open_time), ENSEMBLE_RANKER_STRATEGY_KEY, *retired_params),
    ).fetchall()
    return [
        dict(row)
        for row in rows
        if float(row["weight_suggestion"] or 0) > 0
        and not is_retired_strategy_key(str(row["signal_key"]))
        and not is_retired_strategy_key(str(row["strategy_key"]))
    ]


def _prediction_payload(symbol: str, duration: str, open_time: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_weight = sum(float(row["weight_suggestion"]) for row in rows)
    if total_weight <= 0:
        raise ValueError("insufficient_ensemble_candidates")
    probability_up = _weighted_probability_up(rows, total_weight)
    margin = abs(probability_up - 0.5) * 2
    expected_return = _weighted_expected_return(rows, total_weight)
    direction = "up" if probability_up >= 0.5 else "down"
    return {
        "signal_key": ENSEMBLE_RANKER_STRATEGY_KEY,
        "strategy_key": ENSEMBLE_RANKER_STRATEGY_KEY,
        "symbol": symbol,
        "duration": duration,
        "open_time": int(open_time),
        "direction": direction,
        "probability_up": probability_up,
        "confidence": max(probability_up, 1 - probability_up),
        "certainty_label": "ensemble_simulation",
        "trade_quality_score": margin,
        "trade_quality_passed": margin >= ENSEMBLE_MARGIN_THRESHOLD,
        "trade_quality_gate": "ensemble_margin",
        "expected_return": expected_return,
        "model_version": ENSEMBLE_RANKER_STRATEGY_KEY,
        "model_duration": duration,
        "model_trained_at": _utc_now(),
    }


def _weighted_probability_up(rows: list[dict[str, Any]], total_weight: float) -> float:
    up_score = 0.0
    for row in rows:
        weight = float(row["weight_suggestion"])
        up_score += weight * float(row["probability_up"])
    return up_score / total_weight


def _weighted_expected_return(rows: list[dict[str, Any]], total_weight: float) -> float | None:
    values = [row for row in rows if row["expected_return"] is not None]
    if not values:
        return None
    return sum(float(row["expected_return"]) * float(row["weight_suggestion"]) for row in values) / total_weight


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
