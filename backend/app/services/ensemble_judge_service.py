from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.ensemble_judge_constants import (
    ENSEMBLE_MIN_SETTLED_SAMPLES,
    ENSEMBLE_RANKER_STRATEGY_KEY,
    ENSEMBLE_READY_SAMPLE_COUNT,
    ENSEMBLE_STAGES,
    LOSS_STREAK_THRESHOLD,
    MAJOR_SIGNAL_TYPES,
    STAGE_ENSEMBLE_READY,
    STAGE_OBSERVE,
    STAGE_WEIGHT_READY,
    WEIGHT_READY_MIN_DAYS,
    WEIGHT_READY_SAMPLE_COUNT,
)
from app.services.ensemble_judge_metrics import (
    coverage_from_scores,
    ranking_payload,
    recent_windows,
    score_candidate_rows,
    window_return,
)
from app.services.ensemble_ranking_rows import ensemble_candidate_rows, scored_rows
from app.services.rule_config import DURATION_TO_MINUTES


def refresh_ensemble_judge(symbol: str, duration: str) -> dict[str, Any]:
    sym = _symbol(symbol)
    conn = get_conn()
    try:
        scores = score_candidate_rows(_settled_candidate_rows(conn, sym, duration))
        _replace_scores(conn, sym, duration, scores)
        status = _upsert_stage_status(conn, sym, duration, scores)
        conn.commit()
        return {"status": status, "ranking": ranking_payload(scores)}
    finally:
        conn.close()


def ensemble_status(symbol: str, duration: str) -> dict[str, Any]:
    sym = _symbol(symbol)
    conn = get_conn()
    try:
        row = _stage_row(conn, sym, duration)
        scores = scored_rows(conn, sym, duration)
        return _status_payload(dict(row), scores) if row else _default_status(sym, duration, scores)
    finally:
        conn.close()


def ensemble_ranking(symbol: str, duration: str) -> dict[str, Any]:
    sym = _symbol(symbol)
    conn = get_conn()
    try:
        rows = ensemble_candidate_rows(conn, sym, duration)
        return {"symbol": sym, "duration": duration, "ranking": ranking_payload(rows)}
    finally:
        conn.close()


def confirm_ensemble_stage(symbol: str, duration: str, stage: str) -> dict[str, Any]:
    if stage not in ENSEMBLE_STAGES:
        raise ValueError(f"unsupported ensemble stage: {stage}")
    sym = _symbol(symbol)
    conn = get_conn()
    try:
        row = _stage_row(conn, sym, duration)
        if row is None:
            raise ValueError("ensemble stage has no recommendation; refresh first")
        if str(row["recommended_stage"]) != stage:
            raise ValueError(f"cannot confirm {stage}; recommended stage is {row['recommended_stage']}")
        _confirm_stage(conn, sym, duration, stage)
        if stage == STAGE_ENSEMBLE_READY:
            _ensure_ensemble_strategy_slots(conn, sym, duration)
        conn.commit()
        return ensemble_status(sym, duration)
    finally:
        conn.close()


def _settled_candidate_rows(conn: Any, symbol: str, duration: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT signal_key, strategy_key, symbol, duration, open_time, direction,
               probability_up, confidence, prediction_correct, actual_return,
               high_winrate_rule, model_version
        FROM predictions
        WHERE symbol = ? AND duration = ? AND settled_at IS NOT NULL
          AND signal_key != ?
        ORDER BY open_time, id
        """,
        (symbol, duration, ENSEMBLE_RANKER_STRATEGY_KEY),
    ).fetchall()
    return [dict(row) for row in rows]


def _replace_scores(conn: Any, symbol: str, duration: str, scores: list[dict[str, Any]]) -> None:
    conn.execute("DELETE FROM ensemble_signal_scores WHERE symbol = ? AND duration = ?", (symbol, duration))
    for row in scores:
        conn.execute(
            """
            INSERT INTO ensemble_signal_scores
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol, duration, row["signal_key"], row["signal_type"], row["sample_count"],
                row["win_rate"], row["avg_return"], row["profit_factor"], row["consecutive_losses"],
                row["stability_score"], row["weight_suggestion"], row["score"], _utc_now(),
            ),
        )


def _upsert_stage_status(conn: Any, symbol: str, duration: str, scores: list[dict[str, Any]]) -> dict[str, Any]:
    previous = _stage_row(conn, symbol, duration)
    confirmed = previous["confirmed_stage"] if previous else None
    confirmed_at = previous["confirmed_at"] if previous else None
    recommendation = _stage_recommendation(conn, symbol, duration, scores)
    now = _utc_now()
    conn.execute(
        """
        INSERT OR REPLACE INTO ensemble_stage_status
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            symbol, duration, confirmed or STAGE_OBSERVE, recommendation["stage"],
            recommendation["reason"], confirmed, confirmed_at, now,
        ),
    )
    row = _stage_row(conn, symbol, duration)
    return _status_payload(dict(row), scored_rows(conn, symbol, duration))


def _stage_recommendation(conn: Any, symbol: str, duration: str, scores: list[Any]) -> dict[str, str]:
    coverage = coverage_from_scores(conn, symbol, duration, scores)
    if _ensemble_ready(conn, symbol, duration, coverage):
        return {"stage": STAGE_ENSEMBLE_READY, "reason": "ensemble simulation samples and recent windows are ready"}
    if _weight_ready(coverage):
        return {"stage": STAGE_WEIGHT_READY, "reason": "major signal sources have enough settled samples and stable recent windows"}
    return {"stage": STAGE_OBSERVE, "reason": "waiting for settled samples across major signal sources"}


def _weight_ready(coverage: dict[str, Any]) -> bool:
    by_type = coverage["byMajorSignalType"]
    for signal_type in MAJOR_SIGNAL_TYPES:
        item = by_type.get(signal_type)
        if not item or item["sampleCount"] < WEIGHT_READY_SAMPLE_COUNT:
            return False
        if item["distinctTradingDays"] < WEIGHT_READY_MIN_DAYS:
            return False
        if item["recentProfitFactorBelowOne"] or item["maxConsecutiveLosses"] >= LOSS_STREAK_THRESHOLD:
            return False
    return True


def _ensemble_ready(conn: Any, symbol: str, duration: str, coverage: dict[str, Any]) -> bool:
    if not all(
        coverage["byMajorSignalType"].get(kind, {}).get("sampleCount", 0) >= ENSEMBLE_READY_SAMPLE_COUNT
        for kind in MAJOR_SIGNAL_TYPES
    ):
        return False
    rows = _settled_ensemble_rows(conn, symbol, duration)
    if len(rows) < ENSEMBLE_MIN_SETTLED_SAMPLES:
        return False
    return all(window_return(window) > 0 for window in recent_windows(rows))


def _status_payload(row: dict[str, Any], scores: list[Any]) -> dict[str, Any]:
    conn = get_conn()
    try:
        coverage = coverage_from_scores(conn, row["symbol"], row["duration"], scores)
    finally:
        conn.close()
    return {
        "symbol": row["symbol"],
        "duration": row["duration"],
        "stage": row["stage"],
        "recommendedStage": row["recommended_stage"],
        "recommendationReason": row["recommendation_reason"],
        "confirmedStage": row["confirmed_stage"],
        "confirmedAt": row["confirmed_at"],
        "updatedAt": row["updated_at"],
        "sampleCoverage": coverage,
    }


def _default_status(symbol: str, duration: str, scores: list[Any]) -> dict[str, Any]:
    row = {
        "symbol": symbol,
        "duration": duration,
        "stage": STAGE_OBSERVE,
        "recommended_stage": STAGE_OBSERVE,
        "recommendation_reason": "refresh has not run yet",
        "confirmed_stage": None,
        "confirmed_at": None,
        "updated_at": None,
    }
    return _status_payload(row, scores)


def _stage_row(conn: Any, symbol: str, duration: str) -> Any | None:
    return conn.execute(
        "SELECT * FROM ensemble_stage_status WHERE symbol = ? AND duration = ?",
        (symbol, duration),
    ).fetchone()


def _settled_ensemble_rows(conn: Any, symbol: str, duration: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT open_time, actual_return
        FROM predictions
        WHERE signal_key = ? AND symbol = ? AND duration = ? AND settled_at IS NOT NULL
        ORDER BY open_time
        """,
        (ENSEMBLE_RANKER_STRATEGY_KEY, symbol, duration),
    ).fetchall()
    return [dict(row) for row in rows]


def _confirm_stage(conn: Any, symbol: str, duration: str, stage: str) -> None:
    now = _utc_now()
    conn.execute(
        """
        UPDATE ensemble_stage_status
        SET stage = ?, confirmed_stage = ?, confirmed_at = ?, updated_at = ?
        WHERE symbol = ? AND duration = ?
        """,
        (stage, stage, now, now, symbol, duration),
    )


def _ensure_ensemble_strategy_slots(conn: Any, symbol: str, confirmed_duration: str) -> None:
    now = _utc_now()
    for duration in ("10m", "30m", "60m", "1d"):
        enabled = int(duration == confirmed_duration)
        conn.execute(
            """
            INSERT OR IGNORE INTO auto_trade_strategies(
              strategy_key, duration, enabled, live_trading_enabled,
              symbol, duration_minutes, qty, updated_at
            )
            VALUES(?, ?, ?, 0, ?, ?, 5, ?)
            """,
            (ENSEMBLE_RANKER_STRATEGY_KEY, duration, enabled, symbol, DURATION_TO_MINUTES[duration], now),
        )
        if enabled:
            conn.execute(
                """
                UPDATE auto_trade_strategies
                SET enabled = 1, live_trading_enabled = 0, symbol = ?, updated_at = ?
                WHERE strategy_key = ? AND duration = ?
                """,
                (symbol, now, ENSEMBLE_RANKER_STRATEGY_KEY, duration),
            )


def _symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
