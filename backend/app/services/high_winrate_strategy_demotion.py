from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from app.db.session import get_conn, run_db_write_with_retry
from app.services.background_loop_status import record_loop_failure
from app.services.high_winrate_strategy_metrics import (
    ACTIVE_SAMPLE_COUNT,
    ACTIVE_WIN_RATE_MIN,
    LOSS_STREAK_LIMIT,
    MIN_PROFIT_FACTOR,
    empty_high_winrate_metrics,
    high_winrate_decision,
    high_winrate_metrics,
    high_winrate_thresholds,
)
from app.services.high_winrate_strategy_rotation import (
    DEFAULT_ACTIVE_RANK,
    RANKING_REFRESH_REASON,
    ROTATED_REASON,
    ensure_high_winrate_status_table,
    failed_rank_payload,
    high_winrate_active_rank_from_status,
    high_winrate_candidate_rule,
    high_winrate_failed_ranks_from_status,
    high_winrate_rotation_payload,
    next_high_winrate_candidate_rank,
    refresh_high_winrate_goal,
)
from app.services.event_pnl_rows import settled_event_rows_for_high_winrate_rule
from app.services.strategy_registry import HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY
from app.services.high_winrate_strategy_status_store import (
    current_status as _current_status,
    has_rankings as _has_rankings,
    public_status as _public_status,
    payload_with_live_readiness as _payload_with_live_readiness,
    refresh_failed_payload as _refresh_failed_payload,
    set_strategy_slot as _set_strategy_slot,
    status_payload as _status_payload,
    sync_strategy_slot_for_status as _sync_strategy_slot_for_status,
    utc_now as _utc_now,
    write_status as _write_status,
)

logger = logging.getLogger(__name__)
STATUS_BACKTEST_CANDIDATE = "backtest_candidate"
STATUS_PAPER_COLLECTING = "paper_collecting"
STATUS_PAPER_STABLE = "paper_stable"
STATUS_PAPER_FAILED = "paper_failed"
STATUS_PAPER_LIVE_COLLECTING = STATUS_PAPER_COLLECTING
STATUS_PAPER_LIVE_PASSED = STATUS_PAPER_STABLE
STATUS_TRADABLE = STATUS_PAPER_STABLE
STATUS_DEMOTED = STATUS_PAPER_FAILED
STATUS_PAUSED = "paused"
STATUS_ACTIVE = STATUS_TRADABLE
STATUS_COLLECTING = STATUS_PAPER_COLLECTING
REASON_OFFLINE_PROMOTION = "offline_promotion"
RANKING_REFRESH_FAILED_REASON = "candidate_pool_exhausted_refresh_failed"
RANKING_REFRESH_PENDING_REASON = "ranking_refresh_pending"
RECENT_SAMPLE_LIMIT = 100
LOOP_NAME = "high_winrate_demotion"


@dataclass(frozen=True)
class SettledRowsQuery:
    conn: Any
    symbol: str
    duration: str
    rule: str | None


@dataclass(frozen=True)
class EvaluationContext:
    decision: dict[str, str]
    current: dict[str, Any]
    metrics: dict[str, Any]
    symbol: str
    duration: str
    rank: int
    rule: str | None


def promote_high_winrate_strategy(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    metrics = empty_high_winrate_metrics()
    rotation = high_winrate_rotation_payload(sym, duration, DEFAULT_ACTIVE_RANK)
    payload = _status_payload(STATUS_PAPER_COLLECTING, REASON_OFFLINE_PROMOTION, metrics, rotation)
    conn = get_conn()
    try:
        ensure_high_winrate_status_table(conn)
        _set_strategy_slot(conn, sym, duration, enabled=True, live_trading_enabled=False)
        _write_status(conn, sym, duration, payload)
        conn.commit()
    finally:
        conn.close()
    return {"symbol": sym, "duration": duration, **payload}


def evaluate_high_winrate_demotion(
    symbol: str,
    duration: str,
    *,
    allow_goal_refresh: bool = False,
) -> dict[str, Any]:
    return run_db_write_with_retry(
        lambda: _evaluate_high_winrate_demotion(symbol, duration, allow_goal_refresh=allow_goal_refresh)
    )


def run_pending_high_winrate_goal_refresh(symbol: str, duration: str) -> dict[str, Any] | None:
    sym = symbol.strip().upper()
    status = high_winrate_demotion_status(sym, duration)
    if status.get("reason") != RANKING_REFRESH_PENDING_REASON:
        return None
    return run_db_write_with_retry(lambda: _execute_high_winrate_goal_refresh(sym, duration, status))


def _evaluate_high_winrate_demotion(
    symbol: str,
    duration: str,
    *,
    allow_goal_refresh: bool = False,
) -> dict[str, Any]:
    sym = symbol.strip().upper()
    refresh_required = False
    conn = get_conn()
    try:
        ensure_high_winrate_status_table(conn)
        current = _current_status(conn, sym, duration)
        rank = high_winrate_active_rank_from_status(current)
        rule = high_winrate_candidate_rule(sym, duration, rank)
        rows = _settled_rows(SettledRowsQuery(conn, sym, duration, rule))
        metrics = high_winrate_metrics(rows)
        decision = high_winrate_decision(metrics)
        payload, refresh_required = _evaluation_payload(
            EvaluationContext(decision, current, metrics, sym, duration, rank, rule)
        )
        if refresh_required and not allow_goal_refresh:
            payload = _pending_refresh_payload(payload)
            refresh_required = False
        _sync_strategy_slot_for_status(conn, sym, duration, payload["status"])
        _write_status(conn, sym, duration, payload)
        conn.commit()
    finally:
        conn.close()
    if refresh_required:
        return _execute_high_winrate_goal_refresh(sym, duration, payload)
    return {"symbol": sym, "duration": duration, **payload}


def _execute_high_winrate_goal_refresh(
    symbol: str,
    duration: str,
    fallback_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sym = symbol.strip().upper()
    report = refresh_high_winrate_goal(sym, duration)
    if _has_rankings(report):
        return high_winrate_demotion_status(sym, duration)
    payload = _refresh_failed_payload(fallback_payload or {}, report)
    conn = get_conn()
    try:
        ensure_high_winrate_status_table(conn)
        _sync_strategy_slot_for_status(conn, sym, duration, payload["status"])
        _write_status(conn, sym, duration, payload)
        conn.commit()
    finally:
        conn.close()
    return high_winrate_demotion_status(sym, duration)


def high_winrate_demotion_status(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    conn = get_conn()
    try:
        ensure_high_winrate_status_table(conn)
        row = _current_status(conn, sym, duration)
    finally:
        conn.close()
    return _public_status(row, sym, duration)


def high_winrate_active_rank(symbol: str, duration: str) -> int:
    sym = symbol.strip().upper()
    conn = get_conn()
    try:
        ensure_high_winrate_status_table(conn)
        row = _current_status(conn, sym, duration)
    finally:
        conn.close()
    return high_winrate_active_rank_from_status(row)


def _settled_rows(query: SettledRowsQuery) -> list[dict[str, Any]]:
    try:
        event_rows = settled_event_rows_for_high_winrate_rule(query.conn, query.symbol, query.duration, query.rule)
    except Exception as exc:
        details = {
            "stage": "settled_event_rows",
            "symbol": query.symbol,
            "duration": query.duration,
            "rule": query.rule,
        }
        record_loop_failure(LOOP_NAME, exc, details)
        logger.exception("high winrate settled event rows failed: %s %s", query.symbol, query.duration)
        event_rows = []
    if event_rows:
        return event_rows
    return _settled_prediction_rows(query)


def _settled_prediction_rows(query: SettledRowsQuery) -> list[dict[str, Any]]:
    rule_clause = "" if query.rule is None else " AND high_winrate_rule = ?"
    params = [HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, query.symbol, query.duration]
    if query.rule is not None:
        params.append(query.rule)
    params.append(RECENT_SAMPLE_LIMIT)
    rows = query.conn.execute(
        f"""
        SELECT open_time, prediction_correct, actual_return, high_winrate_rule
        FROM predictions
        WHERE strategy_key = ? AND symbol = ? AND duration = ?
          AND settled_at IS NOT NULL
          {rule_clause}
        ORDER BY open_time DESC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    return [dict(row) for row in rows]


def _evaluation_payload(context: EvaluationContext) -> tuple[dict[str, Any], bool]:
    if context.current.get("status") == STATUS_PAUSED:
        payload = _status_payload(STATUS_PAUSED, str(context.current.get("reason") or "paused"), context.metrics)
        return payload, False
    if context.decision["status"] != STATUS_PAPER_FAILED or context.rule is None:
        failed_ranks = high_winrate_failed_ranks_from_status(context.current)
        rotation = high_winrate_rotation_payload(context.symbol, context.duration, context.rank, failed_ranks)
        return _status_payload(context.decision["status"], context.decision["reason"], context.metrics, rotation), False
    failed = (*high_winrate_failed_ranks_from_status(context.current), context.rank)
    previous = failed_rank_payload(context.rank, context.rule, context.decision, context.metrics)
    next_rank = next_high_winrate_candidate_rank(context.rank)
    if next_rank is None:
        rotation = high_winrate_rotation_payload(context.symbol, context.duration, context.rank, failed, previous)
        return _status_payload(STATUS_DEMOTED, RANKING_REFRESH_REASON, context.metrics, rotation), True
    next_rule = high_winrate_candidate_rule(context.symbol, context.duration, next_rank)
    rotation = high_winrate_rotation_payload(context.symbol, context.duration, next_rank, failed, previous, active_rule=next_rule)
    payload = _status_payload(STATUS_PAPER_COLLECTING, ROTATED_REASON, empty_high_winrate_metrics(), rotation)
    return payload, False


def _pending_refresh_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _payload_with_live_readiness({
        **payload,
        "status": STATUS_PAPER_FAILED,
        "reason": RANKING_REFRESH_PENDING_REASON,
        "pendingGoalRefresh": True,
    })
