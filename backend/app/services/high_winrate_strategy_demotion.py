from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
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
from app.services.rule_config import DURATION_TO_MINUTES
from app.services.strategy_registry import HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY

STATUS_BACKTEST_CANDIDATE = "backtest_candidate"
STATUS_PAPER_LIVE_COLLECTING = "paper_live_collecting"
STATUS_PAPER_LIVE_PASSED = "paper_live_passed"
STATUS_TRADABLE = "tradable"
STATUS_DEMOTED = "demoted"
STATUS_PAUSED = "paused"
STATUS_ACTIVE = STATUS_TRADABLE
STATUS_COLLECTING = STATUS_PAPER_LIVE_COLLECTING
REASON_OFFLINE_PROMOTION = "offline_promotion"
RECENT_SAMPLE_LIMIT = 30
DEFAULT_QTY = 5.0


def promote_high_winrate_strategy(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    metrics = empty_high_winrate_metrics()
    rotation = high_winrate_rotation_payload(sym, duration, DEFAULT_ACTIVE_RANK)
    payload = _status_payload(STATUS_BACKTEST_CANDIDATE, REASON_OFFLINE_PROMOTION, metrics, rotation)
    conn = get_conn()
    try:
        ensure_high_winrate_status_table(conn)
        _set_strategy_slot(conn, sym, duration, enabled=True, live_trading_enabled=False)
        _write_status(conn, sym, duration, payload)
        conn.commit()
    finally:
        conn.close()
    return {"symbol": sym, "duration": duration, **payload}


def evaluate_high_winrate_demotion(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    refresh_required = False
    conn = get_conn()
    try:
        ensure_high_winrate_status_table(conn)
        current = _current_status(conn, sym, duration)
        rank = high_winrate_active_rank_from_status(current)
        rule = high_winrate_candidate_rule(sym, duration, rank)
        rows = _settled_rows(conn, sym, duration, rule)
        metrics = high_winrate_metrics(rows)
        decision = high_winrate_decision(metrics)
        payload, refresh_required = _evaluation_payload(decision, current, metrics, sym, duration, rank, rule)
        _sync_strategy_slot_for_status(conn, sym, duration, payload["status"])
        _write_status(conn, sym, duration, payload)
        conn.commit()
    finally:
        conn.close()
    if refresh_required:
        refresh_high_winrate_goal(sym, duration)
        return high_winrate_demotion_status(sym, duration)
    return {"symbol": sym, "duration": duration, **payload}


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


def _settled_rows(conn: Any, symbol: str, duration: str, rule: str | None) -> list[dict[str, Any]]:
    rule_clause = "" if rule is None else " AND high_winrate_rule = ?"
    params = [HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, symbol, duration]
    if rule is not None:
        params.append(rule)
    params.append(RECENT_SAMPLE_LIMIT)
    rows = conn.execute(
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


def _evaluation_payload(
    decision: dict[str, str],
    current: dict[str, Any],
    metrics: dict[str, Any],
    symbol: str,
    duration: str,
    rank: int,
    rule: str | None,
) -> tuple[dict[str, Any], bool]:
    if current.get("status") == STATUS_PAUSED:
        payload = _status_payload(STATUS_PAUSED, str(current.get("reason") or "paused"), metrics)
        return payload, False
    if decision["status"] != STATUS_DEMOTED or rule is None:
        rotation = high_winrate_rotation_payload(symbol, duration, rank, high_winrate_failed_ranks_from_status(current))
        return _status_payload(decision["status"], decision["reason"], metrics, rotation), False
    failed = (*high_winrate_failed_ranks_from_status(current), rank)
    previous = failed_rank_payload(rank, rule, decision, metrics)
    next_rank = next_high_winrate_candidate_rank(rank)
    if next_rank is None:
        rotation = high_winrate_rotation_payload(symbol, duration, rank, failed, previous)
        return _status_payload(STATUS_DEMOTED, RANKING_REFRESH_REASON, metrics, rotation), True
    next_rule = high_winrate_candidate_rule(symbol, duration, next_rank)
    rotation = high_winrate_rotation_payload(symbol, duration, next_rank, failed, previous, active_rule=next_rule)
    payload = _status_payload(STATUS_PAPER_LIVE_COLLECTING, ROTATED_REASON, empty_high_winrate_metrics(), rotation)
    return payload, False


def _status_payload(
    status: str,
    reason: str,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "status": status,
        "reason": reason,
        "metrics": metrics,
        "thresholds": high_winrate_thresholds(),
        "sampleCount": metrics.get("sampleCount"),
        "settledSampleCount": metrics.get("sampleCount"),
        "winRate": metrics.get("winRate"),
        "profitFactor": metrics.get("profitFactor"),
        "consecutiveLosses": metrics.get("consecutiveLosses"),
        "requiredSampleCount": high_winrate_thresholds()["requiredSampleCount"],
        "tradable": status == STATUS_TRADABLE,
        "evaluatedAt": _utc_now(),
    }
    if extra:
        payload.update(extra)
    return payload


def _write_status(conn: Any, symbol: str, duration: str, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    conn.execute(
        """
        INSERT INTO high_winrate_strategy_status(
          strategy_key, symbol, duration, status, reason, details_json,
          sample_count, win_rate, profit_factor, consecutive_losses, evaluated_at, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(strategy_key, symbol, duration) DO UPDATE SET
          status = excluded.status, reason = excluded.reason, details_json = excluded.details_json,
          sample_count = excluded.sample_count, win_rate = excluded.win_rate,
          profit_factor = excluded.profit_factor, consecutive_losses = excluded.consecutive_losses,
          evaluated_at = excluded.evaluated_at, updated_at = excluded.updated_at
        """,
        (
            HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
            symbol,
            duration,
            payload["status"],
            payload["reason"],
            json.dumps(payload, ensure_ascii=False),
            int(metrics["sampleCount"]),
            metrics["winRate"],
            metrics["profitFactor"],
            int(metrics["consecutiveLosses"]),
            payload["evaluatedAt"],
            _utc_now(),
        ),
    )


def _current_status(conn: Any, symbol: str, duration: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT status, reason, details_json, evaluated_at
        FROM high_winrate_strategy_status
        WHERE strategy_key = ? AND symbol = ? AND duration = ?
        """,
        (HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, symbol, duration),
    ).fetchone()
    if row is None:
        return {}
    return dict(row)


def _public_status(row: dict[str, Any], symbol: str, duration: str) -> dict[str, Any]:
    if not row:
        return {"symbol": symbol, "duration": duration, "status": "unknown", "reason": "not_evaluated"}
    details = json.loads(row["details_json"]) if row.get("details_json") else {}
    return {
        "symbol": symbol,
        "duration": duration,
        "status": details.get("status") or row.get("status"),
        "reason": details.get("reason") or row.get("reason"),
        **details,
    }


def _set_strategy_slot(
    conn: Any,
    symbol: str,
    duration: str,
    *,
    enabled: bool,
    live_trading_enabled: bool | None,
) -> None:
    row = _strategy_slot(conn, duration)
    live_enabled = _live_trading_enabled(row, live_trading_enabled)
    conn.execute(
        """
        INSERT OR REPLACE INTO auto_trade_strategies(
          strategy_key, duration, enabled, live_trading_enabled, symbol, duration_minutes, qty, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
            duration,
            int(enabled),
            int(live_enabled),
            symbol,
            int(DURATION_TO_MINUTES[duration]),
            float(row["qty"]) if row else DEFAULT_QTY,
            _utc_now(),
        ),
    )


def _sync_strategy_slot_for_status(conn: Any, symbol: str, duration: str, status: str) -> None:
    if status == STATUS_PAUSED:
        _set_strategy_slot(conn, symbol, duration, enabled=False, live_trading_enabled=False)
        return
    live_override = None if status == STATUS_TRADABLE else False
    _set_strategy_slot(conn, symbol, duration, enabled=True, live_trading_enabled=live_override)


def _strategy_slot(conn: Any, duration: str) -> Any | None:
    return conn.execute(
        """
        SELECT qty, live_trading_enabled
        FROM auto_trade_strategies
        WHERE strategy_key = ? AND duration = ?
        """,
        (HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, duration),
    ).fetchone()


def _live_trading_enabled(row: Any | None, override: bool | None) -> bool:
    if override is not None:
        return bool(override)
    return bool(row["live_trading_enabled"]) if row else False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
