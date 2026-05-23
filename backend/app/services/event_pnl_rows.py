from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.event_ai_history import settled_expected_profit_usdt
from app.services.factor_combo_simulation_keys import (
    BATCH_COMBO_KEY_PREFIX,
    BATCH_HIGH_WINRATE_KEY_PREFIX,
    is_batch_combo_simulation_strategy,
)
from app.services.strategy_registry import (
    FACTOR_COMBO_STRATEGY_KEY,
    HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
)

RECENT_EVENT_SAMPLE_LIMIT = 30
SETTLED_EVENT_JOIN = """
    FROM events e
    LEFT JOIN orders o ON o.id = (
        SELECT id FROM orders WHERE event_id = e.id ORDER BY id DESC LIMIT 1
    )
    WHERE e.status = 'SETTLED'
      AND e.symbol = ?
      AND e.event_interval = ?
"""


def _events_table_available(conn: Any) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'events'"
    ).fetchone()
    return row is not None


def settled_event_metric_rows(
    conn: Any,
    symbol: str,
    duration: str,
    *,
    strategy_key: str | None = None,
    high_winrate_rule: str | None = None,
    limit: int = RECENT_EVENT_SAMPLE_LIMIT,
) -> list[dict[str, Any]]:
    if not _events_table_available(conn):
        return []
    clauses = [SETTLED_EVENT_JOIN.strip()]
    params: list[Any] = [symbol.strip().upper(), duration]
    if strategy_key is not None:
        clauses.append("AND e.strategy_key = ?")
        params.append(strategy_key)
    if high_winrate_rule is not None:
        clauses.append("AND e.ai_high_winrate_rule = ?")
        params.append(high_winrate_rule)
    params.append(int(limit))
    rows = conn.execute(
        f"""
        SELECT
          e.id AS event_id,
          e.strategy_key,
          e.start_time,
          e.end_time,
          e.prediction_open_time,
          e.ai_predicted_direction,
          e.ai_prediction_correct,
          e.ai_high_winrate_rule,
          e.result,
          o.side AS order_side,
          o.qty AS order_qty,
          o.price AS order_price
        {' '.join(clauses)}
        ORDER BY e.start_time DESC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    return [_metric_row(dict(row)) for row in rows]


def settled_event_rows_for_high_winrate_rule(
    conn: Any,
    symbol: str,
    duration: str,
    rule: str | None,
    *,
    limit: int = RECENT_EVENT_SAMPLE_LIMIT,
) -> list[dict[str, Any]]:
    if not _events_table_available(conn):
        return []
    parent_keys = (HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, FACTOR_COMBO_STRATEGY_KEY)
    if rule is None:
        return settled_event_metric_rows(
            conn,
            symbol,
            duration,
            strategy_key=HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
            limit=limit,
        )
    rows = conn.execute(
        f"""
        SELECT
          e.id AS event_id,
          e.strategy_key,
          e.start_time,
          e.end_time,
          e.prediction_open_time,
          e.ai_predicted_direction,
          e.ai_prediction_correct,
          e.ai_high_winrate_rule,
          e.result,
          o.side AS order_side,
          o.qty AS order_qty,
          o.price AS order_price
        {SETTLED_EVENT_JOIN}
          AND (
            e.strategy_key IN (?, ?)
            OR e.strategy_key LIKE ?
            OR e.strategy_key LIKE ?
          )
          AND e.ai_high_winrate_rule = ?
        ORDER BY e.start_time DESC
        LIMIT ?
        """,
        (
            symbol.strip().upper(),
            duration,
            parent_keys[0],
            parent_keys[1],
            f"{BATCH_HIGH_WINRATE_KEY_PREFIX}%",
            f"{BATCH_COMBO_KEY_PREFIX}%",
            rule,
            int(limit),
        ),
    ).fetchall()
    return [_metric_row(dict(row)) for row in rows]


def batch_combo_strategy_keys(conn: Any, symbol: str, duration: str) -> list[str]:
    if not _events_table_available(conn):
        return []
    rows = conn.execute(
        f"""
        SELECT DISTINCT e.strategy_key
        {SETTLED_EVENT_JOIN}
          AND (
            e.strategy_key LIKE ?
            OR e.strategy_key LIKE ?
          )
        """,
        (
            symbol.strip().upper(),
            duration,
            f"{BATCH_HIGH_WINRATE_KEY_PREFIX}%",
            f"{BATCH_COMBO_KEY_PREFIX}%",
        ),
    ).fetchall()
    return [str(row["strategy_key"]) for row in rows if is_batch_combo_simulation_strategy(row["strategy_key"])]


def _metric_row(row: dict[str, Any]) -> dict[str, Any]:
    pnl = settled_expected_profit_usdt(
        status="SETTLED",
        order_side=row.get("order_side"),
        order_qty=row.get("order_qty"),
        order_price=row.get("order_price"),
        result=row.get("result"),
    )
    qty = _finite_float(row.get("order_qty")) or 1.0
    actual_return = (float(pnl) / qty) if pnl is not None else None
    open_time = row.get("prediction_open_time")
    if open_time is None:
        open_time = _start_time_ms(row.get("start_time"))
    return {
        "event_id": row.get("event_id"),
        "strategy_key": row.get("strategy_key"),
        "open_time": open_time,
        "start_time": row.get("start_time"),
        "prediction_correct": row.get("ai_prediction_correct"),
        "actual_return": actual_return,
        "event_pnl": pnl,
        "order_qty": row.get("order_qty"),
        "high_winrate_rule": row.get("ai_high_winrate_rule"),
        "ai_predicted_direction": row.get("ai_predicted_direction"),
    }


def _start_time_ms(value: str | None) -> int | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        return int(datetime.fromisoformat(normalized).timestamp() * 1000)
    except ValueError:
        return None


def _finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number
