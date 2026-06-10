from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.shadow_event_deviation_metrics import (
    by_strategy as _by_strategy,
    summary as _summary,
    utc_now as _utc_now,
)

FUZZY_WINDOW_MS = 900_000

ORDER_JOIN = """
    LEFT JOIN orders o ON o.id = (
        SELECT id FROM orders WHERE event_id = e.id ORDER BY id DESC LIMIT 1
    )
"""

EXACT_PAIR_SELECT = """
    SELECT
      p.id AS prediction_id,
      p.signal_key,
      p.strategy_key,
      p.open_time,
      p.direction,
      p.prediction_correct,
      p.actual_return AS shadow_return,
      e.id AS event_id,
      e.ai_prediction_correct,
      e.result,
      o.side AS order_side,
      o.qty AS order_qty,
      o.price AS order_price
    FROM events e
    INNER JOIN predictions p
      ON p.symbol = e.symbol
     AND p.duration = e.event_interval
     AND p.strategy_key = e.strategy_key
     AND p.open_time = e.prediction_open_time
     AND p.settled_at IS NOT NULL
    {order_join}
    WHERE e.symbol = ?
      AND e.event_interval = ?
      AND e.status = 'SETTLED'
      AND e.prediction_open_time IS NOT NULL
      AND e.market_regime_gate_passed = 1
    ORDER BY p.open_time DESC
    LIMIT ?
"""


def shadow_event_deviation_report(symbol: str, duration: str, *, limit: int = 500) -> dict[str, Any]:
    from app.db.session import get_conn
    from app.services.event_ai_history import settled_expected_profit_usdt

    sym = symbol.strip().upper()
    conn = get_conn()
    try:
        rows = _load_paired_rows(conn, sym, duration, int(limit))
    finally:
        conn.close()

    pairs = [_pair_row(dict(row), settled_expected_profit_usdt) for row in rows]
    summary = _summary(pairs)
    by_strategy = _by_strategy(pairs)
    return {
        "symbol": sym,
        "duration": duration,
        "updatedAt": _utc_now(),
        "summary": summary,
        "byStrategy": by_strategy,
        "pairs": pairs[:50],
    }


def _load_paired_rows(conn: Any, sym: str, duration: str, limit: int) -> list[Any]:
    if not _market_regime_gate_column_available(conn):
        return []
    exact_rows = conn.execute(
        EXACT_PAIR_SELECT.format(order_join=ORDER_JOIN),
        (sym, duration, limit),
    ).fetchall()
    if len(exact_rows) >= limit:
        return list(exact_rows)

    seen_event_ids = {int(row["event_id"]) for row in exact_rows}
    fuzzy_rows = _load_fuzzy_pairs(
        conn,
        sym,
        duration,
        limit=limit - len(exact_rows),
        exclude_event_ids=seen_event_ids,
    )
    combined = list(exact_rows) + fuzzy_rows
    combined.sort(key=lambda row: int(row["open_time"]), reverse=True)
    return combined[:limit]


def _load_fuzzy_pairs(
    conn: Any,
    sym: str,
    duration: str,
    *,
    limit: int,
    exclude_event_ids: set[int],
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    event_rows = conn.execute(
        f"""
        SELECT
          e.id AS event_id,
          e.strategy_key,
          e.start_time,
          e.ai_predicted_direction,
          e.ai_prediction_correct,
          e.result,
          o.side AS order_side,
          o.qty AS order_qty,
          o.price AS order_price
        FROM events e
        {ORDER_JOIN}
        WHERE e.symbol = ?
          AND e.event_interval = ?
          AND e.status = 'SETTLED'
          AND e.prediction_open_time IS NULL
          AND e.market_regime_gate_passed = 1
        ORDER BY e.start_time DESC
        LIMIT ?
        """,
        (sym, duration, max(limit * 4, 80)),
    ).fetchall()
    if not event_rows:
        return []

    strategy_keys = sorted({str(row["strategy_key"]) for row in event_rows})
    placeholders = ", ".join("?" for _ in strategy_keys)
    prediction_rows = conn.execute(
        f"""
        SELECT
          id AS prediction_id,
          signal_key,
          strategy_key,
          open_time,
          direction,
          prediction_correct,
          actual_return AS shadow_return
        FROM predictions
        WHERE symbol = ?
          AND duration = ?
          AND settled_at IS NOT NULL
          AND strategy_key IN ({placeholders})
        ORDER BY open_time DESC
        LIMIT ?
        """,
        (sym, duration, *strategy_keys, max(limit * 20, 400)),
    ).fetchall()
    predictions_by_strategy: dict[str, list[dict[str, Any]]] = {}
    for row in prediction_rows:
        item = dict(row)
        predictions_by_strategy.setdefault(str(item["strategy_key"]), []).append(item)

    pairs: list[dict[str, Any]] = []
    used_prediction_ids: set[int] = set()
    for event in event_rows:
        event_id = int(event["event_id"])
        if event_id in exclude_event_ids:
            continue
        event_start_ms = _event_start_ms(str(event["start_time"]))
        if event_start_ms is None:
            continue
        strategy_key = str(event["strategy_key"])
        direction = str(event["ai_predicted_direction"] or "")
        matched = _match_fuzzy_prediction(
            predictions_by_strategy.get(strategy_key, []),
            direction=direction,
            event_start_ms=event_start_ms,
            used_prediction_ids=used_prediction_ids,
        )
        if matched is None:
            continue
        used_prediction_ids.add(int(matched["prediction_id"]))
        pairs.append(
            {
                **matched,
                "event_id": event_id,
                "ai_prediction_correct": event["ai_prediction_correct"],
                "result": event["result"],
                "order_side": event["order_side"],
                "order_qty": event["order_qty"],
                "order_price": event["order_price"],
            }
        )
        if len(pairs) >= limit:
            break
    return pairs


def _match_fuzzy_prediction(
    predictions: list[dict[str, Any]],
    *,
    direction: str,
    event_start_ms: int,
    used_prediction_ids: set[int],
) -> dict[str, Any] | None:
    for prediction in predictions:
        prediction_id = int(prediction["prediction_id"])
        if prediction_id in used_prediction_ids:
            continue
        if direction and str(prediction["direction"]) != direction:
            continue
        open_time = int(prediction["open_time"])
        if abs(event_start_ms - open_time) > FUZZY_WINDOW_MS:
            continue
        return prediction
    return None


def _event_start_ms(start_time: str) -> int | None:
    try:
        dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except (TypeError, ValueError):
        return None


def _market_regime_gate_column_available(conn: Any) -> bool:
    columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(events)").fetchall()}
    return "market_regime_gate_passed" in columns


def _pair_row(row: dict[str, Any], pnl_fn) -> dict[str, Any]:
    event_pnl = pnl_fn(
        status="SETTLED",
        order_side=row.get("order_side"),
        order_qty=row.get("order_qty"),
        order_price=row.get("order_price"),
        result=row.get("result"),
    )
    shadow_correct = bool(int(row.get("prediction_correct") or 0))
    event_profitable = event_pnl is not None and float(event_pnl) > 0
    divergence = _divergence_type(shadow_correct, event_profitable)
    return {
        "predictionId": row.get("prediction_id"),
        "eventId": row.get("event_id"),
        "strategyKey": row.get("strategy_key"),
        "signalKey": row.get("signal_key"),
        "openTime": row.get("open_time"),
        "direction": row.get("direction"),
        "shadowCorrect": shadow_correct,
        "shadowReturn": row.get("shadow_return"),
        "eventPnlU": event_pnl,
        "eventProfitable": event_profitable,
        "divergenceType": divergence,
    }


def _divergence_type(shadow_correct: bool, event_profitable: bool) -> str:
    if shadow_correct and event_profitable:
        return "aligned_win"
    if shadow_correct and not event_profitable:
        return "shadow_win_event_loss"
    if not shadow_correct and event_profitable:
        return "shadow_loss_event_win"
    return "aligned_loss"
