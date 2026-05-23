from __future__ import annotations

from datetime import datetime

UNKNOWN_DURATION = -1


def settled_expected_profit_usdt(*, status: str, order_side: str | None, order_qty, order_price, result) -> float | None:
    qty = _finite_float(order_qty)
    price = _finite_float(order_price)
    if qty is None or qty <= 0 or price is None or price < 0:
        return None
    if status != "SETTLED" or result is None or order_side is None:
        return None
    is_correct = (order_side == "BUY" and result == "YES") or (order_side == "SELL" and result == "NO")
    return qty * price if is_correct else -qty


def ai_history_success(conn, symbol: str) -> dict:
    rows = conn.execute(
        """
        SELECT
          e.strategy_key,
          e.start_time,
          e.end_time,
          e.ai_prediction_correct,
          e.status,
          e.result,
          o.side AS order_side,
          o.qty AS order_qty,
          o.price AS order_price
        FROM events e
        LEFT JOIN orders o ON o.id = (
            SELECT id FROM orders WHERE event_id = e.id ORDER BY id DESC LIMIT 1
        )
        WHERE e.symbol = ?
          AND e.status = 'SETTLED'
          AND e.ai_predicted_direction IS NOT NULL
          AND e.ai_prediction_correct IS NOT NULL
        """,
        (symbol.upper(),),
    ).fetchall()

    overall_total = 0
    overall_hits = 0
    overall_pnl = 0.0
    buckets: dict[tuple[str, int], dict] = {}

    for row in rows:
        overall_total += 1
        if int(row["ai_prediction_correct"] or 0) == 1:
            overall_hits += 1
        pnl = settled_expected_profit_usdt(
            status=row["status"],
            order_side=row["order_side"],
            order_qty=row["order_qty"],
            order_price=row["order_price"],
            result=row["result"],
        )
        if pnl is not None:
            overall_pnl += pnl

        strategy_key = row["strategy_key"] or "manual"
        duration_minutes = _duration_minutes(row["start_time"], row["end_time"])
        bucket_key = (strategy_key, duration_minutes)
        bucket = buckets.setdefault(bucket_key, {"total": 0, "hits": 0, "pnlU": 0.0})
        bucket["total"] += 1
        if int(row["ai_prediction_correct"] or 0) == 1:
            bucket["hits"] += 1
        if pnl is not None:
            bucket["pnlU"] += pnl

    by_strategy = [
        {
            "strategyKey": strategy_key,
            "durationMinutes": duration_minutes,
            "total": bucket["total"],
            "hits": bucket["hits"],
            "pnlU": bucket["pnlU"],
            "rate": bucket["hits"] / bucket["total"] if bucket["total"] else None,
        }
        for (strategy_key, duration_minutes), bucket in buckets.items()
    ]
    by_strategy.sort(
        key=lambda item: (
            item["durationMinutes"] if item["durationMinutes"] != UNKNOWN_DURATION else 1_000_000_000,
            -item["pnlU"],
            item["strategyKey"],
        )
    )

    return {
        "overall": {
            "total": overall_total,
            "hits": overall_hits,
            "rate": overall_hits / overall_total if overall_total else None,
            "pnlU": overall_pnl,
        },
        "byStrategy": by_strategy,
    }


def _duration_minutes(start_time: str | None, end_time: str | None) -> int:
    start = _parse_iso_ms(start_time)
    end = _parse_iso_ms(end_time)
    if start is None or end is None or end <= start:
        return UNKNOWN_DURATION
    return round((end - start) / 60000)


def _parse_iso_ms(value: str | None) -> float | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp() * 1000
    except ValueError:
        return None


def _finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number
