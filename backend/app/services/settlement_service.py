from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.db.session import get_conn
from app.services.binance_service import fetch_premium_index
from app.services.live_order_settings import FIXED_PAYOUT_RATIO

LIVE_SETTLEMENT_SOURCE = "premiumIndex.rest.current"


@dataclass(frozen=True)
class SettlementQuote:
    price: float
    quote_time_ms: int
    source: str


def settle_event(event_id: int) -> dict:
    conn = get_conn()
    try:
        event = _load_event(conn, event_id)
        if event["status"] == "SETTLED":
            return {"eventId": event_id, "message": "already settled"}

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        _assert_event_due(event, now_ms)
        quote = _fetch_settlement_quote(event)
        result = evaluate_event_result(event, quote.price)
        _settle_orders(conn, event_id, result)
        ai_correct = _prediction_correct(event, result)
        conn.execute(
            """
            UPDATE events
            SET status = 'SETTLED',
                result = ?,
                settlement_price = ?,
                settlement_quote_time = ?,
                settlement_source = ?,
                ai_prediction_correct = ?
            WHERE id = ?
            """,
            (result, quote.price, quote.quote_time_ms, quote.source, ai_correct, event_id),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "eventId": event_id,
        "result": result,
        "closePrice": quote.price,
        "settlementQuoteTime": quote.quote_time_ms,
        "settlementSource": quote.source,
    }


def _load_event(conn, event_id: int):
    event = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    if not event:
        raise ValueError("event not found")
    return event


def _assert_event_due(event, now_ms: int) -> None:
    end_time_ms = parse_event_end_time_ms(event["end_time"])
    if now_ms < end_time_ms:
        raise ValueError("event has not reached endTime")


def _fetch_settlement_quote(event) -> SettlementQuote:
    end_time_ms = parse_event_end_time_ms(event["end_time"])
    return _quote_from_premium_index(fetch_premium_index(event["symbol"]), end_time_ms)


def _quote_from_premium_index(row: dict, end_time_ms: int) -> SettlementQuote:
    price = float(row.get("indexPrice") or 0)
    quote_time = int(row.get("time") or 0)
    if price <= 0 or quote_time <= 0:
        raise ValueError("premium index settlement response is invalid")
    drift_ms = abs(quote_time - int(end_time_ms))
    return SettlementQuote(price, quote_time, f"{LIVE_SETTLEMENT_SOURCE};driftMs={drift_ms}")


def _settle_orders(conn, event_id: int, result: str) -> None:
    orders = conn.execute("SELECT * FROM orders WHERE event_id = ?", (event_id,)).fetchall()
    settled_at = datetime.now(timezone.utc).isoformat()
    for order in orders:
        pnl = _event_contract_pnl(order, result)
        conn.execute(
            "INSERT INTO settlements(event_id, order_id, pnl, settled_at) VALUES(?, ?, ?, ?)",
            (event_id, order["id"], pnl, settled_at),
        )


def _prediction_correct(event, result: str) -> int | None:
    pred_dir = (event["ai_predicted_direction"] or "").lower() if event["ai_predicted_direction"] else None
    if pred_dir not in {"up", "down"}:
        return None
    hit = (pred_dir == "up" and result == "YES") or (pred_dir == "down" and result == "NO")
    return 1 if hit else 0


def get_due_open_event_ids() -> list[int]:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    conn = get_conn()
    rows = conn.execute("SELECT id, end_time FROM events WHERE status = 'OPEN'").fetchall()
    conn.close()
    due_ids: list[int] = []
    for row in rows:
        try:
            if parse_event_end_time_ms(row["end_time"]) <= now_ms:
                due_ids.append(int(row["id"]))
        except Exception:
            # Skip malformed end_time rows to keep loop resilient.
            continue
    return due_ids


def evaluate_event_result(event: dict, close_price: float) -> str:
    rule_type = (event["rule_type"] or "ABOVE").upper()
    strike_value = float(event["strike_value"])

    if rule_type == "ABOVE":
        return "YES" if close_price > strike_value else "NO"
    if rule_type == "BELOW":
        return "YES" if close_price < strike_value else "NO"
    if rule_type == "RANGE":
        upper_bound = event["upper_bound"]
        if upper_bound is None:
            return "NO"
        low = min(strike_value, float(upper_bound))
        high = max(strike_value, float(upper_bound))
        return "YES" if low <= close_price <= high else "NO"
    return "NO"


def _event_contract_pnl(order, result: str) -> float:
    side = order["side"]
    qty = float(order["qty"])
    price = float(order["price"] or FIXED_PAYOUT_RATIO)
    correct = (side == "BUY" and result == "YES") or (side == "SELL" and result == "NO")
    return qty * price if correct else -qty


def parse_event_end_time_ms(end_time_str: str) -> int:
    normalized = end_time_str.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)
