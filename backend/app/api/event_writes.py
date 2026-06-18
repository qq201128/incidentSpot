from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from app.api.event_ai_validation import validate_ai_trade_probability
from app.api.events_models import EventCreate, OrderCreate
from app.db.session import get_conn
from app.services.binance_service import fetch_premium_index
from app.services.event_search_index import ensure_event_search_index, refresh_event_search_row


def delete_event_records(strategy_key: str | None) -> dict:
    conn = get_conn()
    try:
        if strategy_key is None or not strategy_key.strip():
            conn.execute("DELETE FROM settlements")
            conn.execute("DELETE FROM orders")
            conn.execute("DELETE FROM events")
            ensure_event_search_index(conn)
            conn.commit()
            return {"ok": True}
        deleted = _delete_strategy_events(conn, strategy_key.strip())
        conn.commit()
        return {"ok": True, "deleted": deleted, "strategyKey": strategy_key.strip()}
    finally:
        conn.close()


def insert_event_record(payload: EventCreate, rule_type: str, predicted: str | None, strategy_key: str) -> dict:
    conn = get_conn()
    try:
        event_id = _insert_event(conn, payload, rule_type, predicted, strategy_key)
        refresh_event_search_row(conn, event_id)
        conn.commit()
        return {"id": event_id}
    finally:
        conn.close()


def insert_order_record(event_id: int, side: str, payload: OrderCreate) -> dict:
    conn = get_conn()
    try:
        event = _load_event_for_order(conn, event_id)
        _validate_event_ai(event)
        _update_event_strike(conn, event_id, _entry_price_for_event(event))
        order_id = _insert_order(conn, event_id, side, payload)
        refresh_event_search_row(conn, event_id)
        conn.commit()
        return {"id": order_id}
    finally:
        conn.close()


def _delete_strategy_events(conn: Any, strategy_key: str) -> int:
    conn.execute("DELETE FROM settlements WHERE event_id IN (SELECT id FROM events WHERE strategy_key = ?)", (strategy_key,))
    conn.execute("DELETE FROM orders WHERE event_id IN (SELECT id FROM events WHERE strategy_key = ?)", (strategy_key,))
    cursor = conn.execute("DELETE FROM events WHERE strategy_key = ?", (strategy_key,))
    ensure_event_search_index(conn)
    return int(cursor.rowcount or 0)


def _insert_event(conn: Any, payload: EventCreate, rule_type: str, predicted: str | None, strategy_key: str) -> int:
    cursor = conn.execute(
        """
        INSERT INTO events(
          strategy_key, symbol, title, event_interval, rule_type, strike_value, upper_bound,
          start_time, end_time, status, ai_probability_up, ai_predicted_direction, ai_quality_score,
          ai_quality_passed, ai_high_winrate_gate, ai_high_winrate_rule, ai_high_winrate_passed,
          ai_high_winrate_value, prediction_id
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _event_values(payload, rule_type, predicted, strategy_key),
    )
    return int(cursor.lastrowid)


def _event_values(payload: EventCreate, rule_type: str, predicted: str | None, strategy_key: str) -> tuple:
    return (
        strategy_key, payload.symbol.upper(), payload.title, payload.eventInterval, rule_type,
        payload.strikeValue, payload.upperBound, datetime.now(timezone.utc).isoformat(),
        payload.endTime, payload.aiProbabilityUp, predicted, payload.aiQualityScore,
        int(bool(payload.aiQualityPassed)) if payload.aiQualityPassed is not None else None,
        payload.aiHighWinrateGate, payload.aiHighWinrateRule,
        int(bool(payload.aiHighWinratePassed)) if payload.aiHighWinratePassed is not None else None,
        payload.aiHighWinrateValue, payload.predictionId,
    )


def _load_event_for_order(conn: Any, event_id: int) -> Any:
    event = conn.execute(
        """
        SELECT id, symbol, event_interval, strategy_key, ai_probability_up, ai_quality_score,
               ai_quality_passed, ai_high_winrate_passed
        FROM events WHERE id = ?
        """,
        (event_id,),
    ).fetchone()
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    return event


def _validate_event_ai(event: Any) -> None:
    validate_ai_trade_probability(
        event["ai_probability_up"], event["event_interval"], event["ai_quality_score"],
        event["ai_quality_passed"], event["ai_high_winrate_passed"], strategy_key=event["strategy_key"],
    )


def _entry_price_for_event(event: Any) -> float:
    try:
        row = fetch_premium_index(event["symbol"])
        price = float(row.get("indexPrice") or 0)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"failed to fetch latest entry price: {exc}") from exc
    if price <= 0:
        raise HTTPException(status_code=502, detail="failed to fetch latest entry price: latest index price unavailable")
    return price


def _update_event_strike(conn: Any, event_id: int, entry_price: float) -> None:
    conn.execute("UPDATE events SET strike_value = ? WHERE id = ?", (entry_price, event_id))


def _insert_order(conn: Any, event_id: int, side: str, payload: OrderCreate) -> int:
    cursor = conn.execute(
        """
        INSERT INTO orders(event_id, side, price, qty, status, created_at, external_status, external_response)
        VALUES(?, ?, ?, ?, 'OPEN', ?, 'SIMULATED', ?)
        """,
        (event_id, side, payload.price, payload.qty, datetime.now(timezone.utc).isoformat(), _simulated_order_response()),
    )
    return int(cursor.lastrowid)


def _simulated_order_response() -> str:
    return json.dumps({"simulation": True, "message": "模拟订单：未调用 Binance 下单接口"}, ensure_ascii=False)
