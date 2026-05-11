from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.event_ai_validation import validate_ai_trade_probability
from app.api.event_quick_trade import (
    QuickTradeContext,
    create_quick_trade_record,
    quick_trade_strategy_key,
)
from app.api.event_response import event_response
from app.db.session import get_conn
from app.services.binance_service import fetch_premium_index
from app.services.blind_reverse_martingale_strategy import (
    blind_rm_order_qty_usdt,
    load_blind_rm_settlement_state,
)
from app.services.settlement_service import settle_event
from app.services.strategy_registry import (
    BLIND_REVERSE_MARTINGALE_STRATEGY_KEY,
    MANUAL_STRATEGY_KEY,
    strategy_definition,
)

# 与 auto_trade 一致：仅「随意首单·反向倍投」用面板数量为「基础档」，实际名义按连亏套用 10→20→45（或大于基础的第一档）。
_QUICK_TRADE_MARTINGALE_QTY_KEYS: frozenset[str] = frozenset(
    {BLIND_REVERSE_MARTINGALE_STRATEGY_KEY}
)

router = APIRouter(prefix="/api/events", tags=["events"])

class EventCreate(BaseModel):
    strategyKey: str | None = None
    symbol: str = Field(min_length=6)
    title: str
    eventInterval: str = "30m"
    ruleType: str = "ABOVE"
    strikeValue: float
    upperBound: float | None = None
    endTime: str
    aiProbabilityUp: float | None = None
    aiPredictedDirection: str | None = None
    aiQualityScore: float | None = None
    aiQualityPassed: bool | None = None
    aiHighWinrateGate: str | None = None
    aiHighWinratePassed: bool | None = None
    aiHighWinrateValue: float | None = None

class OrderCreate(BaseModel):
    side: str
    qty: float = Field(gt=0)
    price: float = Field(ge=0)

class QuickTradeCreate(BaseModel):
    event: EventCreate
    order: OrderCreate
    liveTradingEnabled: bool = False
def _fetch_latest_entry_price(symbol: str) -> float:
    row = fetch_premium_index(symbol)
    p = float(row.get("indexPrice") or 0)
    if p <= 0:
        raise ValueError("latest index price unavailable")
    return p

def _entry_price_from_payload(payload: EventCreate) -> float:
    p = float(payload.strikeValue)
    if p <= 0:
        raise HTTPException(status_code=400, detail="strikeValue must be > 0 for quick trade")
    return p

def _validate_event_payload(payload: EventCreate) -> str:
    event_interval = payload.eventInterval
    if event_interval not in {"10m", "30m", "60m", "1d"}:
        raise HTTPException(status_code=400, detail="eventInterval must be one of 10m, 30m, 60m, 1d")

    rule_type = payload.ruleType.upper()
    if rule_type not in {"ABOVE", "BELOW", "RANGE"}:
        raise HTTPException(status_code=400, detail="ruleType must be ABOVE, BELOW, RANGE")
    if rule_type == "RANGE" and payload.upperBound is None:
        raise HTTPException(status_code=400, detail="upperBound is required for RANGE")
    return rule_type

def _normalize_predicted_direction(predicted: str | None) -> str | None:
    if predicted is None:
        return None
    normalized = predicted.lower()
    if normalized not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="aiPredictedDirection must be up or down")
    return normalized

@router.get("")
def list_events() -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute("SELECT * FROM events ORDER BY id DESC").fetchall()
        return [event_response(conn, row) for row in rows]
    finally:
        conn.close()

@router.delete("")
def delete_events(strategyKey: str | None = Query(None)) -> dict:
    """删除事件；不传 strategyKey 时清空全部；传入时仅删除该 strategy_key 对应事件及关联订单、结算。"""
    conn = get_conn()
    try:
        if strategyKey is None or not strategyKey.strip():
            conn.execute("DELETE FROM settlements")
            conn.execute("DELETE FROM orders")
            conn.execute("DELETE FROM events")
            conn.commit()
            return {"ok": True}
        key = strategyKey.strip()
        conn.execute(
            "DELETE FROM settlements WHERE event_id IN (SELECT id FROM events WHERE strategy_key = ?)",
            (key,),
        )
        conn.execute(
            "DELETE FROM orders WHERE event_id IN (SELECT id FROM events WHERE strategy_key = ?)",
            (key,),
        )
        cursor = conn.execute("DELETE FROM events WHERE strategy_key = ?", (key,))
        deleted = int(cursor.rowcount or 0)
        conn.commit()
        return {"ok": True, "deleted": deleted, "strategyKey": key}
    finally:
        conn.close()

@router.post("")
def create_event(payload: EventCreate) -> dict:
    event_interval = payload.eventInterval
    rule_type = _validate_event_payload(payload)
    predicted = _normalize_predicted_direction(payload.aiPredictedDirection)
    strategy_key = _event_strategy_key(payload.strategyKey)
    validate_ai_trade_probability(
        payload.aiProbabilityUp,
        event_interval,
        payload.aiQualityScore,
        payload.aiQualityPassed,
        payload.aiHighWinratePassed,
        strategy_key=strategy_key,
    )
    conn = get_conn()
    start_time = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO events(
          strategy_key, symbol, title, event_interval, rule_type, strike_value, upper_bound,
          start_time, end_time, status,
          ai_probability_up, ai_predicted_direction, ai_quality_score, ai_quality_passed,
          ai_high_winrate_gate, ai_high_winrate_passed, ai_high_winrate_value
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            strategy_key,
            payload.symbol.upper(),
            payload.title,
            event_interval,
            rule_type,
            payload.strikeValue,
            payload.upperBound,
            start_time,
            payload.endTime,
            payload.aiProbabilityUp,
            predicted,
            payload.aiQualityScore,
            int(bool(payload.aiQualityPassed)) if payload.aiQualityPassed is not None else None,
            payload.aiHighWinrateGate,
            int(bool(payload.aiHighWinratePassed)) if payload.aiHighWinratePassed is not None else None,
            payload.aiHighWinrateValue,
        ),
    )
    conn.commit()
    event_id = cursor.lastrowid
    conn.close()
    return {"id": event_id}

@router.post("/quick-trade")
def create_quick_trade(payload: QuickTradeCreate) -> dict:
    side = payload.order.side.upper()
    if side not in {"BUY", "SELL"}:
        raise HTTPException(status_code=400, detail="side must be BUY or SELL")
    event_interval = payload.event.eventInterval
    rule_type = _validate_event_payload(payload.event)
    predicted = _normalize_predicted_direction(payload.event.aiPredictedDirection)
    strategy_key = quick_trade_strategy_key(payload)
    validate_ai_trade_probability(
        payload.event.aiProbabilityUp,
        event_interval,
        payload.event.aiQualityScore,
        payload.event.aiQualityPassed,
        payload.event.aiHighWinratePassed,
        strategy_key=strategy_key,
    )
    symbol = payload.event.symbol.upper()
    entry_price = _entry_price_from_payload(payload.event)
    payload = _adjust_quick_trade_qty_for_martingale(payload, strategy_key, symbol)
    return create_quick_trade_record(
        QuickTradeContext(
            payload=payload,
            strategy_key=strategy_key,
            symbol=symbol,
            side=side,
            event_interval=event_interval,
            rule_type=rule_type,
            predicted=predicted,
            entry_price=entry_price,
            live_trading_enabled=payload.liveTradingEnabled,
        )
    )


def _adjust_quick_trade_qty_for_martingale(
    payload: QuickTradeCreate, strategy_key: str, symbol: str
) -> QuickTradeCreate:
    if strategy_key not in _QUICK_TRADE_MARTINGALE_QTY_KEYS:
        return payload
    base = float(payload.order.qty)
    state = load_blind_rm_settlement_state(strategy_key, symbol.upper())
    qty = blind_rm_order_qty_usdt(base, state)
    if abs(qty - base) < 1e-9:
        return payload
    order = payload.order
    new_order = (
        order.model_copy(update={"qty": qty})
        if hasattr(order, "model_copy")
        else order.copy(update={"qty": qty})
    )
    return (
        payload.model_copy(update={"order": new_order})
        if hasattr(payload, "model_copy")
        else payload.copy(update={"order": new_order})
    )


def _event_strategy_key(value: str | None) -> str:
    key = value or MANUAL_STRATEGY_KEY
    if key != MANUAL_STRATEGY_KEY:
        strategy_definition(key)
    return key


@router.post("/{event_id}/orders")
def create_order(event_id: int, payload: OrderCreate) -> dict:
    side = payload.side.upper()
    if side not in {"BUY", "SELL"}:
        raise HTTPException(status_code=400, detail="side must be BUY or SELL")

    conn = get_conn()
    try:
        event = _load_event_for_order(conn, event_id)
        _validate_event_ai(event)
        entry_price = _entry_price_for_event(event)
        _update_event_strike(conn, event_id, entry_price)
        order_id = _insert_order(conn, event_id, side, payload)
        conn.commit()
        return {"id": order_id}
    finally:
        conn.close()


def _load_event_for_order(conn, event_id: int):
    event = conn.execute(
        """
        SELECT id, symbol, event_interval, strategy_key, ai_probability_up, ai_quality_score, ai_quality_passed,
          ai_high_winrate_passed
        FROM events WHERE id = ?
        """,
        (event_id,),
    ).fetchone()
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    return event


def _validate_event_ai(event) -> None:
    validate_ai_trade_probability(
        event["ai_probability_up"],
        event["event_interval"],
        event["ai_quality_score"],
        event["ai_quality_passed"],
        event["ai_high_winrate_passed"],
        strategy_key=event["strategy_key"],
    )


def _entry_price_for_event(event) -> float:
    try:
        return _fetch_latest_entry_price(event["symbol"])
    except Exception as exc:
        detail = f"failed to fetch latest entry price: {exc}"
        raise HTTPException(status_code=502, detail=detail) from exc


def _update_event_strike(conn, event_id: int, entry_price: float) -> None:
    conn.execute(
        "UPDATE events SET strike_value = ? WHERE id = ?",
        (entry_price, event_id),
    )


def _insert_order(conn, event_id: int, side: str, payload: OrderCreate) -> int:
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO orders(event_id, side, price, qty, status, created_at)
        VALUES(?, ?, ?, ?, 'OPEN', ?)
        """,
        (event_id, side, payload.price, payload.qty, now),
    )
    return int(cursor.lastrowid)


@router.post("/{event_id}/settle")
def settle(event_id: int) -> dict:
    try:
        return settle_event(event_id)
    except ValueError as exc:
        detail = str(exc)
        if detail == "event not found":
            raise HTTPException(status_code=404, detail=detail) from exc
        raise HTTPException(status_code=400, detail=detail) from exc
