from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api.event_ai_validation import validate_ai_trade_probability
from app.api.event_quick_trade import (
    QuickTradeContext,
    create_quick_trade_record,
    quick_trade_strategy_key,
)
from app.api.event_response import event_response
from app.api.event_writes import delete_event_records, insert_event_record, insert_order_record
from app.api.events_models import EventCreate, OrderCreate, QuickTradeCreate
from app.db.session import get_conn
from app.services.event_list_query import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    paginated_events,
)
from app.services.settlement_service import settle_event
from app.services.strategy_registry import (
    MANUAL_STRATEGY_KEY,
    strategy_definition,
)

router = APIRouter(prefix="/api/events", tags=["events"])

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
def list_events(
    symbol: str | None = Query(None, min_length=6),
    strategyKey: str | None = Query(None, min_length=1),
    durationMinutes: int | None = Query(None),
    page: int = Query(1, ge=1),
    pageSize: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    view: str = Query("events"),
    q: str | None = Query(None, description="search event id/title/status/strategy/order response"),
) -> dict:
    conn = get_conn()
    try:
        selected_symbol = symbol if isinstance(symbol, str) else None
        selected_strategy = strategyKey if isinstance(strategyKey, str) else None
        selected_duration = durationMinutes if isinstance(durationMinutes, int) else None
        selected_view = view if isinstance(view, str) else "events"
        query = q if isinstance(q, str) else None
        try:
            payload = paginated_events(
                conn,
                symbol=selected_symbol,
                page=page,
                page_size=pageSize,
                view=selected_view,
                strategy_key=selected_strategy,
                duration_minutes=selected_duration,
                query=query,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            **payload,
            "items": [event_response(conn, row) for row in payload["items"]],
        }
    finally:
        conn.close()

@router.delete("")
def delete_events(strategyKey: str | None = Query(None)) -> dict:
    """删除事件；不传 strategyKey 时清空全部；传入时仅删除该 strategy_key 对应事件及关联订单、结算。"""
    return delete_event_records(strategyKey)

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
    return insert_event_record(payload, rule_type, predicted, strategy_key)

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

    return insert_order_record(event_id, side, payload)


@router.post("/{event_id}/settle")
def settle(event_id: int) -> dict:
    try:
        return settle_event(event_id)
    except ValueError as exc:
        detail = str(exc)
        if detail == "event not found":
            raise HTTPException(status_code=404, detail=detail) from exc
        raise HTTPException(status_code=400, detail=detail) from exc
