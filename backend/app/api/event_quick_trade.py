from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from app.db.session import get_conn
from app.services.binance_event_contract import (
    BinanceEventConfigError,
    BinanceEventOrderError,
    build_event_contract_order_payload,
    place_event_contract_order,
)
from app.services.live_order_failure_log import log_live_order_failure
from app.services.live_order_settings import FIXED_PAYOUT_RATIO
from app.services.position_guard import has_open_position
from app.services.strategy_registry import MANUAL_STRATEGY_KEY, strategy_definition


@dataclass(frozen=True)
class QuickTradeContext:
    payload: Any
    strategy_key: str
    symbol: str
    side: str
    event_interval: str
    rule_type: str
    predicted: str | None
    entry_price: float
    live_trading_enabled: bool


@dataclass(frozen=True)
class OrderInsertContext:
    trade: QuickTradeContext
    event_id: int
    now: str
    external_order: dict[str, Any]


@dataclass(frozen=True)
class QuickTradeResult:
    event_id: int
    order_id: int
    entry_price: float
    external_order: dict[str, Any]


def create_quick_trade_record(ctx: QuickTradeContext) -> dict:
    conn = get_conn()
    now = datetime.now(timezone.utc).isoformat()
    try:
        if has_open_position(conn, ctx.symbol, ctx.strategy_key, event_interval=ctx.event_interval):
            raise HTTPException(status_code=409, detail="已有进行中持仓，需等待上一笔结束后再下单")
        external_order = _place_external_order(ctx)
        event_id = _insert_event(conn=conn, ctx=ctx, now=now)
        order_id = _insert_order(
            conn=conn,
            order=OrderInsertContext(ctx, event_id, now, external_order),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    conn.close()
    return _response(QuickTradeResult(event_id, order_id, ctx.entry_price, external_order))


def _place_external_order(ctx: QuickTradeContext) -> dict[str, Any]:
    if not ctx.live_trading_enabled:
        return _simulated_external_order(ctx)
    try:
        return place_event_contract_order(
            symbol=ctx.symbol,
            event_interval=ctx.event_interval,
            side=ctx.side,
            amount=ctx.payload.order.qty,
            payout_ratio=FIXED_PAYOUT_RATIO,
        )
    except BinanceEventConfigError as exc:
        log_live_order_failure(ctx, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BinanceEventOrderError as exc:
        log_live_order_failure(ctx, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        log_live_order_failure(ctx, exc)
        raise


def _simulated_external_order(ctx: QuickTradeContext) -> dict[str, Any]:
    request = build_event_contract_order_payload(
        symbol=ctx.symbol,
        event_interval=ctx.event_interval,
        side=ctx.side,
        amount=ctx.payload.order.qty,
        payout_ratio=FIXED_PAYOUT_RATIO,
    )
    return {
        "request": request,
        "response": {
            "simulation": True,
            "message": "模拟交易已启用；未调用 Binance 下单接口",
        },
        "externalOrderId": None,
        "externalStatus": "SIMULATED",
        "simulationNotice": "模拟交易：未调用 Binance",
        "simulated": True,
    }


def _insert_event(*, conn: Any, ctx: QuickTradeContext, now: str) -> int:
    cursor = conn.execute(
        """
        INSERT INTO events(
          strategy_key, symbol, title, event_interval, rule_type, strike_value, upper_bound,
          start_time, end_time, status,
          ai_probability_up, ai_predicted_direction, ai_quality_score, ai_quality_passed,
          ai_high_winrate_gate, ai_high_winrate_rule, ai_high_winrate_passed, ai_high_winrate_value
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ctx.strategy_key,
            ctx.symbol,
            ctx.payload.event.title,
            ctx.event_interval,
            ctx.rule_type,
            ctx.entry_price,
            ctx.payload.event.upperBound,
            now,
            ctx.payload.event.endTime,
            ctx.payload.event.aiProbabilityUp,
            ctx.predicted,
            ctx.payload.event.aiQualityScore,
            int(bool(ctx.payload.event.aiQualityPassed)) if ctx.payload.event.aiQualityPassed is not None else None,
            ctx.payload.event.aiHighWinrateGate,
            ctx.payload.event.aiHighWinrateRule,
            (
                int(bool(ctx.payload.event.aiHighWinratePassed))
                if ctx.payload.event.aiHighWinratePassed is not None
                else None
            ),
            ctx.payload.event.aiHighWinrateValue,
        ),
    )
    return int(cursor.lastrowid)


def quick_trade_strategy_key(payload: Any) -> str:
    key = getattr(payload.event, "strategyKey", None) or MANUAL_STRATEGY_KEY
    if key != MANUAL_STRATEGY_KEY:
        strategy_definition(key)
    return key


def _insert_order(
    *,
    conn: Any,
    order: OrderInsertContext,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO orders(
          event_id, side, price, qty, status, created_at,
          external_order_id, external_status, external_response
        )
        VALUES(?, ?, ?, ?, 'OPEN', ?, ?, ?, ?)
        """,
        (
            order.event_id,
            order.trade.side,
            FIXED_PAYOUT_RATIO,
            order.trade.payload.order.qty,
            order.now,
            order.external_order.get("externalOrderId"),
            order.external_order.get("externalStatus"),
            json.dumps(order.external_order, ensure_ascii=False),
        ),
    )
    return int(cursor.lastrowid)


def _response(result: QuickTradeResult) -> dict:
    return {
        "eventId": result.event_id,
        "orderId": result.order_id,
        "strikeValue": result.entry_price,
        "externalOrderId": result.external_order.get("externalOrderId"),
        "externalStatus": result.external_order.get("externalStatus"),
        "binance": result.external_order.get("response"),
        "binanceCalled": result.external_order.get("simulated") is not True,
        "simulationNotice": result.external_order.get("simulationNotice"),
        "simulated": result.external_order.get("simulated") is True,
    }
