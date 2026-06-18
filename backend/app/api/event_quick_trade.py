from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from app.db.session import get_conn
from app.services.binance_event_contract import (
    build_event_contract_order_payload,
    place_event_contract_order,
)
from app.services.live_order_settings import payout_ratio_for_duration
from app.services.live_order_failure_log import log_live_order_failure
from app.services.market_regime_trade_gate import (
    MARKET_REGIME_TRADE_GATE_VERSION,
    MarketRegimeTradeDecision,
    evaluate_market_regime_trade_gate,
)
from app.services.position_guard import has_open_position
from app.services.event_search_index import refresh_event_search_row
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
    prediction_open_time: int | None = None


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
        if has_open_position(
            conn,
            ctx.symbol,
            ctx.strategy_key,
            event_interval=ctx.event_interval,
            require_market_regime_gate_passed=_requires_market_regime_position_guard(ctx),
        ):
            raise HTTPException(status_code=409, detail="已有进行中持仓，需等待上一笔结束后再下单")
        regime_decision = _enforce_market_regime_trade_gate(ctx)
        external_order = _place_external_order(ctx)
        event_id = _insert_event(conn=conn, ctx=ctx, now=now, regime_decision=regime_decision)
        order_id = _insert_order(
            conn=conn,
            order=OrderInsertContext(ctx, event_id, now, external_order),
        )
        refresh_event_search_row(conn, event_id)
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    conn.close()
    return _response(QuickTradeResult(event_id, order_id, ctx.entry_price, external_order))


def _enforce_market_regime_trade_gate(ctx: QuickTradeContext) -> MarketRegimeTradeDecision | None:
    if ctx.strategy_key == MANUAL_STRATEGY_KEY or ctx.predicted is None:
        return None
    if ctx.prediction_open_time is None:
        raise HTTPException(status_code=400, detail="prediction_open_time is required for market regime trade gate")
    decision = evaluate_market_regime_trade_gate(
        symbol=ctx.symbol,
        duration=ctx.event_interval,
        open_time=ctx.prediction_open_time,
        direction=ctx.predicted,
    )
    if decision.allowed:
        return decision
    raise HTTPException(
        status_code=409,
        detail={
            "reason": "market_regime_trade_gate_blocked",
            "gateReason": decision.reason,
            "mode": decision.mode,
            "regime": decision.regime,
        },
    )


def _requires_market_regime_position_guard(ctx: QuickTradeContext) -> bool:
    return ctx.strategy_key != MANUAL_STRATEGY_KEY and ctx.predicted is not None


def _place_external_order(ctx: QuickTradeContext) -> dict[str, Any]:
    payout_ratio = payout_ratio_for_duration(ctx.event_interval)
    if not ctx.live_trading_enabled:
        return _simulated_external_order(ctx, payout_ratio)
    try:
        return place_event_contract_order(
            symbol=ctx.symbol,
            event_interval=ctx.event_interval,
            side=ctx.side,
            amount=ctx.payload.order.qty,
            payout_ratio=payout_ratio,
        )
    except Exception as exc:
        log_live_order_failure(ctx, exc)
        raise


def _simulated_external_order(ctx: QuickTradeContext, payout_ratio: float) -> dict[str, Any]:
    request = build_event_contract_order_payload(
        symbol=ctx.symbol,
        event_interval=ctx.event_interval,
        side=ctx.side,
        amount=ctx.payload.order.qty,
        payout_ratio=payout_ratio,
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


def _insert_event(
    *,
    conn: Any,
    ctx: QuickTradeContext,
    now: str,
    regime_decision: MarketRegimeTradeDecision | None,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO events(
          strategy_key, symbol, title, event_interval, rule_type, strike_value, upper_bound,
          start_time, end_time, status, prediction_open_time, prediction_id,
          ai_probability_up, ai_predicted_direction, ai_quality_score, ai_quality_passed,
          ai_high_winrate_gate, ai_high_winrate_rule, ai_high_winrate_passed, ai_high_winrate_value,
          market_regime_gate_version, market_regime_gate_passed, market_regime_gate_reason,
          market_regime_gate_mode, market_regime_label
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ctx.prediction_open_time,
            ctx.payload.event.predictionId,
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
            *_market_regime_event_values(regime_decision),
        ),
    )
    return int(cursor.lastrowid)


def _market_regime_event_values(decision: MarketRegimeTradeDecision | None) -> tuple:
    if decision is None:
        return (None, None, None, None, None)
    return (
        MARKET_REGIME_TRADE_GATE_VERSION,
        int(decision.allowed),
        decision.reason,
        decision.mode,
        decision.regime.get("regimeLabel"),
    )


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
    payout_ratio = payout_ratio_for_duration(order.trade.event_interval)
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
            payout_ratio,
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
