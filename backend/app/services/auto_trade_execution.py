from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.api.event_quick_trade import QuickTradeContext, create_quick_trade_record
from app.services.auto_trade_types import (
    AutoTradeEventPayload,
    AutoTradeOrderPayload,
    AutoTradePayload,
    AutoTradeSettings,
)
from app.services.binance_service import fetch_premium_index
from app.services.live_order_settings import FIXED_PAYOUT_RATIO
from app.db.session import get_conn
from app.services.blind_reverse_martingale_strategy import (
    blind_rm_order_qty_usdt,
    load_blind_rm_settlement_state,
)
from app.services.strategy_registry import (
    BLIND_REVERSE_MARTINGALE_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_15M_MG_51020_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_MG_5102045_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY,
    ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY,
    strategy_definition,
)

MARTINGALE_MAX_USDT = 20.0
MARTINGALE_NOTIONAL_MAX_CONSECUTIVE_LOSSES = 3

ORDERBOOK_NOTIONAL_MG_5102045_AMOUNTS_USDT: tuple[float, ...] = (5.0, 10.0, 20.0, 45.0)
ORDERBOOK_NOTIONAL_MG_5102045_RESET_AFTER_LOSSES = len(ORDERBOOK_NOTIONAL_MG_5102045_AMOUNTS_USDT)

ORDERBOOK_NOTIONAL_15M_MG_51020_AMOUNTS_USDT: tuple[float, ...] = (5.0, 10.0, 20.0)
ORDERBOOK_NOTIONAL_15M_MG_51020_RESET_AFTER_LOSSES = len(ORDERBOOK_NOTIONAL_15M_MG_51020_AMOUNTS_USDT)

PERCENT_SCALE = 1000
PERCENT_DECIMALS = 10


def create_trade_from_prediction(settings: AutoTradeSettings, prediction: dict[str, Any]) -> dict:
    entry_price = _fetch_latest_entry_price(settings.symbol)
    side = "BUY" if prediction["direction"] == "up" else "SELL"
    payload = _build_quick_trade_payload(settings, prediction, entry_price, side=side)
    return create_quick_trade_record(
        QuickTradeContext(
            payload=payload,
            strategy_key=settings.strategy_key,
            symbol=settings.symbol,
            side=side,
            event_interval=settings.duration,
            rule_type="ABOVE",
            predicted=prediction["direction"],
            entry_price=entry_price,
            live_trading_enabled=settings.live_trading_enabled,
        )
    )


def _build_quick_trade_payload(
    settings: AutoTradeSettings,
    prediction: dict[str, Any],
    entry_price: float,
    *,
    side: str,
) -> AutoTradePayload:
    end_time = datetime.now(timezone.utc) + timedelta(minutes=settings.duration_minutes)
    return AutoTradePayload(
        event=AutoTradeEventPayload(
            strategyKey=settings.strategy_key,
            symbol=settings.symbol,
            title=_event_title(settings, prediction, side),
            eventInterval=settings.duration,
            ruleType="ABOVE",
            strikeValue=entry_price,
            upperBound=None,
            endTime=end_time.isoformat(),
            aiProbabilityUp=float(prediction["probability_up"]),
            aiPredictedDirection=prediction["direction"],
            aiQualityScore=prediction.get("trade_quality_score"),
            aiQualityPassed=_as_bool(prediction.get("trade_quality_passed")),
            aiHighWinrateGate=prediction.get("high_winrate_gate"),
            aiHighWinrateRule=prediction.get("high_winrate_rule"),
            aiHighWinratePassed=_as_bool(prediction.get("high_winrate_gate_passed")),
            aiHighWinrateValue=prediction.get("high_winrate_gate_value"),
        ),
        order=AutoTradeOrderPayload(side=side, qty=martingale_order_qty_usdt(settings), price=FIXED_PAYOUT_RATIO),
    )


def martingale_order_qty_usdt(settings: AutoTradeSettings) -> float:
    base = float(settings.qty)
    key = settings.strategy_key
    if key == ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY:
        return _orderbook_notional_mg_qty(settings, base)
    if key == ORDERBOOK_NOTIONAL_MG_5102045_STRATEGY_KEY:
        return _orderbook_notional_mg_5102045_qty(settings)
    if key == ORDERBOOK_NOTIONAL_15M_MG_51020_STRATEGY_KEY:
        return _orderbook_notional_mg_51020_15m_qty(settings)
    if key == BLIND_REVERSE_MARTINGALE_STRATEGY_KEY:
        state = load_blind_rm_settlement_state(key, settings.symbol)
        return blind_rm_order_qty_usdt(base, state)
    if key != ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY:
        return base
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT o.qty AS qty, e.ai_prediction_correct AS correct
            FROM events e
            INNER JOIN orders o ON o.event_id = e.id
            WHERE e.strategy_key = ? AND e.symbol = ? AND e.status = 'SETTLED'
              AND e.ai_prediction_correct IS NOT NULL
            ORDER BY e.end_time DESC, o.id DESC
            LIMIT 1
            """,
            (settings.strategy_key, settings.symbol.upper()),
        ).fetchone()
    finally:
        conn.close()
    if row is None or int(row["correct"] or 0) == 1:
        return base
    prev = float(row["qty"] or 0)
    step = base * 2.0 if prev <= 0 else prev * 2.0
    return min(step, MARTINGALE_MAX_USDT)


def _orderbook_notional_mg_qty(settings: AutoTradeSettings, base: float) -> float:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT o.qty AS qty, e.ai_prediction_correct AS correct
            FROM events e
            INNER JOIN orders o ON o.event_id = e.id
            WHERE e.strategy_key = ? AND e.symbol = ? AND e.status = 'SETTLED'
              AND e.ai_prediction_correct IS NOT NULL
            ORDER BY e.end_time DESC, o.id DESC
            LIMIT 50
            """,
            (settings.strategy_key, settings.symbol.upper()),
        ).fetchall()
    finally:
        conn.close()
    if not rows or int(rows[0]["correct"] or 0) == 1:
        return base
    streak = 0
    for row in rows:
        if int(row["correct"] or 0) == 1:
            break
        streak += 1
    if streak >= MARTINGALE_NOTIONAL_MAX_CONSECUTIVE_LOSSES:
        return base
    prev = float(rows[0]["qty"] or 0)
    step = base * 2.0 if prev <= 0 else prev * 2.0
    return min(step, MARTINGALE_MAX_USDT)


def _orderbook_notional_mg_5102045_qty(settings: AutoTradeSettings) -> float:
    """固定阶梯 5→10→20→45；面板 qty 不参与名义（与策略说明一致）。"""
    ladder = ORDERBOOK_NOTIONAL_MG_5102045_AMOUNTS_USDT
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT o.qty AS qty, e.ai_prediction_correct AS correct
            FROM events e
            INNER JOIN orders o ON o.event_id = e.id
            WHERE e.strategy_key = ? AND e.symbol = ? AND e.status = 'SETTLED'
              AND e.ai_prediction_correct IS NOT NULL
            ORDER BY e.end_time DESC, o.id DESC
            LIMIT 50
            """,
            (settings.strategy_key, settings.symbol.upper()),
        ).fetchall()
    finally:
        conn.close()
    if not rows or int(rows[0]["correct"] or 0) == 1:
        return float(ladder[0])
    streak = 0
    for row in rows:
        if int(row["correct"] or 0) == 1:
            break
        streak += 1
    if streak >= ORDERBOOK_NOTIONAL_MG_5102045_RESET_AFTER_LOSSES:
        return float(ladder[0])
    return float(ladder[streak])


def _orderbook_notional_mg_51020_15m_qty(settings: AutoTradeSettings) -> float:
    """15M 信号专用阶梯 5→10→20；面板 qty 不参与名义。"""
    ladder = ORDERBOOK_NOTIONAL_15M_MG_51020_AMOUNTS_USDT
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT o.qty AS qty, e.ai_prediction_correct AS correct
            FROM events e
            INNER JOIN orders o ON o.event_id = e.id
            WHERE e.strategy_key = ? AND e.symbol = ? AND e.status = 'SETTLED'
              AND e.ai_prediction_correct IS NOT NULL
            ORDER BY e.end_time DESC, o.id DESC
            LIMIT 50
            """,
            (settings.strategy_key, settings.symbol.upper()),
        ).fetchall()
    finally:
        conn.close()
    if not rows or int(rows[0]["correct"] or 0) == 1:
        return float(ladder[0])
    streak = 0
    for row in rows:
        if int(row["correct"] or 0) == 1:
            break
        streak += 1
    if streak >= ORDERBOOK_NOTIONAL_15M_MG_51020_RESET_AFTER_LOSSES:
        return float(ladder[0])
    return float(ladder[streak])


def _event_title(settings: AutoTradeSettings, prediction: dict[str, Any], side: str) -> str:
    probability = _side_probability(float(prediction["probability_up"]), side)
    confidence = round(probability * PERCENT_SCALE) / PERCENT_DECIMALS
    direction = "看涨" if side == "BUY" else "看跌"
    strategy_name = strategy_definition(settings.strategy_key).name
    return f"{settings.symbol} {strategy_name}{_duration_label(settings.duration_minutes)} {direction} 置信{confidence:.1f}%"


def _fetch_latest_entry_price(symbol: str) -> float:
    row = fetch_premium_index(symbol)
    price = float(row.get("indexPrice") or 0)
    if price <= 0:
        raise ValueError("latest index price unavailable")
    return price


def _side_probability(probability_up: float, side: str) -> float:
    return probability_up if side == "BUY" else 1 - probability_up


def _duration_label(minutes: int) -> str:
    return "1天" if minutes == 1440 else f"{minutes}分钟"


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)
