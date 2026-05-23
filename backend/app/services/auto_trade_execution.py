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
from app.services.factor_combo_simulation_keys import is_batch_combo_simulation_strategy
from app.services.strategy_registry import strategy_definition

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
            prediction_open_time=int(prediction["open_time"]) if prediction.get("open_time") is not None else None,
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
        order=AutoTradeOrderPayload(side=side, qty=float(settings.qty), price=FIXED_PAYOUT_RATIO),
    )


def _event_title(settings: AutoTradeSettings, prediction: dict[str, Any], side: str) -> str:
    probability = _side_probability(float(prediction["probability_up"]), side)
    confidence = round(probability * PERCENT_SCALE) / PERCENT_DECIMALS
    direction = "看涨" if side == "BUY" else "看跌"
    strategy_name = _event_strategy_label(settings, prediction)
    return f"{settings.symbol} {strategy_name}{_duration_label(settings.duration_minutes)} {direction} 置信{confidence:.1f}%"


def _event_strategy_label(settings: AutoTradeSettings, prediction: dict[str, Any]) -> str:
    combo_rule = str(prediction.get("high_winrate_rule") or "").strip()
    if is_batch_combo_simulation_strategy(settings.strategy_key) and combo_rule:
        if combo_rule.startswith("goal_combo__"):
            return f"GE70·{combo_rule.removeprefix('goal_combo__')[:48]}"
        return f"组合·{combo_rule[:48]}"
    return strategy_definition(settings.strategy_key).name


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
