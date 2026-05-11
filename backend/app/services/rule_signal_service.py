from __future__ import annotations

from typing import Any

from app.services.blind_reverse_martingale_strategy import predict_blind_reverse_martingale_direction
from app.services.orderbook_notional_strategy import predict_orderbook_notional_direction
from app.services.orderbook_trade_flow_strategy import predict_orderbook_trade_flow_direction
from app.services.rule_config import RULE_DURATION
from app.services.three_bar_10m_reverse_martingale_strategy import (
    predict_five_bar_10m_reverse_martingale_direction,
    predict_four_bar_10m_reverse_martingale_direction,
    predict_three_bar_10m_reverse_martingale_direction,
)
from app.services.strategy_registry import (
    BLIND_REVERSE_MARTINGALE_STRATEGY_KEY,
    DEFAULT_STRATEGY_KEY,
    FIVE_BAR_10M_RM_STRATEGY_KEY,
    FOUR_BAR_10M_RM_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_MG_5102045_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_STRATEGY_KEY,
    ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY,
    ORDERBOOK_TRADE_FLOW_STRATEGY_KEY,
    THREE_BAR_10M_RM_STRATEGY_KEY,
    strategy_definition,
)


def predict_rule_direction(
    symbol: str,
    duration: str = RULE_DURATION,
    *,
    entry_open_time: int | None = None,
    strategy_key: str | None = DEFAULT_STRATEGY_KEY,
) -> dict[str, Any]:
    if duration != RULE_DURATION:
        raise ValueError(f"rule engine supports only {RULE_DURATION}, got {duration}")
    symbol = symbol.upper()
    strategy = strategy_definition(strategy_key)
    if strategy.key == ORDERBOOK_NOTIONAL_STRATEGY_KEY:
        return predict_orderbook_notional_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
        )
    if strategy.key in (ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY, ORDERBOOK_NOTIONAL_MG_5102045_STRATEGY_KEY):
        return predict_orderbook_notional_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
            result_strategy_key=strategy.key,
        )
    if strategy.key == ORDERBOOK_TRADE_FLOW_STRATEGY_KEY:
        return predict_orderbook_trade_flow_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
        )
    if strategy.key == ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY:
        return predict_orderbook_trade_flow_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
            result_strategy_key=strategy.key,
        )
    if strategy.key == BLIND_REVERSE_MARTINGALE_STRATEGY_KEY:
        return predict_blind_reverse_martingale_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
        )
    if strategy.key == THREE_BAR_10M_RM_STRATEGY_KEY:
        return predict_three_bar_10m_reverse_martingale_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
        )
    if strategy.key == FOUR_BAR_10M_RM_STRATEGY_KEY:
        return predict_four_bar_10m_reverse_martingale_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
        )
    if strategy.key == FIVE_BAR_10M_RM_STRATEGY_KEY:
        return predict_five_bar_10m_reverse_martingale_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
        )
    raise ValueError(f"unsupported strategy for live prediction: {strategy.key}")
