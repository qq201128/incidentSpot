from __future__ import annotations

from typing import Any

from app.services.blind_reverse_martingale_strategy import predict_blind_reverse_martingale_direction
from app.services.factor_combo_strategy import predict_factor_combo_direction
from app.services.orderbook_notional_strategy import predict_orderbook_notional_direction
from app.services.orderbook_trade_flow_strategy import predict_orderbook_trade_flow_direction
from app.services.rule_config import RULE_DURATION, SUPPORTED_RULE_DURATIONS
from app.services.three_bar_10m_reverse_martingale_strategy import (
    predict_five_bar_10m_reverse_martingale_direction,
    predict_four_bar_10m_reverse_martingale_direction,
    predict_three_bar_10m_reverse_martingale_direction,
)
from app.services.strategy_registry import (
    BLIND_REVERSE_MARTINGALE_STRATEGY_KEY,
    DEFAULT_STRATEGY_KEY,
    FACTOR_COMBO_STRATEGY_KEY,
    FIVE_BAR_10M_RM_STRATEGY_KEY,
    FOUR_BAR_10M_RM_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_10M_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_15M_MG_51020_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_15M_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_MG_5102045_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_STRATEGY_KEY,
    ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY,
    ORDERBOOK_TRADE_FLOW_STRATEGY_KEY,
    THREE_BAR_10M_RM_STRATEGY_KEY,
    strategy_definition,
)

_ORDERBOOK_NOTIONAL_KEYS = frozenset(
    {
        ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY,
        ORDERBOOK_NOTIONAL_MG_5102045_STRATEGY_KEY,
        ORDERBOOK_NOTIONAL_10M_STRATEGY_KEY,
        ORDERBOOK_NOTIONAL_15M_STRATEGY_KEY,
        ORDERBOOK_NOTIONAL_15M_MG_51020_STRATEGY_KEY,
    }
)


def predict_rule_direction(
    symbol: str,
    duration: str = RULE_DURATION,
    *,
    entry_open_time: int | None = None,
    strategy_key: str | None = DEFAULT_STRATEGY_KEY,
) -> dict[str, Any]:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"rule engine supports only {sorted(SUPPORTED_RULE_DURATIONS)}, got {duration}")
    symbol = symbol.upper()
    strategy = strategy_definition(strategy_key)
    return _predict_strategy(
        strategy.key,
        symbol,
        duration,
        entry_open_time=entry_open_time,
        entry_grace_ms=strategy.entry_grace_ms,
    )


def _predict_strategy(
    strategy_key: str,
    symbol: str,
    duration: str,
    *,
    entry_open_time: int | None,
    entry_grace_ms: int,
) -> dict[str, Any]:
    for resolver in (_predict_factor_combo, _predict_orderbook, _predict_n_bar):
        result = resolver(
            strategy_key,
            symbol=symbol,
            duration=duration,
            entry_open_time=entry_open_time,
            entry_grace_ms=entry_grace_ms,
        )
        if result is not None:
            return result
    raise ValueError(f"unsupported strategy for live prediction: {strategy_key}")


def _predict_factor_combo(
    strategy_key: str,
    *,
    symbol: str,
    duration: str,
    entry_open_time: int | None,
    entry_grace_ms: int,
) -> dict[str, Any] | None:
    if strategy_key != FACTOR_COMBO_STRATEGY_KEY:
        return None
    return predict_factor_combo_direction(
        symbol,
        duration,
        entry_open_time=entry_open_time,
        entry_grace_ms=entry_grace_ms,
    )


def _predict_orderbook(
    strategy_key: str,
    *,
    symbol: str,
    duration: str,
    entry_open_time: int | None,
    entry_grace_ms: int,
) -> dict[str, Any] | None:
    del entry_grace_ms
    if strategy_key == ORDERBOOK_NOTIONAL_STRATEGY_KEY:
        return predict_orderbook_notional_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
        )
    if strategy_key in _ORDERBOOK_NOTIONAL_KEYS:
        return predict_orderbook_notional_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
            result_strategy_key=strategy_key,
        )
    if strategy_key == ORDERBOOK_TRADE_FLOW_STRATEGY_KEY:
        return predict_orderbook_trade_flow_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
        )
    if strategy_key == ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY:
        return predict_orderbook_trade_flow_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
            result_strategy_key=strategy_key,
        )
    if strategy_key == BLIND_REVERSE_MARTINGALE_STRATEGY_KEY:
        return predict_blind_reverse_martingale_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
        )
    return None


def _predict_n_bar(
    strategy_key: str,
    *,
    symbol: str,
    duration: str,
    entry_open_time: int | None,
    entry_grace_ms: int,
) -> dict[str, Any] | None:
    del entry_grace_ms
    if strategy_key == THREE_BAR_10M_RM_STRATEGY_KEY:
        return predict_three_bar_10m_reverse_martingale_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
        )
    if strategy_key == FOUR_BAR_10M_RM_STRATEGY_KEY:
        return predict_four_bar_10m_reverse_martingale_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
        )
    if strategy_key == FIVE_BAR_10M_RM_STRATEGY_KEY:
        return predict_five_bar_10m_reverse_martingale_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
        )
    return None
