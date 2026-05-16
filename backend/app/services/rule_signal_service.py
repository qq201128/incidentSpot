from __future__ import annotations

from typing import Any

from app.services.factor_combo_strategy import (
    predict_factor_combo_direction,
    predict_high_winrate_factor_combo_direction,
)
from app.services.lstm_prediction_service import predict_lstm_shadow_prediction
from app.services.rule_config import RULE_DURATION, SUPPORTED_RULE_DURATIONS
from app.services.strategy_registry import (
    DEFAULT_STRATEGY_KEY,
    FACTOR_COMBO_STRATEGY_KEY,
    HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
    is_lstm_shadow_strategy,
    strategy_definition,
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
    if duration not in strategy.supported_durations:
        supported = ", ".join(sorted(strategy.supported_durations))
        raise ValueError(f"strategy {strategy.key} supports only {supported}, got {duration}")
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
    for resolver in (_predict_lstm, _predict_factor_combo):
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


def _predict_lstm(
    strategy_key: str,
    *,
    symbol: str,
    duration: str,
    entry_open_time: int | None,
    entry_grace_ms: int,
) -> dict[str, Any] | None:
    del entry_grace_ms
    if not is_lstm_shadow_strategy(strategy_key):
        return None
    return predict_lstm_shadow_prediction(symbol, duration, entry_open_time=entry_open_time)


def _predict_factor_combo(
    strategy_key: str,
    *,
    symbol: str,
    duration: str,
    entry_open_time: int | None,
    entry_grace_ms: int,
) -> dict[str, Any] | None:
    if strategy_key != FACTOR_COMBO_STRATEGY_KEY:
        if strategy_key != HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY:
            return None
        return predict_high_winrate_factor_combo_direction(
            symbol,
            duration,
            entry_open_time=entry_open_time,
            entry_grace_ms=entry_grace_ms,
        )
    return predict_factor_combo_direction(
        symbol,
        duration,
        entry_open_time=entry_open_time,
        entry_grace_ms=entry_grace_ms,
    )
