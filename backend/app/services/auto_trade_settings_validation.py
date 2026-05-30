from __future__ import annotations

from app.services.auto_trade_types import AutoTradeSettings
from app.services.ensemble_judge_constants import ENSEMBLE_RANKER_STRATEGY_KEY
from app.services.factor_combo_simulation_keys import is_batch_combo_simulation_strategy
from app.services.model_family_config import is_model_family_shadow_strategy
from app.services.rule_config import DURATION_TO_MINUTES
from app.services.strategy_registry import strategy_definition, strategy_supports_duration

SUPPORTED_AUTO_DURATIONS = frozenset({"10m", "30m", "60m", "1d"})


def validated_auto_trade_settings(settings: AutoTradeSettings) -> AutoTradeSettings:
    strategy = strategy_definition(settings.strategy_key)
    _validate_strategy_constraints(settings, strategy)
    symbol = _normalized_symbol(settings.symbol)
    _validate_execution_shape(settings, strategy)
    return AutoTradeSettings(
        strategy_key=strategy.key,
        enabled=settings.enabled,
        symbol=symbol,
        duration=settings.duration,
        duration_minutes=DURATION_TO_MINUTES[settings.duration],
        qty=settings.qty,
        live_trading_enabled=settings.live_trading_enabled,
    )


def _validate_strategy_constraints(settings: AutoTradeSettings, strategy) -> None:
    if settings.enabled and not strategy.tradable:
        raise ValueError(strategy.disabled_reason or f"strategy is not tradable: {strategy.key}")
    if settings.live_trading_enabled and _simulation_only_strategy(strategy.key):
        raise ValueError(f"{strategy.key} supports simulation only; live trading must stay disabled")


def _simulation_only_strategy(strategy_key: str) -> bool:
    return (
        is_model_family_shadow_strategy(strategy_key)
        or is_batch_combo_simulation_strategy(strategy_key)
        or strategy_key == ENSEMBLE_RANKER_STRATEGY_KEY
    )


def _normalized_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if len(normalized) < 6:
        raise ValueError("symbol must contain at least 6 characters")
    return normalized


def _validate_execution_shape(settings: AutoTradeSettings, strategy) -> None:
    if settings.enabled and settings.duration not in SUPPORTED_AUTO_DURATIONS:
        raise ValueError("backend auto trade duration must be one of " + ", ".join(sorted(SUPPORTED_AUTO_DURATIONS)))
    if settings.enabled and not strategy_supports_duration(settings.strategy_key, settings.duration):
        supported = ", ".join(sorted(strategy.supported_durations))
        raise ValueError(f"strategy {strategy.key} does not support duration {settings.duration}, supported: {supported}")
    if settings.duration_minutes <= 0:
        raise ValueError("durationMinutes must be > 0")
    if settings.qty <= 0:
        raise ValueError("qty must be > 0")
