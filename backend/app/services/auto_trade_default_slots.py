from __future__ import annotations

from typing import Any

from app.services.model_family_config import MODEL_FAMILIES, model_family_strategy_key
from app.services.runtime_symbols import configured_runtime_symbols
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY

DEFAULT_SIMULATION_STRATEGY_KEYS = frozenset({FACTOR_COMBO_STRATEGY_KEY})


def default_slot_flags(strategy_key: str) -> tuple[int, int]:
    enabled = int(_is_default_simulation_strategy(strategy_key))
    return enabled, 0


def default_slot_enabled(strategy_key: str) -> bool:
    enabled, _live = default_slot_flags(strategy_key)
    return bool(enabled)


def default_live_trading_enabled(strategy_key: str) -> bool:
    _enabled, live = default_slot_flags(strategy_key)
    return bool(live)


def enable_default_simulation_strategy_slots(
    conn: Any,
    durations: tuple[str, ...],
    duration_minutes: dict[str, int],
    updated_at: str,
) -> None:
    for symbol in configured_runtime_symbols():
        for key, duration in _default_simulation_slots(durations):
            _enable_slot(conn, key, symbol, duration, duration_minutes[duration], updated_at)


def disable_simulation_only_live_trading(conn: Any, durations: tuple[str, ...]) -> None:
    for key in _simulation_only_strategy_keys(durations):
        conn.execute(
            """
            UPDATE auto_trade_strategies
            SET live_trading_enabled = 0
            WHERE strategy_key = ?
            """,
            (key,),
        )


def _is_default_simulation_strategy(strategy_key: str) -> bool:
    return strategy_key in DEFAULT_SIMULATION_STRATEGY_KEYS


def _default_simulation_slots(durations: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((key, duration) for key in DEFAULT_SIMULATION_STRATEGY_KEYS for duration in durations)


def _simulation_only_strategy_keys(durations: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        model_family_strategy_key(family, duration)
        for family in MODEL_FAMILIES
        for duration in durations
    )


def _enable_slot(
    conn: Any,
    strategy_key: str,
    symbol: str,
    duration: str,
    duration_minutes: int,
    updated_at: str,
) -> None:
    conn.execute(
        """
        UPDATE auto_trade_strategies
        SET enabled = 1,
            live_trading_enabled = 0,
            duration_minutes = ?,
            updated_at = ?
        WHERE strategy_key = ?
          AND symbol = ?
          AND duration = ?
          AND enabled = 0
          AND live_trading_enabled = 0
        """,
        (duration_minutes, updated_at, strategy_key, symbol, duration),
    )
