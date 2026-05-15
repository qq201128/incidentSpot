from __future__ import annotations

from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY, HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY

FACTOR_COMBO_TOP_SIMULATION_RANKS = (1, 2, 3)
FACTOR_COMBO_SHADOW_RANKS = (2, 3)
HIGH_WINRATE_COMBO_NAME_PREFIX = "goal_combo__"


def factor_combo_simulation_strategy_key(rank: int) -> str:
    if rank == 1:
        return FACTOR_COMBO_STRATEGY_KEY
    return factor_combo_shadow_strategy_key(rank)


def factor_combo_shadow_strategy_key(rank: int) -> str:
    if rank not in FACTOR_COMBO_SHADOW_RANKS:
        raise ValueError(f"unsupported factor combo shadow rank: {rank}")
    return f"{FACTOR_COMBO_STRATEGY_KEY}_top{rank}"


def high_winrate_factor_combo_simulation_strategy_key(rank: int) -> str:
    if rank == 1:
        return HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY
    if rank not in FACTOR_COMBO_SHADOW_RANKS:
        raise ValueError(f"unsupported high-winrate combo shadow rank: {rank}")
    return f"{HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY}_top{rank}"


def simulation_strategy_key_for_combo(factor_name: str, rank: int) -> str:
    if is_high_winrate_combo_name(factor_name):
        return high_winrate_factor_combo_simulation_strategy_key(rank)
    return factor_combo_simulation_strategy_key(rank)


def is_high_winrate_combo_name(factor_name: str | None) -> bool:
    return str(factor_name or "").startswith(HIGH_WINRATE_COMBO_NAME_PREFIX)


def factor_combo_simulation_strategy_keys() -> tuple[str, ...]:
    return (
        FACTOR_COMBO_STRATEGY_KEY,
        *(factor_combo_shadow_strategy_key(rank) for rank in FACTOR_COMBO_SHADOW_RANKS),
    )


def high_winrate_factor_combo_simulation_strategy_keys() -> tuple[str, ...]:
    return (
        HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
        *(high_winrate_factor_combo_simulation_strategy_key(rank) for rank in FACTOR_COMBO_SHADOW_RANKS),
    )
