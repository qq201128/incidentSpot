from __future__ import annotations

from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY

FACTOR_COMBO_TOP_SIMULATION_RANKS = (1, 2, 3)
FACTOR_COMBO_SHADOW_RANKS = (2, 3)


def factor_combo_simulation_strategy_key(rank: int) -> str:
    if rank == 1:
        return FACTOR_COMBO_STRATEGY_KEY
    return factor_combo_shadow_strategy_key(rank)


def factor_combo_shadow_strategy_key(rank: int) -> str:
    if rank not in FACTOR_COMBO_SHADOW_RANKS:
        raise ValueError(f"unsupported factor combo shadow rank: {rank}")
    return f"{FACTOR_COMBO_STRATEGY_KEY}_top{rank}"


def factor_combo_simulation_strategy_keys() -> tuple[str, ...]:
    return (
        FACTOR_COMBO_STRATEGY_KEY,
        *(factor_combo_shadow_strategy_key(rank) for rank in FACTOR_COMBO_SHADOW_RANKS),
    )
