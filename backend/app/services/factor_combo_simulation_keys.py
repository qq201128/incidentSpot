from __future__ import annotations

import hashlib

from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY, HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY

FACTOR_COMBO_TOP_SIMULATION_RANKS = (1, 2, 3)
FACTOR_COMBO_SHADOW_RANKS = (2, 3)
HIGH_WINRATE_COMBO_NAME_PREFIX = "goal_combo__"
BATCH_COMBO_KEY_PREFIX = f"{FACTOR_COMBO_STRATEGY_KEY}_combo_"
BATCH_HIGH_WINRATE_KEY_PREFIX = f"{HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY}_combo_"


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
    if rank in FACTOR_COMBO_TOP_SIMULATION_RANKS:
        return factor_combo_simulation_strategy_key(rank)
    return simulation_strategy_key_for_factor_name(factor_name)


def simulation_strategy_key_for_factor_name(factor_name: str) -> str:
    """One batch simulation key per combo factor name (combo__ and goal_combo__ share the same prefix)."""
    digest = hashlib.sha1(str(factor_name).encode("utf-8")).hexdigest()[:12]
    return f"{BATCH_COMBO_KEY_PREFIX}{digest}"


def is_batch_combo_simulation_strategy(strategy_key: str | None) -> bool:
    key = str(strategy_key or "")
    return key.startswith(BATCH_COMBO_KEY_PREFIX) or key.startswith(BATCH_HIGH_WINRATE_KEY_PREFIX)


def factor_combo_event_strategy_filter() -> tuple[str, tuple[str, ...]]:
    """SQL clause matching factor-combo primary, shadow, and per-combo batch simulation keys."""
    static = (
        *factor_combo_simulation_strategy_keys(),
        *high_winrate_factor_combo_simulation_strategy_keys(),
    )
    placeholders = ",".join("?" for _key in static)
    clause = (
        f"(strategy_key IN ({placeholders}) "
        f"OR strategy_key LIKE ? OR strategy_key LIKE ?)"
    )
    return clause, (*static, f"{BATCH_COMBO_KEY_PREFIX}%", f"{BATCH_HIGH_WINRATE_KEY_PREFIX}%")


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
