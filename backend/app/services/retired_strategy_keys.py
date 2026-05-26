from __future__ import annotations

RETIRED_AUTO_TRADE_STRATEGY_KEYS: frozenset[str] = frozenset({
    "complete_day_10m_production",
    "vegas_fib_resonance",
    "high_winrate_rules",
    "pure_rule_precision",
    "win70_trade_max_rules",
    "daily_trade_floor_tree",
    "orderbook_notional_40m",
    "orderbook_notional_40m_mg",
    "orderbook_notional_10m_mg_5102045",
    "orderbook_notional_10m",
    "orderbook_notional_15m",
    "orderbook_notional_15m_mg_51020",
    "orderbook_trade_flow_1k",
    "orderbook_trade_flow_1k_invert_mg",
    "blind_reverse_martingale_v1",
    "three_bar_10m_reverse_martingale_v1",
    "four_bar_10m_reverse_martingale_v1",
    "five_bar_10m_reverse_martingale_v1",
    "high_winrate_factor_combo_v1",
})


def is_retired_strategy_key(key: str | None) -> bool:
    if not key:
        return False
    return key.strip() in RETIRED_AUTO_TRADE_STRATEGY_KEYS


def retired_strategy_sql_filter(
    *,
    signal_column: str = "signal_key",
    strategy_column: str = "strategy_key",
    table_prefix: str = "",
) -> tuple[str, tuple[str, ...]]:
    keys = tuple(sorted(RETIRED_AUTO_TRADE_STRATEGY_KEYS))
    if not keys:
        return "", ()
    prefix = f"{table_prefix}." if table_prefix else ""
    placeholders = ", ".join("?" * len(keys))
    clause = (
        f" AND {prefix}{signal_column} NOT IN ({placeholders})"
        f" AND {prefix}{strategy_column} NOT IN ({placeholders})"
    )
    return clause, keys + keys
