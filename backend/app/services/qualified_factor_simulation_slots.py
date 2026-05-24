from __future__ import annotations

from typing import Any

from app.db.session import get_conn
from app.services.factor_backtest_gate import meets_backtest_gate
from app.services.factor_candidate_signal_keys import factor_candidate_signal_key
from app.services.factor_combo_batch_predictions import eligible_factor_combo_rows
from app.services.factor_combo_simulation_keys import simulation_strategy_key_for_factor_name
from app.services.factor_learning_common import utc_now
from app.services.factor_ranking_cache_service import get_cached_ranking
from app.services.rule_config import DURATION_TO_MINUTES, SUPPORTED_RULE_DURATIONS


def sync_qualified_simulation_slots(symbol: str, duration: str, *, qty: float = 5.0) -> dict[str, Any]:
    sym = symbol.strip().upper()
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    single_keys = _qualified_single_factor_strategy_keys(sym, duration)
    combo_keys = _qualified_combo_strategy_keys(sym, duration)
    enabled = _enable_simulation_slots(sym, duration, [*single_keys, *combo_keys], qty=qty)
    return {
        "symbol": sym,
        "duration": duration,
        "singleFactorSlots": len(single_keys),
        "comboFactorSlots": len(combo_keys),
        "enabledSlots": enabled,
        "thresholds": {
            "minWinRate": 0.62,
            "minProfitFactor": 1.05,
            "minTotalPeriods": 100,
        },
    }


def _qualified_single_factor_strategy_keys(symbol: str, duration: str) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for row in _single_factor_candidate_rows(symbol, duration):
        if not meets_backtest_gate(row):
            continue
        strategy_key = factor_candidate_signal_key(str(row["factorName"]))
        if strategy_key in seen:
            continue
        seen.add(strategy_key)
        keys.append(strategy_key)
    return keys


def _single_factor_candidate_rows(symbol: str, duration: str) -> list[dict[str, Any]]:
    from app.services.agent_mined_factor_library import agent_factor_rows_for_duration

    rows: list[dict[str, Any]] = []
    cache = get_cached_ranking(symbol, duration)
    if cache is not None:
        rows.extend(dict(row) for row in cache.get("ranking") or [] if isinstance(row, dict))
    for row in agent_factor_rows_for_duration(symbol, duration):
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        rows.append(
            {
                **row,
                "winRate": metrics.get("winRate"),
                "profitFactor": metrics.get("profitFactor"),
                "totalPeriods": metrics.get("totalPeriods"),
            }
        )
    return rows


def _qualified_combo_strategy_keys(symbol: str, duration: str) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for row in eligible_factor_combo_rows(symbol, duration):
        strategy_key = simulation_strategy_key_for_factor_name(str(row["factorName"]))
        if strategy_key in seen:
            continue
        seen.add(strategy_key)
        keys.append(strategy_key)
    return keys


def _enable_simulation_slots(
    symbol: str,
    duration: str,
    strategy_keys: list[str],
    *,
    qty: float,
) -> int:
    if not strategy_keys:
        return 0
    ts = utc_now()
    enabled = 0
    conn = get_conn()
    try:
        for strategy_key in strategy_keys:
            conn.execute(
                """
                INSERT OR REPLACE INTO auto_trade_strategies(
                  strategy_key, duration, enabled, live_trading_enabled, symbol, duration_minutes, qty, updated_at
                )
                VALUES(?, ?, 1, 0, ?, ?, ?, ?)
                """,
                (
                    strategy_key,
                    duration,
                    symbol,
                    int(DURATION_TO_MINUTES[duration]),
                    float(qty),
                    ts,
                ),
            )
            enabled += 1
        conn.commit()
    finally:
        conn.close()
    return enabled
