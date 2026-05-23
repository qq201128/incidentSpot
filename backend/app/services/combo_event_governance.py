from __future__ import annotations

from app.services.batch_combo_event_demotion import evaluate_batch_combo_event_demotion
from app.services.combo_event_governance_cache import (
    get_cached_governance,
    get_cached_shadow_report,
    get_warm_monitoring_snapshot,
)
from app.services.high_winrate_strategy_demotion import evaluate_high_winrate_demotion
from app.services.shadow_event_deviation_service import shadow_event_deviation_report


def compute_combo_event_governance(symbol: str, duration: str, *, allow_goal_refresh: bool = False) -> dict:
    sym = symbol.strip().upper()
    return {
        "symbol": sym,
        "duration": duration,
        "highWinrateStatus": evaluate_high_winrate_demotion(
            sym,
            duration,
            allow_goal_refresh=allow_goal_refresh,
        ),
        "batchComboDemotion": evaluate_batch_combo_event_demotion(sym, duration),
    }


def compute_combo_event_monitoring(symbol: str, duration: str, *, allow_goal_refresh: bool = False) -> dict:
    sym = symbol.strip().upper()
    governance = compute_combo_event_governance(sym, duration, allow_goal_refresh=allow_goal_refresh)
    return {
        "symbol": sym,
        "duration": duration,
        "shadowEventDeviation": shadow_event_deviation_report(sym, duration),
        "highWinrateStatus": governance["highWinrateStatus"],
        "batchComboDemotion": governance["batchComboDemotion"],
    }


def run_combo_event_governance(symbol: str, duration: str, *, allow_goal_refresh: bool = False) -> dict:
    return compute_combo_event_governance(symbol, duration, allow_goal_refresh=allow_goal_refresh)


def combo_event_monitoring(symbol: str, duration: str) -> dict:
    sym = symbol.strip().upper()
    return get_warm_monitoring_snapshot(sym, duration)


def cached_combo_event_governance(symbol: str, duration: str) -> dict:
    sym = symbol.strip().upper()
    return get_cached_governance(
        sym,
        duration,
        compute=lambda s, d: compute_combo_event_governance(s, d, allow_goal_refresh=False),
    )


def cached_shadow_event_deviation_report(symbol: str, duration: str) -> dict:
    sym = symbol.strip().upper()
    return get_cached_shadow_report(
        sym,
        duration,
        compute=shadow_event_deviation_report,
    )
