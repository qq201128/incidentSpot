from __future__ import annotations

from app.services.batch_combo_event_demotion import evaluate_batch_combo_event_demotion
from app.services.factor_candidate_event_demotion import evaluate_factor_candidate_event_demotion
from app.services.combo_event_governance_cache import (
    get_cached_governance,
    get_cached_shadow_report,
    get_warm_monitoring_snapshot,
)
from app.services.shadow_event_deviation_service import shadow_event_deviation_report


def compute_combo_event_governance(symbol: str, duration: str) -> dict:
    sym = symbol.strip().upper()
    return {
        "symbol": sym,
        "duration": duration,
        "batchComboDemotion": evaluate_batch_combo_event_demotion(sym, duration),
    }


def compute_combo_event_monitoring(symbol: str, duration: str) -> dict:
    sym = symbol.strip().upper()
    batch_combo = evaluate_batch_combo_event_demotion(sym, duration)
    factor_candidate = evaluate_factor_candidate_event_demotion(sym, duration)
    return {
        "symbol": sym,
        "duration": duration,
        "shadowEventDeviation": shadow_event_deviation_report(sym, duration),
        "batchComboDemotion": batch_combo,
        "factorCandidateDemotion": factor_candidate,
        "simulationObservation": _simulation_observation_summary(batch_combo, factor_candidate),
    }


def _simulation_observation_summary(batch_combo: dict, factor_candidate: dict) -> dict:
    watchlist = [
        *batch_combo.get("watchlist", []),
        *factor_candidate.get("watchlist", []),
    ]
    return {
        "evaluatedCount": int(batch_combo.get("evaluatedCount") or 0) + int(factor_candidate.get("evaluatedCount") or 0),
        "watchlistCount": len(watchlist),
        "batchComboWatchlistCount": int(batch_combo.get("watchlistCount") or 0),
        "factorCandidateWatchlistCount": int(factor_candidate.get("watchlistCount") or 0),
    }


def run_combo_event_governance(symbol: str, duration: str) -> dict:
    return compute_combo_event_governance(symbol, duration)


def combo_event_monitoring(symbol: str, duration: str) -> dict:
    sym = symbol.strip().upper()
    return get_warm_monitoring_snapshot(sym, duration)


def cached_combo_event_governance(symbol: str, duration: str) -> dict:
    sym = symbol.strip().upper()
    return get_cached_governance(
        sym,
        duration,
        compute=lambda s, d: compute_combo_event_governance(s, d),
    )


def cached_shadow_event_deviation_report(symbol: str, duration: str) -> dict:
    sym = symbol.strip().upper()
    return get_cached_shadow_report(
        sym,
        duration,
        compute=shadow_event_deviation_report,
    )
