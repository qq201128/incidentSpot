from __future__ import annotations

from app.services.batch_combo_event_demotion import evaluate_batch_combo_event_demotion
from app.services.factor_candidate_event_demotion import evaluate_factor_candidate_event_demotion
from app.services.combo_event_governance_cache import (
    get_cached_governance,
    get_cached_shadow_report,
    get_warm_monitoring_snapshot,
)
from app.services.model_shadow_simulation_monitor import model_shadow_simulation_report
from app.services.shadow_event_deviation_service import shadow_event_deviation_report
from app.services.simulation_event_demotion import sort_simulation_rows_by_win_rate


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
    model_shadow = model_shadow_simulation_report(sym, duration)
    return {
        "symbol": sym,
        "duration": duration,
        "shadowEventDeviation": shadow_event_deviation_report(sym, duration),
        "batchComboDemotion": batch_combo,
        "factorCandidateDemotion": factor_candidate,
        "modelShadowSimulation": model_shadow,
        "simulationObservation": _simulation_observation_summary(batch_combo, factor_candidate, model_shadow),
    }


def _simulation_observation_summary(batch_combo: dict, factor_candidate: dict, model_shadow: dict) -> dict:
    watchlist = [
        *batch_combo.get("watchlist", []),
        *factor_candidate.get("watchlist", []),
    ]
    model_summary = model_shadow.get("summary") or {}
    return {
        "evaluatedCount": int(batch_combo.get("evaluatedCount") or 0) + int(factor_candidate.get("evaluatedCount") or 0),
        "watchlistCount": len(watchlist),
        "batchComboWatchlistCount": int(batch_combo.get("watchlistCount") or 0),
        "factorCandidateWatchlistCount": int(factor_candidate.get("watchlistCount") or 0),
        "modelShadowEventCount": int(model_summary.get("simulationEventCount") or 0),
        "modelShadowQualityPassedCount": int(model_summary.get("qualityPassedCount") or 0),
    }


def run_combo_event_governance(symbol: str, duration: str) -> dict:
    return compute_combo_event_governance(symbol, duration)


def combo_event_monitoring(symbol: str, duration: str) -> dict:
    sym = symbol.strip().upper()
    return _sorted_monitoring_payload(get_warm_monitoring_snapshot(sym, duration))


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


def _sorted_monitoring_payload(payload: dict) -> dict:
    return {
        **payload,
        "batchComboDemotion": _sorted_demotion_section(payload["batchComboDemotion"]),
        "factorCandidateDemotion": _sorted_demotion_section(payload["factorCandidateDemotion"]),
    }


def _sorted_demotion_section(section: dict) -> dict:
    return {
        **section,
        "evaluations": sort_simulation_rows_by_win_rate(section["evaluations"]),
        "watchlist": sort_simulation_rows_by_win_rate(section["watchlist"]),
    }
