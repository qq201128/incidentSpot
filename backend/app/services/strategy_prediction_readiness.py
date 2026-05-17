from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.factor_cache_metadata import cache_is_usable_for_live_signal, live_signal_cache_reason
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.high_winrate_combo_cache_service import get_cached_high_winrate_combo_ranking
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY, HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY

RECOVERABLE_REASONS = {
    "ranking_cache_missing",
    "ranking_cache_empty",
    "ranking_cache_market_data_changed",
    "ranking_cache_legacy_without_fingerprint",
}
FAILURE_LIMIT = 20


@dataclass(frozen=True)
class PredictionReadiness:
    ready: bool
    reason: str
    recoverable: bool = False
    recovery_attempted: bool = False
    recovery_status: str | None = None
    diagnostics: dict[str, Any] | None = None


def strategy_prediction_readiness(
    strategy_key: str,
    symbol: str,
    duration: str,
    *,
    attempt_recovery: bool = False,
) -> PredictionReadiness:
    if strategy_key == FACTOR_COMBO_STRATEGY_KEY:
        return _readiness_with_recovery(
            symbol,
            duration,
            get_cached_combination_ranking,
            _recover_factor_combo_ranking,
            attempt_recovery=attempt_recovery,
        )
    if strategy_key == HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY:
        return _readiness_with_recovery(
            symbol,
            duration,
            get_cached_high_winrate_combo_ranking,
            _recover_high_winrate_ranking,
            attempt_recovery=attempt_recovery,
        )
    return PredictionReadiness(True, "ready")


def _readiness_with_recovery(
    symbol: str,
    duration: str,
    loader,
    recovery,
    *,
    attempt_recovery: bool,
) -> PredictionReadiness:
    initial = _ranking_cache_readiness(loader(symbol, duration))
    if initial.ready or not attempt_recovery or not initial.recoverable:
        return initial
    diagnostics = recovery(symbol, duration)
    after = _ranking_cache_readiness(loader(symbol, duration))
    if after.ready:
        return PredictionReadiness(
            True,
            "ready",
            recoverable=False,
            recovery_attempted=True,
            recovery_status="recovered",
            diagnostics=diagnostics,
        )
    return PredictionReadiness(
        after.ready,
        after.reason,
        recoverable=after.recoverable,
        recovery_attempted=True,
        recovery_status="failed",
        diagnostics={**(diagnostics or {}), **(after.diagnostics or {})},
    )


def _ranking_cache_readiness(cache: dict[str, Any] | None) -> PredictionReadiness:
    if cache is None:
        return _not_ready("ranking_cache_missing", {})
    if not cache_is_usable_for_live_signal(cache):
        return _not_ready(f"ranking_cache_{live_signal_cache_reason(cache)}", _cache_diagnostics(cache))
    ranking = cache.get("ranking")
    if not isinstance(ranking, list):
        return _not_ready("ranking_cache_malformed", _cache_diagnostics(cache))
    if not ranking:
        return _not_ready("ranking_cache_empty", _cache_diagnostics(cache))
    return PredictionReadiness(True, "ready")


def _not_ready(reason: str, diagnostics: dict[str, Any]) -> PredictionReadiness:
    return PredictionReadiness(
        False,
        reason,
        recoverable=reason in RECOVERABLE_REASONS,
        diagnostics=diagnostics,
    )


def _recover_factor_combo_ranking(symbol: str, duration: str) -> dict[str, Any]:
    from app.services.factor_combination_background import _refresh_duration_klines
    from app.services.factor_combination_cache_service import save_cached_combination_ranking
    from app.services.factor_combination_service import run_factor_combination_ranking
    from app.services.experiment_profiles import EXPERIMENT_PROFILE_FAST, combination_search_config_for_profile
    from app.services.factor_mined_library import (
        mined_factor_rows_for_duration,
        upsert_good_combinations,
    )

    sym = symbol.strip().upper()
    _refresh_duration_klines(sym, duration)
    library_report = _library_factor_combo_report(sym, duration)
    if library_report is not None:
        save_cached_combination_ranking(library_report)
        diagnostics = _ranking_report_diagnostics(
            library_report,
            {
                "symbol": sym,
                "duration": duration,
                "promoted": 0,
                "libraryTotal": len(mined_factor_rows_for_duration(sym, duration)),
            },
        )
        diagnostics["recoveryProfile"] = "library"
        return diagnostics
    report = run_factor_combination_ranking(sym, duration)
    if not (report.get("ranking") or []):
        report = run_factor_combination_ranking(
            sym,
            duration,
            combination_search_config_for_profile(EXPERIMENT_PROFILE_FAST),
        )
    save_cached_combination_ranking(report)
    promotion = upsert_good_combinations(report)
    diagnostics = _ranking_report_diagnostics(report, promotion)
    diagnostics["recoveryProfile"] = (
        "default"
        if report.get("searchConfig", {}).get("baseFactorLimit") == 16
        else "fast"
    )
    return diagnostics


def _library_factor_combo_report(symbol: str, duration: str) -> dict[str, Any] | None:
    from app.services.factor_combination_payloads import config_payload
    from app.services.experiment_profiles import EXPERIMENT_PROFILE_FAST, combination_search_config_for_profile
    from app.services.factor_frame_service import load_factor_frame
    from app.services.factor_learning_controls import learning_blocked_factor_names, load_factor_learning_memory_for
    from app.services.factor_mined_candidates import materialize_mined_factor_frame_for_rows
    from app.services.factor_mined_library import (
        mined_factor_rows_for_duration,
        regular_library_combination_rows_for_duration,
    )

    cfg = combination_search_config_for_profile(EXPERIMENT_PROFILE_FAST)
    source_rows = mined_factor_rows_for_duration(symbol, duration)
    ranking = regular_library_combination_rows_for_duration(
        symbol,
        duration,
        limit=cfg.result_limit,
    )
    if not ranking:
        return None
    learning_memory = load_factor_learning_memory_for(symbol, duration)
    materialized = materialize_mined_factor_frame_for_rows(
        load_factor_frame(symbol, duration),
        symbol=symbol,
        duration=duration,
        target_rows=ranking,
        source_rows=source_rows,
        excluded_factor_names=learning_blocked_factor_names(learning_memory),
    )
    available = {str(column) for column in materialized.frame.columns}
    ranking = [row for row in ranking if str(row.get("factorName")) in available]
    if not ranking:
        return None
    return {
        "symbol": symbol,
        "duration": duration,
        "ranking": ranking,
        "total": len(ranking),
        "searchConfig": {
            **config_payload(cfg),
            "source": "mined_factor_library",
        },
        "baseFactors": [],
        "baseFactorCount": 0,
        "minedFactorSourceCount": len(source_rows),
        "minedFactorUsedCount": len(ranking),
        "agentMinedFactorUsedCount": 0,
        "testedCombinationCount": 0,
        "failureCount": len(materialized.failures),
        "failures": list(materialized.failures)[:FAILURE_LIMIT],
    }


def _recover_high_winrate_ranking(symbol: str, duration: str) -> dict[str, Any]:
    from app.services.high_winrate_strategy_rotation import refresh_high_winrate_goal

    report = refresh_high_winrate_goal(symbol.strip().upper(), duration)
    return _ranking_report_diagnostics(report, report.get("promotion"))


def _ranking_report_diagnostics(
    report: dict[str, Any],
    promotion: dict[str, Any] | None,
) -> dict[str, Any]:
    failures = report.get("failures") or []
    return {
        "symbol": report.get("symbol"),
        "duration": report.get("duration"),
        "rankingTotal": len(report.get("ranking") or []),
        "testedCombinationCount": report.get("testedCombinationCount"),
        "baseFactorCount": report.get("baseFactorCount"),
        "failureCount": report.get("failureCount") or len(failures),
        "failures": failures[:FAILURE_LIMIT] if isinstance(failures, list) else [],
        "promotion": promotion or {},
    }


def _cache_diagnostics(cache: dict[str, Any]) -> dict[str, Any]:
    ranking = cache.get("ranking")
    failures = cache.get("failures") or []
    return {
        "rankingTotal": len(ranking) if isinstance(ranking, list) else None,
        "testedCombinationCount": cache.get("testedCombinationCount"),
        "baseFactorCount": cache.get("baseFactorCount"),
        "failureCount": cache.get("failureCount") or len(failures),
        "failures": failures[:FAILURE_LIMIT] if isinstance(failures, list) else [],
        "cacheStatus": cache.get("cacheStatus"),
        "updatedAt": cache.get("updatedAt"),
    }
