from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable

from app.services.experiment_profiles import (
    EXPERIMENT_PROFILE_FAST,
    combination_search_config_for_profile,
    normalize_experiment_profile,
)
from app.services.factor_combination_cache_service import save_cached_combination_ranking
from app.services.factor_combination_service import run_factor_combination_ranking
from app.services.factor_mined_library import upsert_good_combinations
from app.services.factor_ranking_cache_service import factor_ranking_precomputed_symbols
from app.services.kline_prediction_refresh import refresh_prediction_klines
from app.services.kline_timing import MS_PER_MINUTE, current_rule_entry_open_time_for_duration
from app.services.lstm_candidate_library import attempted_search_keys, record_lstm_candidate
from app.services.lstm_candidate_keys import search_key_for_config
from app.services.lstm_candidate_progress import (
    complete_lstm_candidate_progress,
    finish_lstm_candidate_progress,
    start_lstm_candidate_progress,
)
from app.services.lstm_candidate_search import (
    LstmCandidateSearchConfig,
    LstmCandidateSearchRequest,
    next_candidate_configs,
    search_space_size,
)
from app.services.lstm_config import LstmTrainingConfig
from app.services.lstm_prediction_service import lstm_model_status
from app.services.lstm_training_service import publish_lstm_staged_model, train_lstm_model
from app.services.rule_config import DURATION_TO_MINUTES, SUPPORTED_RULE_DURATIONS
from app.services.lstm_candidate_retry_payloads import (
    duration_ms as _duration_ms,
    failed_training_report as _failed_training_report,
    normalized_symbols as _normalized_symbols,
    search_exhausted_result as _search_exhausted_result,
    skipped_result as _skipped_result,
    summary_status as _summary_status,
    trained_result as _trained_result,
    utc_now as _utc_now,
    validated_durations as _validated_durations,
)

DEFAULT_RETRY_DURATIONS = ("10m", "60m")
RETRY_TRAIN_STATUSES = {"untrained", "validation_failed", "failed", "insufficient_samples"}
TRAINED_STATUSES = {"trained", "shadow_active", "trade_active"}


@dataclass(frozen=True)
class LstmCandidateRetryConfig:
    symbols: tuple[str, ...] = ()
    durations: tuple[str, ...] = DEFAULT_RETRY_DURATIONS
    profile: str = EXPERIMENT_PROFILE_FAST
    search: LstmCandidateSearchConfig = LstmCandidateSearchConfig()
    manual_trigger: bool = False
    reset_history: bool = False


@dataclass(frozen=True)
class LstmCandidateRetryDependencies:
    lstm_status: Callable[..., dict[str, Any]]
    refresh_klines: Callable[[str, str, int], None]
    run_combination_ranking: Callable[..., dict[str, Any]]
    save_combination_ranking: Callable[[dict[str, Any]], None]
    promote_combinations: Callable[[dict[str, Any]], dict[str, Any]]
    train_lstm: Callable[[LstmTrainingConfig], dict[str, Any]]
    attempted_keys: Callable[[str, str], frozenset[str]] = attempted_search_keys
    record_candidate: Callable[[LstmTrainingConfig, str, dict[str, Any]], dict[str, Any]] = record_lstm_candidate
    publish_trade_candidate: Callable[[LstmTrainingConfig, dict[str, Any]], None] = publish_lstm_staged_model
    start_progress: Callable[..., dict[str, Any]] = start_lstm_candidate_progress
    complete_progress: Callable[..., dict[str, Any]] = complete_lstm_candidate_progress
    finish_progress: Callable[..., dict[str, Any]] = finish_lstm_candidate_progress


@dataclass(frozen=True)
class CandidateTrainingResult:
    config: LstmTrainingConfig
    report: dict[str, Any]


def run_lstm_candidate_retry(
    config: LstmCandidateRetryConfig | None = None,
    deps: LstmCandidateRetryDependencies | None = None,
) -> dict[str, Any]:
    cfg = validated_lstm_candidate_retry_config(config or LstmCandidateRetryConfig())
    active_deps = deps or default_lstm_candidate_retry_dependencies()
    results = [
        _retry_symbol_duration(symbol, duration, cfg, active_deps)
        for symbol in cfg.symbols
        for duration in cfg.durations
    ]
    return {
        "status": _summary_status(results),
        "runAt": _utc_now(),
        "profile": cfg.profile,
        "results": results,
    }


def validated_lstm_candidate_retry_config(config: LstmCandidateRetryConfig) -> LstmCandidateRetryConfig:
    symbols = _normalized_symbols(config.symbols or tuple(factor_ranking_precomputed_symbols()))
    durations = _validated_durations(config.durations)
    profile = normalize_experiment_profile(config.profile)
    return LstmCandidateRetryConfig(
        symbols=symbols,
        durations=durations,
        profile=profile,
        search=config.search,
        manual_trigger=config.manual_trigger,
        reset_history=config.reset_history,
    )


def default_lstm_candidate_retry_dependencies() -> LstmCandidateRetryDependencies:
    return LstmCandidateRetryDependencies(
        lstm_status=lstm_model_status,
        refresh_klines=refresh_prediction_klines,
        run_combination_ranking=run_factor_combination_ranking,
        save_combination_ranking=save_cached_combination_ranking,
        promote_combinations=upsert_good_combinations,
        train_lstm=_train_search_candidate,
    )


def _retry_symbol_duration(
    symbol: str,
    duration: str,
    config: LstmCandidateRetryConfig,
    deps: LstmCandidateRetryDependencies,
) -> dict[str, Any]:
    status = deps.lstm_status(symbol, duration)
    decision = _retry_decision(status, manual_trigger=config.manual_trigger)
    if not decision["shouldTrain"]:
        return _skipped_result(symbol, duration, status, decision)
    configs = _next_search_configs(symbol, duration, config, deps)
    if not configs:
        return _search_exhausted_result(symbol, duration, status, decision, config)
    search_total = search_space_size(config.search)
    deps.start_progress(
        symbol=symbol,
        duration=duration,
        profile=config.profile,
        total=len(configs),
        search_space_total=search_total,
        parallel_workers=config.search.parallel_workers,
    )
    try:
        _refresh_inputs(symbol, duration, deps)
        ranking = deps.run_combination_ranking(symbol, duration, combination_search_config_for_profile(config.profile))
        deps.save_combination_ranking(ranking)
        promotion = deps.promote_combinations(ranking)
        trainings = _train_candidates(configs, config.profile, config.search.parallel_workers, deps)
        if not config.manual_trigger:
            _publish_best_trade_candidate(trainings, deps)
        reports = [item.report for item in trainings]
        result = _trained_result(symbol, duration, status, decision, ranking, promotion, reports)
        deps.finish_progress(symbol=symbol, duration=duration, status=result["status"])
        return result
    except Exception:
        deps.finish_progress(symbol=symbol, duration=duration, status="failed")
        raise


def _retry_decision(status: dict[str, Any], *, manual_trigger: bool = False) -> dict[str, Any]:
    if manual_trigger:
        return {"shouldTrain": True, "reason": "manual_candidate_search"}
    if str(status.get("lastAttemptStatus") or "") == "training":
        return {"shouldTrain": False, "reason": "training_in_progress"}
    active_status = str(status.get("activeModelStatus") or status.get("status") or "")
    if active_status == "shadow_active":
        return {"shouldTrain": True, "reason": "shadow_active_candidate_search"}
    if active_status in RETRY_TRAIN_STATUSES:
        return {"shouldTrain": True, "reason": f"model_status_{active_status}"}
    if not bool(status.get("comboSnapshotMatches")):
        return {"shouldTrain": True, "reason": "combo_snapshot_mismatch"}
    if not bool(status.get("artifactsReady")):
        return {"shouldTrain": True, "reason": "artifacts_incomplete"}
    if bool(status.get("shadowPredictionReady")) and bool(status.get("comboSnapshotMatches")):
        return {"shouldTrain": False, "reason": "active_model_ready"}
    return {"shouldTrain": False, "reason": "not_retryable"}


def _next_search_configs(
    symbol: str,
    duration: str,
    config: LstmCandidateRetryConfig,
    deps: LstmCandidateRetryDependencies,
) -> list[LstmTrainingConfig]:
    request = LstmCandidateSearchRequest(
        symbol=symbol,
        duration=duration,
        profile=config.profile,
        attempted_keys=frozenset() if config.reset_history else deps.attempted_keys(symbol, duration),
        search_config=config.search,
    )
    return next_candidate_configs(request)


def _train_candidates(
    configs: list[LstmTrainingConfig],
    profile: str,
    workers: int,
    deps: LstmCandidateRetryDependencies,
) -> list[CandidateTrainingResult]:
    trained = []
    for item in _train_candidate_reports(configs, profile, workers, deps):
        trained.append(item)
        deps.record_candidate(item.config, profile, item.report)
        deps.complete_progress(
            config=item.config,
            profile=profile,
            report=item.report,
            completed=len(trained),
            total=len(configs),
        )
    return trained


def _train_candidate_reports(
    configs: list[LstmTrainingConfig],
    profile: str,
    workers: int,
    deps: LstmCandidateRetryDependencies,
) -> Iterator[CandidateTrainingResult]:
    if workers <= 1 or len(configs) <= 1:
        for config in configs:
            yield _train_candidate(config, profile, deps)
        return
    max_workers = min(int(workers), len(configs))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_train_candidate, candidate, profile, deps) for candidate in configs]
        for future in as_completed(futures):
            yield future.result()


def _train_candidate(
    config: LstmTrainingConfig,
    profile: str,
    deps: LstmCandidateRetryDependencies,
) -> CandidateTrainingResult:
    try:
        report = deps.train_lstm(config)
    except Exception as exc:
        report = _failed_training_report(config, profile, exc)
    report = {**report, "searchKey": search_key_for_config(config, profile)}
    return CandidateTrainingResult(config, report)


def _publish_best_trade_candidate(
    trainings: list[CandidateTrainingResult],
    deps: LstmCandidateRetryDependencies,
) -> None:
    best = _best_trade_candidate(trainings)
    if best is not None:
        deps.publish_trade_candidate(best.config, best.report)


def _best_trade_candidate(trainings: list[CandidateTrainingResult]) -> CandidateTrainingResult | None:
    trade_ready = [item for item in trainings if str(item.report.get("status") or "") in {"trade_active", "trained"}]
    if not trade_ready:
        return None
    return max(trade_ready, key=lambda item: _trade_candidate_score(item.report))


def _trade_candidate_score(report: dict[str, Any]) -> tuple[float, float, int]:
    validation = report.get("validation") or {}
    test = report.get("test") or {}
    win_rate = min(float(validation.get("winRate") or 0.0), float(test.get("winRate") or 0.0))
    profit = min(float(validation.get("profitFactor") or 0.0), float(test.get("profitFactor") or 0.0))
    samples = int((report.get("sampleCounts") or {}).get("test") or 0)
    return win_rate, profit, samples


def _train_search_candidate(config: LstmTrainingConfig) -> dict[str, Any]:
    return train_lstm_model(
        config,
        publish_shadow_active=False,
        publish_trade_active=False,
        write_attempt=False,
    )


def _refresh_inputs(
    symbol: str,
    duration: str,
    deps: LstmCandidateRetryDependencies,
) -> None:
    entry_open_time = current_rule_entry_open_time_for_duration(duration)
    deps.refresh_klines(symbol, "1m", entry_open_time - MS_PER_MINUTE)
    deps.refresh_klines(symbol, duration, entry_open_time - _duration_ms(duration))

