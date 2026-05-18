from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from app.services.experiment_profiles import (
    EXPERIMENT_PROFILE_FAST,
    combination_search_config_for_profile,
    lstm_training_config_for_profile,
    normalize_experiment_profile,
)
from app.services.factor_combination_cache_service import save_cached_combination_ranking
from app.services.factor_combination_service import run_factor_combination_ranking
from app.services.factor_mined_library import upsert_good_combinations
from app.services.factor_ranking_cache_service import factor_ranking_precomputed_symbols
from app.services.kline_prediction_refresh import refresh_prediction_klines
from app.services.kline_timing import MS_PER_MINUTE, current_rule_entry_open_time_for_duration
from app.services.lstm_config import LstmTrainingConfig
from app.services.lstm_prediction_service import lstm_model_status
from app.services.lstm_training_service import train_lstm_model
from app.services.rule_config import DURATION_TO_MINUTES, SUPPORTED_RULE_DURATIONS

DEFAULT_RETRY_DURATIONS = ("10m",)
RETRY_TRAIN_STATUSES = {"untrained", "validation_failed", "failed", "insufficient_samples"}


@dataclass(frozen=True)
class LstmCandidateRetryConfig:
    symbols: tuple[str, ...] = ()
    durations: tuple[str, ...] = DEFAULT_RETRY_DURATIONS
    profile: str = EXPERIMENT_PROFILE_FAST


@dataclass(frozen=True)
class LstmCandidateRetryDependencies:
    lstm_status: Callable[..., dict[str, Any]]
    refresh_klines: Callable[[str, str, int], None]
    run_combination_ranking: Callable[..., dict[str, Any]]
    save_combination_ranking: Callable[[dict[str, Any]], None]
    promote_combinations: Callable[[dict[str, Any]], dict[str, Any]]
    train_lstm: Callable[[LstmTrainingConfig], dict[str, Any]]


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
    return LstmCandidateRetryConfig(symbols=symbols, durations=durations, profile=profile)


def default_lstm_candidate_retry_dependencies() -> LstmCandidateRetryDependencies:
    return LstmCandidateRetryDependencies(
        lstm_status=lstm_model_status,
        refresh_klines=refresh_prediction_klines,
        run_combination_ranking=run_factor_combination_ranking,
        save_combination_ranking=save_cached_combination_ranking,
        promote_combinations=upsert_good_combinations,
        train_lstm=train_lstm_model,
    )


def _retry_symbol_duration(
    symbol: str,
    duration: str,
    config: LstmCandidateRetryConfig,
    deps: LstmCandidateRetryDependencies,
) -> dict[str, Any]:
    status = deps.lstm_status(symbol, duration)
    decision = _retry_decision(status)
    if not decision["shouldTrain"]:
        return _skipped_result(symbol, duration, status, decision)
    _refresh_inputs(symbol, duration, deps)
    ranking = deps.run_combination_ranking(symbol, duration, combination_search_config_for_profile(config.profile))
    deps.save_combination_ranking(ranking)
    promotion = deps.promote_combinations(ranking)
    training = deps.train_lstm(lstm_training_config_for_profile(symbol, duration, config.profile))
    return _trained_result(symbol, duration, status, decision, ranking, promotion, training)


def _retry_decision(status: dict[str, Any]) -> dict[str, Any]:
    if str(status.get("lastAttemptStatus") or "") == "training":
        return {"shouldTrain": False, "reason": "training_in_progress"}
    if bool(status.get("shadowPredictionReady")) and bool(status.get("comboSnapshotMatches")):
        return {"shouldTrain": False, "reason": "active_model_ready"}
    active_status = str(status.get("activeModelStatus") or status.get("status") or "")
    if active_status in RETRY_TRAIN_STATUSES:
        return {"shouldTrain": True, "reason": f"model_status_{active_status}"}
    if not bool(status.get("comboSnapshotMatches")):
        return {"shouldTrain": True, "reason": "combo_snapshot_mismatch"}
    if not bool(status.get("artifactsReady")):
        return {"shouldTrain": True, "reason": "artifacts_incomplete"}
    return {"shouldTrain": False, "reason": "not_retryable"}


def _refresh_inputs(
    symbol: str,
    duration: str,
    deps: LstmCandidateRetryDependencies,
) -> None:
    entry_open_time = current_rule_entry_open_time_for_duration(duration)
    deps.refresh_klines(symbol, "1m", entry_open_time - MS_PER_MINUTE)
    deps.refresh_klines(symbol, duration, entry_open_time - _duration_ms(duration))


def _skipped_result(
    symbol: str,
    duration: str,
    status: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "duration": duration,
        "status": "skipped",
        "reason": decision["reason"],
        "activeModelStatus": status.get("activeModelStatus") or status.get("status"),
        "lastAttemptStatus": status.get("lastAttemptStatus"),
    }


def _trained_result(
    symbol: str,
    duration: str,
    status: dict[str, Any],
    decision: dict[str, Any],
    ranking: dict[str, Any],
    promotion: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "duration": duration,
        "status": str(training.get("status") or "trained"),
        "reason": decision["reason"],
        "previousActiveModelStatus": status.get("activeModelStatus") or status.get("status"),
        "rankingTotal": len(ranking.get("ranking") or []),
        "promotion": promotion,
        "training": _training_summary(training),
    }


def _training_summary(report: dict[str, Any]) -> dict[str, Any]:
    keys = ("status", "modelVersion", "trainedAt", "sampleCounts", "validationFailureReason")
    return {key: report.get(key) for key in keys if key in report}


def _summary_status(results: list[dict[str, Any]]) -> str:
    if any(result.get("status") not in {"skipped", "trained"} for result in results):
        return "completed_with_rejections"
    if any(result.get("status") == "trained" for result in results):
        return "trained"
    return "skipped"


def _normalized_symbols(symbols: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(symbol.strip().upper() for symbol in symbols if symbol.strip())
    if not normalized:
        raise ValueError("at least one LSTM retry symbol is required")
    return normalized


def _validated_durations(durations: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(duration.strip() for duration in durations if duration.strip())
    unsupported = [duration for duration in selected if duration not in SUPPORTED_RULE_DURATIONS]
    if not selected:
        raise ValueError("at least one LSTM retry duration is required")
    if unsupported:
        raise ValueError(f"unsupported LSTM retry durations: {', '.join(unsupported)}")
    return selected


def _duration_ms(duration: str) -> int:
    return int(DURATION_TO_MINUTES[duration]) * MS_PER_MINUTE


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
