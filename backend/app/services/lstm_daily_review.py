from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from app.services.factor_backtest_batch_service import BACKTEST_DURATION_ORDER
from app.services.factor_combination_cache_service import save_cached_combination_ranking
from app.services.factor_combination_service import run_factor_combination_ranking
from app.services.factor_learning_service import refresh_factor_learning_memory
from app.services.factor_mined_library import upsert_good_combinations
from app.services.factor_ranking_cache_service import factor_ranking_precomputed_symbols
from app.services.experiment_profiles import (
    EXPERIMENT_PROFILE_FULL,
    ShadowGateResult,
    combination_search_config_for_profile,
    lstm_training_config_for_profile,
    normalize_experiment_profile,
    shadow_gate_for_full_profile,
)
from app.services.kline_backfill import count_klines, oldest_open_time, upsert_klines_rows
from app.services.lstm_config import LstmTrainingConfig
from app.services.lstm_training_service import train_lstm_model
from app.services.market_context_ingest_service import (
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_CONTEXT_PERIOD,
    ingest_market_context_data,
)
from app.services.binance_service import fetch_klines
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

BINANCE_KLINE_LIMIT_MAX = 1000
DEFAULT_TARGET_KLINE_ROWS = 20_000
DEFAULT_BACKFILL_CHUNK = 1000
DEFAULT_MAX_BACKFILL_ROUNDS = 200


@dataclass(frozen=True)
class LstmDailyReviewConfig:
    symbols: tuple[str, ...] = ()
    durations: tuple[str, ...] = BACKTEST_DURATION_ORDER
    experiment_profile: str = EXPERIMENT_PROFILE_FULL
    kline_limit: int = BINANCE_KLINE_LIMIT_MAX
    target_kline_rows: int = DEFAULT_TARGET_KLINE_ROWS
    backfill_chunk: int = DEFAULT_BACKFILL_CHUNK
    max_backfill_rounds: int = DEFAULT_MAX_BACKFILL_ROUNDS
    context_period: str = DEFAULT_CONTEXT_PERIOD
    context_limit: int = DEFAULT_CONTEXT_LIMIT
    run_llm_agent: bool = False


@dataclass(frozen=True)
class LstmDailyReviewDependencies:
    fetch_klines: Callable[..., list[dict[str, Any]]]
    upsert_klines: Callable[[str, str, list[dict[str, Any]]], None]
    count_klines: Callable[[str, str], int]
    oldest_open_time: Callable[[str, str], int | None]
    ingest_market_context: Callable[..., dict[str, Any]]
    run_combination_ranking: Callable[..., dict[str, Any]]
    save_combination_ranking: Callable[[dict[str, Any]], None]
    promote_combinations: Callable[[dict[str, Any]], dict[str, Any]]
    train_lstm: Callable[[LstmTrainingConfig], dict[str, Any]]
    refresh_learning_memory: Callable[..., dict[str, Any]]


def run_lstm_daily_review(
    config: LstmDailyReviewConfig | None = None,
    deps: LstmDailyReviewDependencies | None = None,
) -> dict[str, Any]:
    cfg = validated_lstm_daily_review_config(config or LstmDailyReviewConfig())
    active_deps = deps or default_lstm_daily_review_dependencies()
    symbols = [_run_symbol_review(symbol, cfg, active_deps) for symbol in cfg.symbols]
    status = "blocked" if any(_symbol_report_blocked(report) for report in symbols) else "completed"
    return {
        "status": status,
        "runAt": _utc_now(),
        "symbols": symbols,
        "durations": list(cfg.durations),
        "runLlmAgent": cfg.run_llm_agent and cfg.experiment_profile == EXPERIMENT_PROFILE_FULL,
        "experimentProfile": cfg.experiment_profile,
    }


def validated_lstm_daily_review_config(config: LstmDailyReviewConfig) -> LstmDailyReviewConfig:
    symbols = _normalized_symbols(config.symbols or tuple(factor_ranking_precomputed_symbols()))
    durations = _validated_durations(config.durations)
    profile = normalize_experiment_profile(config.experiment_profile)
    _validate_positive("kline_limit", config.kline_limit)
    _validate_positive("target_kline_rows", config.target_kline_rows)
    _validate_positive("backfill_chunk", config.backfill_chunk)
    _validate_positive("max_backfill_rounds", config.max_backfill_rounds)
    if config.kline_limit > BINANCE_KLINE_LIMIT_MAX or config.backfill_chunk > BINANCE_KLINE_LIMIT_MAX:
        raise ValueError("Binance kline request limit must be <= 1000")
    return LstmDailyReviewConfig(**{**config.__dict__, "symbols": symbols, "durations": durations, "experiment_profile": profile})


def default_lstm_daily_review_dependencies() -> LstmDailyReviewDependencies:
    return LstmDailyReviewDependencies(
        fetch_klines=fetch_klines,
        upsert_klines=upsert_klines_rows,
        count_klines=count_klines,
        oldest_open_time=oldest_open_time,
        ingest_market_context=ingest_market_context_data,
        run_combination_ranking=run_factor_combination_ranking,
        save_combination_ranking=save_cached_combination_ranking,
        promote_combinations=upsert_good_combinations,
        train_lstm=train_lstm_model,
        refresh_learning_memory=refresh_factor_learning_memory,
    )


def _run_symbol_review(
    symbol: str,
    config: LstmDailyReviewConfig,
    deps: LstmDailyReviewDependencies,
) -> dict[str, Any]:
    duration_reports = [_run_duration_review(symbol, duration, config, deps) for duration in config.durations]
    if config.experiment_profile == EXPERIMENT_PROFILE_FULL and any(_blocked_duration(report) for report in duration_reports):
        return {
            "symbol": symbol,
            "profile": config.experiment_profile,
            "status": "blocked",
            "durations": duration_reports,
        }
    data_report = _collect_symbol_data(symbol, config, deps)
    return {"symbol": symbol, "profile": config.experiment_profile, "data": data_report, "durations": duration_reports}


def _collect_symbol_data(
    symbol: str,
    config: LstmDailyReviewConfig,
    deps: LstmDailyReviewDependencies,
) -> dict[str, Any]:
    latest = _refresh_latest_klines(symbol, config, deps)
    backfill = _backfill_history_strict(symbol, config, deps)
    context = deps.ingest_market_context(symbol, period=config.context_period, limit=config.context_limit)
    return {"latestKlines": latest, "backfill": backfill, "marketContext": context}


def _refresh_latest_klines(
    symbol: str,
    config: LstmDailyReviewConfig,
    deps: LstmDailyReviewDependencies,
) -> dict[str, int]:
    before = deps.count_klines(symbol, "1m")
    rows = deps.fetch_klines(symbol, "1m", limit=config.kline_limit)
    if not rows:
        raise RuntimeError(f"no latest 1m klines fetched for {symbol}")
    deps.upsert_klines(symbol, "1m", rows)
    return {"beforeRows": before, "afterRows": deps.count_klines(symbol, "1m"), "fetchedRows": len(rows)}


def _backfill_history_strict(
    symbol: str,
    config: LstmDailyReviewConfig,
    deps: LstmDailyReviewDependencies,
) -> dict[str, int]:
    before = deps.count_klines(symbol, "1m")
    end_time = _initial_backfill_end_time(symbol, deps)
    rounds = 0
    while deps.count_klines(symbol, "1m") < config.target_kline_rows:
        if rounds >= config.max_backfill_rounds:
            break
        rounds += 1
        end_time = _backfill_once(symbol, end_time, config, deps)
    after = deps.count_klines(symbol, "1m")
    return {"beforeRows": before, "afterRows": after, "targetRows": config.target_kline_rows, "rounds": rounds}


def _run_duration_review(
    symbol: str,
    duration: str,
    config: LstmDailyReviewConfig,
    deps: LstmDailyReviewDependencies,
) -> dict[str, Any]:
    profile = config.experiment_profile
    if profile == EXPERIMENT_PROFILE_FULL:
        gate = shadow_gate_for_full_profile(symbol, duration)
        if not gate.ready:
            return _blocked_duration_report(duration, gate, profile)
    ranking = deps.run_combination_ranking(symbol, duration, combination_search_config_for_profile(profile))
    deps.save_combination_ranking(ranking)
    promotion = deps.promote_combinations(ranking)
    training = deps.train_lstm(_training_config(symbol, duration, config))
    learning = deps.refresh_learning_memory(
        symbol,
        duration,
        ranking_report=ranking,
        run_llm_agent=config.run_llm_agent and profile == EXPERIMENT_PROFILE_FULL,
    )
    return _duration_report(duration, profile, ranking, promotion, training, learning)


def _backfill_once(
    symbol: str,
    end_time: int | None,
    config: LstmDailyReviewConfig,
    deps: LstmDailyReviewDependencies,
) -> int:
    rows = deps.fetch_klines(symbol, "1m", limit=config.backfill_chunk, end_time=end_time)
    if not rows:
        raise RuntimeError(f"no historical 1m klines fetched for {symbol} before {end_time}")
    deps.upsert_klines(symbol, "1m", rows)
    new_oldest = min(int(row["openTime"]) for row in rows)
    if end_time is not None and new_oldest >= end_time:
        raise RuntimeError(f"historical kline backfill did not move earlier for {symbol}")
    return new_oldest - 1


def _initial_backfill_end_time(symbol: str, deps: LstmDailyReviewDependencies) -> int | None:
    oldest = deps.oldest_open_time(symbol, "1m")
    return int(oldest) - 1 if oldest is not None else None


def _training_config(symbol: str, duration: str, config: LstmDailyReviewConfig) -> LstmTrainingConfig:
    return lstm_training_config_for_profile(symbol, duration, config.experiment_profile)


def _duration_report(
    duration: str,
    profile: str,
    ranking: dict[str, Any],
    promotion: dict[str, Any],
    training: dict[str, Any],
    learning: dict[str, Any],
) -> dict[str, Any]:
    return {
        "duration": duration,
        "profile": profile,
        "rankingTotal": len(ranking.get("ranking") or []),
        "promotion": promotion,
        "training": _training_summary(training),
        "learning": _learning_summary(learning),
    }


def _blocked_duration_report(duration: str, gate: ShadowGateResult, profile: str) -> dict[str, Any]:
    return {
        "duration": duration,
        "profile": profile,
        "status": "blocked",
        "reason": gate.reason,
        "shadowGate": {"ready": gate.ready, **gate.diagnostics, "reason": gate.reason},
        "rankingTotal": 0,
        "promotion": {"promoted": 0, "libraryTotal": 0},
        "training": {"status": "skipped"},
        "learning": {"status": "skipped"},
    }


def _blocked_duration(report: dict[str, Any]) -> bool:
    return str(report.get("status") or "") == "blocked"


def _symbol_report_blocked(report: dict[str, Any]) -> bool:
    return str(report.get("status") or "") == "blocked"


def _training_summary(report: dict[str, Any]) -> dict[str, Any]:
    keys = ("status", "modelVersion", "trainedAt", "sampleCounts", "validation", "test")
    return {key: report.get(key) for key in keys if key in report}


def _learning_summary(memory: dict[str, Any]) -> dict[str, Any]:
    agent = memory.get("llmAgent") if isinstance(memory.get("llmAgent"), dict) else {}
    return {
        "version": memory.get("version"),
        "symbol": memory.get("symbol"),
        "duration": memory.get("duration"),
        "updatedAt": memory.get("updatedAt"),
        "llmAgentStatus": agent.get("status"),
    }


def _normalized_symbols(symbols: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(symbol.strip().upper() for symbol in symbols if symbol.strip())
    if not normalized:
        raise ValueError("at least one symbol is required")
    return normalized


def _validated_durations(durations: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(duration.strip() for duration in durations if duration.strip())
    unsupported = [duration for duration in selected if duration not in SUPPORTED_RULE_DURATIONS]
    if not selected:
        raise ValueError("at least one duration is required")
    if unsupported:
        raise ValueError(f"unsupported durations: {', '.join(unsupported)}")
    return selected


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
