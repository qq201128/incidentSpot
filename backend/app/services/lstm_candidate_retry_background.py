from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

from app.services.background_loop_status import (
    record_loop_failure,
    record_loop_start,
    record_loop_stopped,
    record_loop_success,
)
from app.services.experiment_profiles import normalize_experiment_profile
from app.services.lstm_candidate_search import LstmCandidateSearchConfig
from app.services.lstm_candidate_retry import LstmCandidateRetryConfig, validated_lstm_candidate_retry_config
from app.services.model_search_resource import ModelSearchResourceConfig, resource_payload, validated_resource_config
from app.services.model_search_untrained_enqueue import enqueue_untrained_model_search_jobs

logger = logging.getLogger("uvicorn.error")
LOOP_NAME = "lstm_candidate_retry"

DEFAULT_RETRY_INTERVAL_SECONDS = 1800
DEFAULT_RETRY_PROFILE = "full"
DEFAULT_RETRY_DURATIONS = "10m,60m"
DEFAULT_FEATURE_WINDOWS = "24,32,48,64,96"
DEFAULT_MIN_MOVE_BPS = "8,10,12,15,20"
DEFAULT_EPOCHS = "8,12,16"
DEFAULT_SEEDS = "20260513,20260519,20260601"
DEFAULT_CANDIDATES_PER_DURATION = "all"
DEFAULT_PARALLEL_WORKERS = 1


@dataclass(frozen=True)
class CandidateSearchSpace:
    feature_windows: tuple[int, ...]
    min_move_bps_values: tuple[float, ...]
    epoch_values: tuple[int, ...]
    seeds: tuple[int, ...]


def lstm_candidate_retry_enabled() -> bool:
    """LSTM_CANDIDATE_RETRY_ENABLED: 默认关闭；多策略/集成场景下通常无需后台网格搜索。"""
    raw = os.getenv("LSTM_CANDIDATE_RETRY_ENABLED", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _retry_interval_seconds() -> float:
    raw = os.getenv("LSTM_CANDIDATE_RETRY_INTERVAL_SECONDS", str(DEFAULT_RETRY_INTERVAL_SECONDS))
    value = float(raw)
    if value <= 0:
        raise ValueError("LSTM_CANDIDATE_RETRY_INTERVAL_SECONDS must be positive")
    return value


def _retry_profile() -> str:
    raw = os.getenv("LSTM_CANDIDATE_RETRY_PROFILE", DEFAULT_RETRY_PROFILE)
    return normalize_experiment_profile(raw)


def _retry_durations() -> tuple[str, ...]:
    raw = os.getenv("LSTM_CANDIDATE_RETRY_DURATIONS", DEFAULT_RETRY_DURATIONS)
    durations = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not durations:
        raise ValueError("LSTM_CANDIDATE_RETRY_DURATIONS must include at least one duration")
    return durations


def _retry_config() -> LstmCandidateRetryConfig:
    return LstmCandidateRetryConfig(
        durations=_retry_durations(),
        profile=_retry_profile(),
        search=_search_config(),
    )


def _search_config() -> LstmCandidateSearchConfig:
    space = CandidateSearchSpace(
        feature_windows=_int_tuple("LSTM_CANDIDATE_FEATURE_WINDOWS", DEFAULT_FEATURE_WINDOWS),
        min_move_bps_values=_float_tuple("LSTM_CANDIDATE_MIN_MOVE_BPS", DEFAULT_MIN_MOVE_BPS),
        epoch_values=_int_tuple("LSTM_CANDIDATE_EPOCHS", DEFAULT_EPOCHS),
        seeds=_int_tuple("LSTM_CANDIDATE_SEEDS", DEFAULT_SEEDS),
    )
    return LstmCandidateSearchConfig(
        feature_windows=space.feature_windows,
        min_move_bps_values=space.min_move_bps_values,
        epoch_values=space.epoch_values,
        seeds=space.seeds,
        candidates_per_duration=_candidates_per_duration(space),
        parallel_workers=_parallel_workers(),
    )


def _candidates_per_duration(space: CandidateSearchSpace) -> int:
    raw = os.getenv("LSTM_CANDIDATE_PER_DURATION", DEFAULT_CANDIDATES_PER_DURATION).strip().lower()
    if raw == "all":
        return _search_space_size(space)
    value = int(raw)
    if value <= 0:
        raise ValueError("LSTM_CANDIDATE_PER_DURATION must be positive or 'all'")
    return value


def _search_space_size(space: CandidateSearchSpace) -> int:
    return len(space.feature_windows) * len(space.min_move_bps_values) * len(space.epoch_values) * len(space.seeds)


def _parallel_workers() -> int:
    raw = os.getenv("LSTM_CANDIDATE_PARALLEL_WORKERS", str(DEFAULT_PARALLEL_WORKERS))
    value = int(raw)
    if value <= 0:
        raise ValueError("LSTM_CANDIDATE_PARALLEL_WORKERS must be positive")
    return value


def _int_tuple(env_name: str, default: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _csv_env(env_name, default))


def _float_tuple(env_name: str, default: str) -> tuple[float, ...]:
    return tuple(float(part) for part in _csv_env(env_name, default))


def _csv_env(env_name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(env_name, default)
    selected = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not selected:
        raise ValueError(f"{env_name} must include at least one value")
    return selected


async def _sleep_for(stop_event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except TimeoutError:
        return


async def lstm_candidate_retry_loop(stop_event: asyncio.Event) -> None:
    try:
        interval_seconds = _retry_interval_seconds()
        config = _retry_config()
    except Exception as exc:
        record_loop_failure(LOOP_NAME, exc, {"stage": "startup_config"})
        logger.exception("lstm candidate retry startup config failed")
        raise
    logger.info(
        "lstm candidate retry scheduled: intervalSeconds=%s firstRunDelaySeconds=%s "
        "profile=%s durations=%s search=%s",
        interval_seconds,
        interval_seconds,
        config.profile,
        list(config.durations),
        config.search,
    )
    record_loop_start(
        LOOP_NAME,
        {"intervalSeconds": interval_seconds, "profile": config.profile, "durations": list(config.durations)},
    )
    while not stop_event.is_set():
        await _sleep_for(stop_event, interval_seconds)
        if stop_event.is_set():
            record_loop_stopped(LOOP_NAME, "stop_before_retry")
            return
        await _run_retry_once(config)


async def _run_retry_once(config: LstmCandidateRetryConfig) -> None:
    try:
        queued = await asyncio.to_thread(_enqueue_lstm_candidate_jobs, config)
        logger.info("lstm candidate retry enqueued jobs: %s", queued)
        record_loop_success(LOOP_NAME, {"status": "queued", "jobCount": queued.get("total")})
    except Exception as exc:
        record_loop_failure(LOOP_NAME, exc, {"stage": "retry"})
        logger.exception("lstm candidate retry failed")


def _enqueue_lstm_candidate_jobs(config: LstmCandidateRetryConfig) -> dict:
    cfg = validated_lstm_candidate_retry_config(config)
    return enqueue_untrained_model_search_jobs(
        symbols=cfg.symbols,
        durations=cfg.durations,
        families=("lstm",),
        profile=cfg.profile,
        reset_existing=False,
        reset_history=False,
        resource=resource_payload(
            validated_resource_config(
                ModelSearchResourceConfig(parallel_workers=cfg.search.parallel_workers)
            )
        ),
    )
