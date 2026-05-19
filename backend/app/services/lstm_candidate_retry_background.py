from __future__ import annotations

import asyncio
import logging
import os

from app.services.experiment_profiles import normalize_experiment_profile
from app.services.lstm_candidate_search_notification import notify_lstm_candidate_search_finished
from app.services.lstm_candidate_search import LstmCandidateSearchConfig
from app.services.lstm_candidate_retry import LstmCandidateRetryConfig, run_lstm_candidate_retry
from app.services.lstm_torch_backend import is_torch_available

logger = logging.getLogger("uvicorn.error")

DEFAULT_RETRY_INTERVAL_SECONDS = 1800
DEFAULT_RETRY_PROFILE = "full"
DEFAULT_RETRY_DURATIONS = "10m,30m,60m"
DEFAULT_FEATURE_WINDOWS = "24,32,48,64,96"
DEFAULT_MIN_MOVE_BPS = "8,10,12,15,20"
DEFAULT_EPOCHS = "8,12,16"
DEFAULT_SEEDS = "20260513,20260519,20260601"
DEFAULT_CANDIDATES_PER_DURATION = "all"
DEFAULT_PARALLEL_WORKERS = 2


def lstm_candidate_retry_enabled() -> bool:
    raw = os.getenv("LSTM_CANDIDATE_RETRY_ENABLED", "1").strip().lower()
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
    feature_windows = _int_tuple("LSTM_CANDIDATE_FEATURE_WINDOWS", DEFAULT_FEATURE_WINDOWS)
    min_move_bps_values = _float_tuple("LSTM_CANDIDATE_MIN_MOVE_BPS", DEFAULT_MIN_MOVE_BPS)
    epoch_values = _int_tuple("LSTM_CANDIDATE_EPOCHS", DEFAULT_EPOCHS)
    seeds = _int_tuple("LSTM_CANDIDATE_SEEDS", DEFAULT_SEEDS)
    return LstmCandidateSearchConfig(
        feature_windows=feature_windows,
        min_move_bps_values=min_move_bps_values,
        epoch_values=epoch_values,
        seeds=seeds,
        candidates_per_duration=_candidates_per_duration(
            feature_windows,
            min_move_bps_values,
            epoch_values,
            seeds,
        ),
        parallel_workers=_parallel_workers(),
    )


def _candidates_per_duration(
    feature_windows: tuple[int, ...],
    min_move_bps_values: tuple[float, ...],
    epoch_values: tuple[int, ...],
    seeds: tuple[int, ...],
) -> int:
    raw = os.getenv("LSTM_CANDIDATE_PER_DURATION", DEFAULT_CANDIDATES_PER_DURATION).strip().lower()
    if raw == "all":
        return _search_space_size(feature_windows, min_move_bps_values, epoch_values, seeds)
    value = int(raw)
    if value <= 0:
        raise ValueError("LSTM_CANDIDATE_PER_DURATION must be positive or 'all'")
    return value


def _search_space_size(
    feature_windows: tuple[int, ...],
    min_move_bps_values: tuple[float, ...],
    epoch_values: tuple[int, ...],
    seeds: tuple[int, ...],
) -> int:
    return len(feature_windows) * len(min_move_bps_values) * len(epoch_values) * len(seeds)


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
    interval_seconds = _retry_interval_seconds()
    config = _retry_config()
    logger.info(
        "lstm candidate retry scheduled: intervalSeconds=%s torch_installed=%s profile=%s durations=%s search=%s",
        interval_seconds,
        is_torch_available(),
        config.profile,
        list(config.durations),
        config.search,
    )
    while not stop_event.is_set():
        await _run_retry_once(config)
        await _sleep_for(stop_event, interval_seconds)


async def _run_retry_once(config: LstmCandidateRetryConfig) -> None:
    if not is_torch_available():
        logger.warning("lstm candidate retry skipped: PyTorch not available")
        return
    try:
        report = await asyncio.to_thread(run_lstm_candidate_retry, config)
        notification = await asyncio.to_thread(notify_lstm_candidate_search_finished, report)
        logger.info("lstm candidate retry finished status=%s results=%s", report.get("status"), report.get("results"))
        logger.info("lstm candidate retry notification: %s", notification)
    except Exception:
        logger.exception("lstm candidate retry failed")
