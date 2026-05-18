from __future__ import annotations

import asyncio
import logging
import os

from app.services.experiment_profiles import normalize_experiment_profile
from app.services.lstm_candidate_retry import LstmCandidateRetryConfig, run_lstm_candidate_retry
from app.services.lstm_torch_backend import is_torch_available

logger = logging.getLogger("uvicorn.error")

DEFAULT_RETRY_INTERVAL_SECONDS = 1800
DEFAULT_RETRY_PROFILE = "fast"
DEFAULT_RETRY_DURATIONS = "10m"


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
    return LstmCandidateRetryConfig(durations=_retry_durations(), profile=_retry_profile())


async def _sleep_for(stop_event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except TimeoutError:
        return


async def lstm_candidate_retry_loop(stop_event: asyncio.Event) -> None:
    interval_seconds = _retry_interval_seconds()
    config = _retry_config()
    logger.info(
        "lstm candidate retry scheduled: intervalSeconds=%s torch_installed=%s profile=%s durations=%s",
        interval_seconds,
        is_torch_available(),
        config.profile,
        list(config.durations),
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
        logger.info("lstm candidate retry finished status=%s results=%s", report.get("status"), report.get("results"))
    except Exception:
        logger.exception("lstm candidate retry failed")
