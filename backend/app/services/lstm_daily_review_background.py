from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from app.services.experiment_profiles import normalize_experiment_profile
from app.services.lstm_daily_review import LstmDailyReviewConfig, run_lstm_daily_review
from app.services.lstm_torch_backend import is_torch_available

logger = logging.getLogger("uvicorn.error")

_DEFAULT_TZ = "Asia/Shanghai"
_DEFAULT_AT = "02:00"
_DEFAULT_PROFILE = "full"


def lstm_daily_review_enabled() -> bool:
    """LSTM_DAILY_REVIEW_ENABLED: 1/true/yes/on 开启（默认开启），0/false/off 关闭。"""
    raw = os.getenv("LSTM_DAILY_REVIEW_ENABLED", "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _parse_daily_at(raw: str) -> time:
    parts = raw.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"LSTM_DAILY_REVIEW_AT must be HH:MM, got: {raw!r}")
    hour_s, minute_s = parts[0].strip(), parts[1].strip()
    hour, minute = int(hour_s), int(minute_s)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"LSTM_DAILY_REVIEW_AT out of range: {raw!r}")
    return time(hour=hour, minute=minute)


def _timezone() -> ZoneInfo:
    tz_name = os.getenv("LSTM_DAILY_REVIEW_TZ", _DEFAULT_TZ).strip() or _DEFAULT_TZ
    try:
        return ZoneInfo(tz_name)
    except Exception:
        logger.warning("invalid LSTM_DAILY_REVIEW_TZ=%r, using %s", tz_name, _DEFAULT_TZ)
        return ZoneInfo(_DEFAULT_TZ)


def _daily_time() -> time:
    raw = os.getenv("LSTM_DAILY_REVIEW_AT", _DEFAULT_AT).strip() or _DEFAULT_AT
    try:
        return _parse_daily_at(raw)
    except ValueError as exc:
        logger.warning("%s, using %s", exc, _DEFAULT_AT)
        return _parse_daily_at(_DEFAULT_AT)


def _llm_agent_enabled() -> bool:
    return os.getenv("LSTM_DAILY_REVIEW_LLM", "0").strip().lower() in ("1", "true", "yes", "on")


def _profile() -> str:
    raw = os.getenv("LSTM_DAILY_REVIEW_PROFILE", _DEFAULT_PROFILE)
    return normalize_experiment_profile(raw)


def seconds_until_next_lstm_daily_review(
    now: datetime | None = None,
    *,
    zone: ZoneInfo | None = None,
    daily_at: time | None = None,
) -> float:
    tz = zone or _timezone()
    at = daily_at if daily_at is not None else _daily_time()
    current = now or datetime.now(tz)
    current = _localized(current, tz)
    target = datetime.combine(current.date(), at, tz)
    if target <= current:
        target += timedelta(days=1)
    return (target - current).total_seconds()


def _localized(value: datetime, tz: ZoneInfo) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=tz)
    return value.astimezone(tz)


def _review_config() -> LstmDailyReviewConfig:
    return LstmDailyReviewConfig(
        experiment_profile=_profile(),
        run_llm_agent=_llm_agent_enabled(),
    )


async def _sleep_for(stop_event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except TimeoutError:
        return


async def lstm_daily_review_loop(stop_event: asyncio.Event) -> None:
    tz = _timezone()
    at = _daily_time()
    logger.info(
        "lstm daily review scheduled: tz=%s local_time=%02d:%02d torch_installed=%s llm_agent=%s profile=%s",
        str(tz),
        at.hour,
        at.minute,
        is_torch_available(),
        _llm_agent_enabled(),
        _profile(),
    )
    while not stop_event.is_set():
        await _sleep_for(stop_event, seconds_until_next_lstm_daily_review(zone=tz, daily_at=at))
        if stop_event.is_set():
            return
        if not is_torch_available():
            logger.warning(
                "lstm daily review skipped: PyTorch not available (pip install torch). "
                "Will retry next scheduled run."
            )
            continue
        try:
            report = await asyncio.to_thread(run_lstm_daily_review, _review_config())
            logger.info(
                "lstm daily review finished runAt=%s symbols=%s",
                report.get("runAt"),
                [item.get("symbol") for item in (report.get("symbols") or [])],
            )
        except Exception:
            logger.exception("lstm daily review batch failed")
