from __future__ import annotations

import asyncio
from datetime import datetime, time
from zoneinfo import ZoneInfo

from app.services import lstm_daily_review_background as lstm_bg
from app.services import lstm_candidate_retry_background as retry_bg
from app.services.background_loop_status import background_loop_statuses, reset_background_loop_statuses

SECONDS_PER_MINUTE = 60
SECONDS_PER_DAY = 86400


def test_seconds_until_next_targets_configured_clock() -> None:
    tz = ZoneInfo("Asia/Shanghai")
    at = time(hour=2, minute=0)
    before = datetime(2026, 5, 13, 1, 59, tzinfo=tz)
    exactly = datetime(2026, 5, 13, 2, 0, tzinfo=tz)
    assert lstm_bg.seconds_until_next_lstm_daily_review(before, zone=tz, daily_at=at) == SECONDS_PER_MINUTE
    assert lstm_bg.seconds_until_next_lstm_daily_review(exactly, zone=tz, daily_at=at) == SECONDS_PER_DAY


def test_lstm_candidate_retry_interval_rejects_non_positive_env(monkeypatch) -> None:
    monkeypatch.setenv("LSTM_CANDIDATE_RETRY_INTERVAL_SECONDS", "0")

    try:
        retry_bg._retry_interval_seconds()
    except ValueError as exc:
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("expected invalid retry interval to raise")


def test_lstm_candidate_retry_background_defaults_to_full_search(monkeypatch) -> None:
    monkeypatch.delenv("LSTM_CANDIDATE_FEATURE_WINDOWS", raising=False)
    monkeypatch.delenv("LSTM_CANDIDATE_MIN_MOVE_BPS", raising=False)
    monkeypatch.delenv("LSTM_CANDIDATE_EPOCHS", raising=False)
    monkeypatch.delenv("LSTM_CANDIDATE_SEEDS", raising=False)
    monkeypatch.delenv("LSTM_CANDIDATE_PER_DURATION", raising=False)

    config = retry_bg._search_config()

    assert config.candidates_per_duration == 225


def test_lstm_candidate_retry_background_accepts_limited_search_env(monkeypatch) -> None:
    monkeypatch.setenv("LSTM_CANDIDATE_PER_DURATION", "10")
    monkeypatch.setenv("LSTM_CANDIDATE_PARALLEL_WORKERS", "3")

    config = retry_bg._search_config()

    assert config.candidates_per_duration == 10
    assert config.parallel_workers == 3


def test_lstm_candidate_retry_background_defaults_to_10m_and_60m_with_parallel_1(monkeypatch) -> None:
    monkeypatch.delenv("LSTM_CANDIDATE_FEATURE_WINDOWS", raising=False)
    monkeypatch.delenv("LSTM_CANDIDATE_MIN_MOVE_BPS", raising=False)
    monkeypatch.delenv("LSTM_CANDIDATE_EPOCHS", raising=False)
    monkeypatch.delenv("LSTM_CANDIDATE_SEEDS", raising=False)
    monkeypatch.delenv("LSTM_CANDIDATE_PER_DURATION", raising=False)
    monkeypatch.delenv("LSTM_CANDIDATE_PARALLEL_WORKERS", raising=False)

    config = retry_bg._retry_config()

    assert config.durations == ("10m", "60m")
    assert config.search.parallel_workers == 1


def test_lstm_candidate_retry_loop_waits_before_first_retry(monkeypatch) -> None:
    calls = []

    async def fake_sleep(stop_event: asyncio.Event, seconds: float) -> None:
        calls.append(("sleep", seconds))
        stop_event.set()

    async def fake_run_retry_once(_config) -> None:
        calls.append(("run", None))

    monkeypatch.setattr(retry_bg, "_retry_interval_seconds", lambda: 30.0)
    monkeypatch.setattr(retry_bg, "_sleep_for", fake_sleep)
    monkeypatch.setattr(retry_bg, "_run_retry_once", fake_run_retry_once)

    asyncio.run(retry_bg.lstm_candidate_retry_loop(asyncio.Event()))

    assert calls == [("sleep", 30.0)]


def test_lstm_candidate_retry_startup_failure_is_recorded(monkeypatch) -> None:
    reset_background_loop_statuses()
    monkeypatch.setenv("LSTM_CANDIDATE_RETRY_INTERVAL_SECONDS", "bad")

    try:
        asyncio.run(retry_bg.lstm_candidate_retry_loop(asyncio.Event()))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid retry config was not exposed")

    status = background_loop_statuses()["lstm_candidate_retry"]
    assert status["status"] == "failed"
    assert status["lastFailureDetails"]["stage"] == "startup_config"


def test_lstm_candidate_retry_failure_is_recorded(monkeypatch) -> None:
    reset_background_loop_statuses()

    monkeypatch.setattr(
        retry_bg,
        "enqueue_untrained_model_search_jobs",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("enqueue failed")),
    )

    asyncio.run(retry_bg._run_retry_once(retry_bg._retry_config()))

    status = background_loop_statuses()["lstm_candidate_retry"]
    assert status["status"] == "failed"
    assert status["lastError"] == "enqueue failed"
    assert status["lastFailureDetails"]["stage"] == "retry"


def test_lstm_candidate_retry_once_enqueues_jobs(monkeypatch) -> None:
    reset_background_loop_statuses()
    calls = []
    monkeypatch.setattr(
        retry_bg,
        "enqueue_untrained_model_search_jobs",
        lambda **kwargs: calls.append(kwargs) or {"total": 2, "jobs": []},
    )

    asyncio.run(retry_bg._run_retry_once(retry_bg._retry_config()))

    assert calls[0]["families"] == ("lstm",)
    assert calls[0]["durations"] == ("10m", "60m")
    assert calls[0]["resource"]["parallelWorkers"] == 1
    assert background_loop_statuses()["lstm_candidate_retry"]["status"] == "passed"


def test_lstm_daily_review_invalid_time_is_recorded(monkeypatch) -> None:
    reset_background_loop_statuses()
    monkeypatch.setenv("LSTM_DAILY_REVIEW_AT", "bad")

    try:
        asyncio.run(lstm_bg.lstm_daily_review_loop(asyncio.Event()))
    except ValueError as exc:
        assert "LSTM_DAILY_REVIEW_AT must be HH:MM" in str(exc)
    else:
        raise AssertionError("invalid daily review time was not exposed")

    status = background_loop_statuses()["lstm_daily_review"]
    assert status["status"] == "failed"
    assert status["lastFailureDetails"]["stage"] == "startup_config"


def test_lstm_daily_review_failure_is_recorded(monkeypatch) -> None:
    reset_background_loop_statuses()

    monkeypatch.setattr(lstm_bg, "is_torch_available", lambda: True)
    monkeypatch.setattr(
        lstm_bg,
        "run_lstm_daily_review",
        lambda _config: (_ for _ in ()).throw(RuntimeError("review failed")),
    )

    async def run_once() -> None:
        stop = asyncio.Event()

        calls = {"count": 0}

        async def sleep_once(_stop_event: asyncio.Event, _seconds: float) -> None:
            calls["count"] += 1
            if calls["count"] > 1:
                _stop_event.set()

        monkeypatch.setattr(lstm_bg, "_sleep_for", sleep_once)
        await lstm_bg.lstm_daily_review_loop(stop)

    asyncio.run(run_once())

    status = background_loop_statuses()["lstm_daily_review"]
    assert status["status"] == "failed"
    assert status["lastError"] == "review failed"
    assert status["lastFailureDetails"]["stage"] == "review"
