from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from app.services import lstm_daily_review_background as lstm_bg
from app.services import lstm_candidate_retry_background as retry_bg

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
