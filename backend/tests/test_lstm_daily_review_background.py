from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from app.services import lstm_daily_review_background as lstm_bg

SECONDS_PER_MINUTE = 60
SECONDS_PER_DAY = 86400


def test_seconds_until_next_targets_configured_clock() -> None:
    tz = ZoneInfo("Asia/Shanghai")
    at = time(hour=2, minute=0)
    before = datetime(2026, 5, 13, 1, 59, tzinfo=tz)
    exactly = datetime(2026, 5, 13, 2, 0, tzinfo=tz)
    assert lstm_bg.seconds_until_next_lstm_daily_review(before, zone=tz, daily_at=at) == SECONDS_PER_MINUTE
    assert lstm_bg.seconds_until_next_lstm_daily_review(exactly, zone=tz, daily_at=at) == SECONDS_PER_DAY
