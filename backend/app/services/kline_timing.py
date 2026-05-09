from __future__ import annotations

from datetime import datetime, timezone

from app.services.rule_config import MS_PER_MINUTE, RULE_HORIZON_MINUTES

RULE_INTERVAL_MS = RULE_HORIZON_MINUTES * MS_PER_MINUTE
KLINE_ENTRY_GRACE_MS = 30_000


def current_rule_entry_open_time(now_ms: int | None = None) -> int:
    timestamp = now_ms if now_ms is not None else utc_now_ms()
    return (int(timestamp) // RULE_INTERVAL_MS) * RULE_INTERVAL_MS


def next_rule_entry_open_time(now_ms: int | None = None) -> int:
    timestamp = now_ms if now_ms is not None else utc_now_ms()
    return current_rule_entry_open_time(timestamp) + RULE_INTERVAL_MS


def seconds_until_next_rule_entry(now_ms: int | None = None) -> float:
    timestamp = now_ms if now_ms is not None else utc_now_ms()
    return max((next_rule_entry_open_time(timestamp) - int(timestamp)) / 1000.0, 0.0)


def is_rule_entry_boundary(open_time: int) -> bool:
    return int(open_time) % RULE_INTERVAL_MS == 0


def is_within_entry_grace(
    open_time: int,
    now_ms: int | None = None,
    *,
    grace_ms: int = KLINE_ENTRY_GRACE_MS,
) -> bool:
    if grace_ms < 0:
        raise ValueError("grace_ms must be >= 0")
    timestamp = now_ms if now_ms is not None else utc_now_ms()
    age_ms = int(timestamp) - int(open_time)
    return 0 <= age_ms <= int(grace_ms)


def exit_open_time(entry_open_time: int) -> int:
    return int(entry_open_time) + RULE_INTERVAL_MS


def utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)
