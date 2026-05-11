from __future__ import annotations

from datetime import datetime, timezone

from app.services.rule_config import MS_PER_MINUTE, RULE_DURATION, RULE_HORIZON_MINUTES, horizon_minutes_for_duration

RULE_INTERVAL_MS = RULE_HORIZON_MINUTES * MS_PER_MINUTE
KLINE_ENTRY_GRACE_MS = 30_000

# 三连/四连/五连指数 10m：同一根 K 内 open_time 不变，仅放宽「开盘后 30 秒」会导致整段错过；
# 与图表对齐为「当前 10m 桶内任意时刻均可生成预测/下单」（形态仍相对该桶起点前的已收盘指数 K）。
N_BAR_10M_RM_ENTRY_GRACE_MS = 10 * MS_PER_MINUTE - 1


def rule_interval_ms_for_duration(duration: str) -> int:
    return horizon_minutes_for_duration(duration) * MS_PER_MINUTE


def current_rule_entry_open_time_for_duration(duration: str, now_ms: int | None = None) -> int:
    interval_ms = rule_interval_ms_for_duration(duration)
    timestamp = now_ms if now_ms is not None else utc_now_ms()
    return (int(timestamp) // interval_ms) * interval_ms


def next_rule_entry_open_time_for_duration(duration: str, now_ms: int | None = None) -> int:
    return current_rule_entry_open_time_for_duration(duration, now_ms) + rule_interval_ms_for_duration(duration)


def seconds_until_next_rule_entry_for_duration(duration: str, now_ms: int | None = None) -> float:
    timestamp = now_ms if now_ms is not None else utc_now_ms()
    nxt = next_rule_entry_open_time_for_duration(duration, timestamp)
    return max((nxt - int(timestamp)) / 1000.0, 0.0)


def current_rule_entry_open_time(now_ms: int | None = None) -> int:
    """当前 UTC 10m 桶起点（兼容旧逻辑）。"""
    return current_rule_entry_open_time_for_duration(RULE_DURATION, now_ms)


def next_rule_entry_open_time(now_ms: int | None = None) -> int:
    return next_rule_entry_open_time_for_duration(RULE_DURATION, now_ms)


def seconds_until_next_rule_entry(now_ms: int | None = None) -> float:
    return seconds_until_next_rule_entry_for_duration(RULE_DURATION, now_ms)


def is_rule_entry_boundary(open_time: int) -> bool:
    return int(open_time) % RULE_INTERVAL_MS == 0


def is_rule_entry_boundary_for_duration(open_time: int, duration: str) -> bool:
    interval_ms = rule_interval_ms_for_duration(duration)
    return int(open_time) % interval_ms == 0


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


def align_to_rule_interval_bucket(ts_ms: int) -> int:
    """Floor any instant to the UTC 10m bucket open (matches Binance index 10m ``openTime`` alignment)."""
    return align_to_rule_interval_bucket_for_duration(ts_ms, RULE_DURATION)


def align_to_rule_interval_bucket_for_duration(ts_ms: int, duration: str) -> int:
    interval_ms = rule_interval_ms_for_duration(duration)
    return (int(ts_ms) // interval_ms) * interval_ms
