from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.factor_mined_library import mined_factor_rows_for_duration
from app.services.factor_registry import get_factor

FACTOR_TIMING_KLINE_CLOSE = "kline_close"
FACTOR_TIMING_PASSED = "passed"
FACTOR_TIMING_NOT_KLINE_CLOSE = "factor_timing_not_kline_close"

_KLINE_CLOSE_CATEGORIES = frozenset(
    {
        "return",
        "volatility",
        "moving_average",
        "momentum",
        "volume",
        "structure",
        "multi_timeframe",
        "smc",
        "statistic",
    }
)


@dataclass(frozen=True)
class FactorSignalTiming:
    mode: str
    passed: bool
    reason: str
    eligible_members: tuple[str, ...]
    blocked_members: tuple[str, ...]


def combination_kline_close_timing(
    row: dict[str, Any],
    *,
    symbol: str,
    duration: str,
    mined_by_name: dict[str, dict[str, Any]] | None = None,
) -> FactorSignalTiming:
    eligible: list[str] = []
    blocked: list[str] = []
    mined_rows = mined_by_name if mined_by_name is not None else _mined_rows_by_name(symbol, duration)
    for member in _members(row):
        name = _member_name(member)
        is_kline_close = _member_is_kline_close(member, mined_rows, visiting=frozenset())
        target = eligible if is_kline_close else blocked
        target.append(name)
    return _timing_result(eligible, blocked)


def _member_is_kline_close(
    member: dict[str, Any],
    mined_by_name: dict[str, dict[str, Any]],
    *,
    visiting: frozenset[str],
) -> bool:
    name = _member_name(member)
    factor = get_factor(name)
    if factor is not None:
        if factor.category.value == "performance":
            return factor.source_file != "mined_factor_library.json"
        return factor.category.value in _KLINE_CLOSE_CATEGORIES
    mined_row = mined_by_name.get(name)
    if mined_row is not None:
        return _mined_row_is_kline_close(mined_row, mined_by_name, visiting=visiting)
    return str(member.get("category") or "") in _KLINE_CLOSE_CATEGORIES


def _mined_row_is_kline_close(
    row: dict[str, Any],
    mined_by_name: dict[str, dict[str, Any]],
    *,
    visiting: frozenset[str],
) -> bool:
    name = str(row.get("factorName") or "")
    if name in visiting:
        raise ValueError(f"cycle in mined factor timing: {name}")
    next_visiting = visiting | frozenset({name})
    return all(
        _member_is_kline_close(member, mined_by_name, visiting=next_visiting)
        for member in _members(row)
    )


def _mined_rows_by_name(symbol: str, duration: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("factorName")): row for row in mined_factor_rows_for_duration(symbol, duration)}


def _timing_result(eligible: list[str], blocked: list[str]) -> FactorSignalTiming:
    if blocked:
        return FactorSignalTiming(
            mode=FACTOR_TIMING_KLINE_CLOSE,
            passed=False,
            reason=FACTOR_TIMING_NOT_KLINE_CLOSE,
            eligible_members=tuple(eligible),
            blocked_members=tuple(blocked),
        )
    return FactorSignalTiming(
        mode=FACTOR_TIMING_KLINE_CLOSE,
        passed=True,
        reason=FACTOR_TIMING_PASSED,
        eligible_members=tuple(eligible),
        blocked_members=(),
    )


def _members(row: dict[str, Any]) -> list[dict[str, Any]]:
    members = row.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"combination row missing members: {row.get('factorName')}")
    return [dict(member) for member in members]


def _member_name(member: dict[str, Any]) -> str:
    return str(member["name"])
