from __future__ import annotations

import re
from typing import Any

from app.services.factor_registry import get_factor

COMBO_PREFIXES = ("combo__", "goal_combo__")


def combo_display_name(members: list[dict[str, Any]]) -> str:
    leaves = collect_leaf_factor_names(members)
    if not leaves:
        return "组合"
    labels = [short_factor_label(name) for name in leaves]
    return "组合：" + " + ".join(labels)


def collect_leaf_factor_names(members: list[dict[str, Any]]) -> list[str]:
    leaves: list[str] = []
    seen: set[str] = set()
    for member in members:
        if not isinstance(member, dict):
            continue
        nested = member.get("members")
        if isinstance(nested, list) and nested:
            for name in collect_leaf_factor_names(nested):
                _append_unique(leaves, seen, name)
            continue
        name = str(member.get("name") or "").strip()
        if not name:
            continue
        _append_factor_leaves(leaves, seen, name)
    return leaves


def parse_combo_member_names(combo_name: str) -> list[str]:
    if combo_name.startswith("goal_combo__"):
        segments = combo_name.split("__")
        members, _ = _parse_combo_segments(segments, 2)
        return members
    if combo_name.startswith("combo__"):
        segments = combo_name.split("__")
        members, _ = _parse_combo_segments(segments, 1)
        return members
    return [combo_name] if combo_name else []


def short_factor_label(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return "未知"
    for pattern, renderer in _NAME_LABEL_RULES:
        match = pattern.match(raw)
        if match:
            return renderer(match)
    factor = get_factor(raw)
    if factor is not None:
        shortened = _shorten_description(str(factor.description or ""))
        if shortened:
            return shortened
    return _fallback_label(raw)


def _parse_combo_segments(segments: list[str], index: int) -> tuple[list[str], int]:
    members: list[str] = []
    cursor = index
    while cursor < len(segments):
        member, cursor = _parse_combo_segment_member(segments, cursor)
        if member:
            members.append(member)
    return members, cursor


def _parse_combo_segment_member(segments: list[str], index: int) -> tuple[str, int]:
    if index >= len(segments):
        return "", index
    if segments[index] != "combo":
        return segments[index], index + 1
    for end in range(len(segments), index, -1):
        candidate = "__".join(segments[index:end])
        if not candidate.startswith("combo__"):
            continue
        parsed = parse_combo_member_names(candidate)
        if parsed and _combo_name_from_members(parsed) == candidate:
            return candidate, end
    return segments[index], index + 1


def _combo_name_from_members(members: list[str]) -> str:
    return "combo__" + "__".join(members)


def _append_factor_leaves(leaves: list[str], seen: set[str], name: str) -> None:
    if not _is_combo_name(name):
        _append_unique(leaves, seen, name)
        return
    parsed = parse_combo_member_names(name)
    if parsed == [name]:
        _append_unique(leaves, seen, name)
        return
    for parsed_name in parsed:
        _append_factor_leaves(leaves, seen, parsed_name)


def _is_combo_name(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in COMBO_PREFIXES)


def _append_unique(leaves: list[str], seen: set[str], name: str) -> None:
    if name in seen:
        return
    seen.add(name)
    leaves.append(name)


def _shorten_description(description: str) -> str:
    text = description.strip()
    if not text:
        return ""
    rules = (
        (re.compile(r"^ADX趋势强度（(\d+)周期）$"), r"ADX(\1)"),
        (re.compile(r"^(\d+)周期均线偏离$"), r"均线偏离(\1)"),
        (re.compile(r"^(\d+)周期盈亏比$"), r"盈亏比(\1)"),
        (re.compile(r"^(\d+)周期滚动夏普$"), r"滚动夏普(\1)"),
        (re.compile(r"^(\d+)周期上涨胜率$"), r"胜率(\1)"),
        (re.compile(r"^MACD信号线$"), "MACD"),
        (re.compile(r"^MACD线$"), "MACD"),
        (re.compile(r"^MACD直方图$"), "MACD柱"),
        (re.compile(r"^(.+?)（(\d+)周期）$"), r"\1(\2)"),
    )
    for pattern, replacement in rules:
        updated, count = pattern.subn(replacement, text)
        if count:
            return updated
    return text


def _fallback_label(name: str) -> str:
    if name.startswith("agent__"):
        return "Agent因子"
    return name.replace("_", " ")


def _label_adx(match: re.Match[str]) -> str:
    return f"ADX({match.group(1)})"


def _label_ma_ratio(match: re.Match[str]) -> str:
    return f"均线偏离({match.group(1)})"


def _label_profit_factor(match: re.Match[str]) -> str:
    return f"盈亏比({match.group(1)})"


def _label_rolling_sharpe(match: re.Match[str]) -> str:
    return f"滚动夏普({match.group(1)})"


def _label_win_rate(match: re.Match[str]) -> str:
    return f"胜率({match.group(1)})"


def _label_macd_signal(_match: re.Match[str]) -> str:
    return "MACD"


def _label_macd(_match: re.Match[str]) -> str:
    return "MACD"


def _label_atr(match: re.Match[str]) -> str:
    return f"ATR({match.group(1)})"


def _label_ret(match: re.Match[str]) -> str:
    return f"收益({match.group(1)})"


def _label_rsi(match: re.Match[str]) -> str:
    return f"RSI({match.group(1)})"


_NAME_LABEL_RULES: tuple[tuple[re.Pattern[str], Any], ...] = (
    (re.compile(r"^adx_(\d+)$"), _label_adx),
    (re.compile(r"^ma_ratio_(\d+)$"), _label_ma_ratio),
    (re.compile(r"^profit_factor_(\d+)$"), _label_profit_factor),
    (re.compile(r"^rolling_sharpe_(\d+)$"), _label_rolling_sharpe),
    (re.compile(r"^win_rate_(\d+)$"), _label_win_rate),
    (re.compile(r"^macd_signal$"), _label_macd_signal),
    (re.compile(r"^macd$"), _label_macd),
    (re.compile(r"^atr_(\d+)$"), _label_atr),
    (re.compile(r"^ret_(\d+)$"), _label_ret),
    (re.compile(r"^rsi_(\d+)$"), _label_rsi),
)
