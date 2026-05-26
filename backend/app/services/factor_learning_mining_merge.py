from __future__ import annotations

from typing import Any

from app.services.factor_operator_library import factor_operator_summary

COMBO_FACTOR_PREFIXES = ("combo__", "goal_combo__")


def factor_mining_payload(
    *,
    previous_memory: dict[str, Any] | None,
    current_success: list[dict[str, Any]],
    current_forbidden: list[dict[str, Any]],
    now: str,
) -> dict[str, Any]:
    return {
        "operatorLibrary": factor_operator_summary(),
        "successPatterns": _merge_success_patterns(previous_memory, current_success, now),
        "forbiddenRegions": _merge_forbidden_regions(previous_memory, current_forbidden, now),
    }


def _merge_success_patterns(
    previous_memory: dict[str, Any] | None,
    current: list[dict[str, Any]],
    now: str,
) -> list[dict[str, Any]]:
    previous = _previous_factor_mining_items(previous_memory, "successPatterns")
    by_key = {_success_pattern_key(item): dict(item) for item in previous if _success_pattern_key(item)}
    for item in current:
        key = _success_pattern_key(item)
        if key:
            by_key[key] = _merge_success_pattern(by_key.get(key), item, now)
    return sorted(by_key.values(), key=_success_sort_key, reverse=True)


def _merge_success_pattern(previous: dict[str, Any] | None, current: dict[str, Any], now: str) -> dict[str, Any]:
    if previous is None:
        return {**current, "firstSeenAt": now, "lastSeenAt": now}
    previous_support = int(previous.get("support") or 0)
    current_support = int(current.get("support") or 0)
    support = previous_support + current_support
    return {
        **previous,
        **current,
        "support": support,
        "score": _weighted_average(
            previous_value=previous.get("score"),
            previous_weight=previous_support,
            current_value=current.get("score"),
            current_weight=current_support,
        ),
        "factors": _merged_list(previous.get("factors"), current.get("factors")),
        "firstSeenAt": previous.get("firstSeenAt") or now,
        "lastSeenAt": now,
    }


def _merge_forbidden_regions(
    previous_memory: dict[str, Any] | None,
    current: list[dict[str, Any]],
    now: str,
) -> list[dict[str, Any]]:
    previous = _clean_forbidden_regions(_previous_factor_mining_items(previous_memory, "forbiddenRegions"))
    current = _clean_forbidden_regions(current)
    by_key = {_forbidden_region_key(item): dict(item) for item in previous if _forbidden_region_key(item)}
    for item in current:
        key = _forbidden_region_key(item)
        if key:
            by_key[key] = _merge_forbidden_region(by_key.get(key), item, now)
    return sorted(by_key.values(), key=_forbidden_sort_key, reverse=True)


def _merge_forbidden_region(previous: dict[str, Any] | None, current: dict[str, Any], now: str) -> dict[str, Any]:
    if previous is None:
        return {**current, "firstSeenAt": now, "lastSeenAt": now}
    previous_support = int(previous.get("support") or 0)
    current_support = int(current.get("support") or 0)
    support = previous_support + current_support
    return {
        **previous,
        **current,
        "support": support,
        "avgAbsCorrelation": _weighted_average(
            previous_value=previous.get("avgAbsCorrelation"),
            previous_weight=previous_support,
            current_value=current.get("avgAbsCorrelation"),
            current_weight=current_support,
        ),
        "members": _merged_list(previous.get("members"), current.get("members")),
        "firstSeenAt": previous.get("firstSeenAt") or now,
        "lastSeenAt": now,
    }


def _previous_factor_mining_items(previous_memory: dict[str, Any] | None, key: str) -> list[dict[str, Any]]:
    if not isinstance(previous_memory, dict):
        return []
    factor_mining = previous_memory.get("factorMining")
    if not isinstance(factor_mining, dict):
        return []
    items = factor_mining.get(key)
    return [dict(item) for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _success_pattern_key(item: dict[str, Any]) -> str:
    return str(item.get("pattern") or "")


def _forbidden_region_key(item: dict[str, Any]) -> str:
    return str(item.get("region") or "")


def _clean_forbidden_regions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for item in items if (row := _clean_forbidden_region(item)) is not None]


def _clean_forbidden_region(item: dict[str, Any]) -> dict[str, Any] | None:
    key = _forbidden_region_key(item)
    if not key or _is_combo_factor_name(_forbidden_region_seed(key)):
        return None
    members = item.get("members")
    if not isinstance(members, list):
        return dict(item)
    filtered = [name for name in _member_names(members) if not _is_combo_factor_name(name)]
    if not filtered or (len(filtered) != len(members) and len(filtered) < 2):
        return None
    return {**item, "members": filtered, "support": _cleaned_support(item, filtered)}


def _forbidden_region_seed(key: str) -> str:
    prefix = "correlation_cluster:"
    return key[len(prefix):] if key.startswith(prefix) else key


def _member_names(members: list[Any]) -> list[str]:
    return [str(member or "").strip() for member in members if str(member or "").strip()]


def _is_combo_factor_name(name: str) -> bool:
    return name.startswith(COMBO_FACTOR_PREFIXES)


def _cleaned_support(item: dict[str, Any], members: list[str]) -> int:
    support = int(item.get("support") or 0)
    return min(support, len(members)) if support > 0 else len(members)


def _weighted_average(*, previous_value: Any, previous_weight: int, current_value: Any, current_weight: int) -> float:
    total = previous_weight + current_weight
    if total <= 0:
        return 0.0
    value = float(previous_value or 0.0) * previous_weight + float(current_value or 0.0) * current_weight
    return round(value / total, 4)


def _merged_list(previous: Any, current: Any) -> list[Any]:
    values = []
    for item in [*(previous or []), *(current or [])]:
        if item not in values:
            values.append(item)
    return values


def _success_sort_key(item: dict[str, Any]) -> tuple[float, int]:
    return float(item.get("score") or 0.0), int(item.get("support") or 0)


def _forbidden_sort_key(item: dict[str, Any]) -> tuple[float, int]:
    return float(item.get("avgAbsCorrelation") or 0.0), int(item.get("support") or 0)
