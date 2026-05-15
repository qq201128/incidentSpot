from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from app.services.factor_mined_library import (
    MINED_FACTOR_SOURCE_FILE,
    mined_factor_definition,
    mined_factor_payload,
    mined_factor_rows,
    mined_factor_rows_for_duration,
)
from app.services.factor_registry import (
    FactorCategory,
    FactorDefinition,
    factor_payload,
    get_factor,
    list_factor_categories,
    list_factor_payloads,
)


def list_single_factor_payloads(category: str | None = None) -> list[dict[str, Any]]:
    return list_factor_payloads(category)


def list_combo_factor_payloads() -> list[dict[str, Any]]:
    rows = [row for row in _latest_mined_rows_by_name() if _is_combo_factor_row(row)]
    return [mined_factor_payload(row) for row in rows]


def list_single_factor_categories() -> list[dict[str, Any]]:
    return list_factor_categories()


def list_all_factor_payloads(category: str | None = None) -> list[dict[str, Any]]:
    combo_payloads = list_combo_factor_payloads() if _includes_combo_category(category) else []
    return [*list_single_factor_payloads(category), *combo_payloads]


def list_all_factor_categories() -> list[dict[str, Any]]:
    static = list_single_factor_categories()
    counts = Counter(str(row.get("category") or "performance") for row in list_combo_factor_rows())
    return [
        {**item, "count": int(item.get("count") or 0) + counts.get(item["key"], 0)}
        for item in static
    ]


def list_combo_factor_rows() -> list[dict[str, Any]]:
    return [deepcopy(row) for row in _latest_mined_rows_by_name() if _is_combo_factor_row(row)]


def get_factor_payload_by_name(name: str) -> dict[str, Any] | None:
    factor = get_factor(name)
    if factor is not None:
        return factor_payload(factor)
    row = _latest_mined_row_by_name(name)
    return mined_factor_payload(row) if row is not None else None


def factor_definition_for_backtest(name: str, symbol: str, duration: str) -> FactorDefinition:
    factor = get_factor(name)
    if factor is not None:
        return factor
    row = mined_factor_row_for_backtest(name, symbol, duration)
    if row is None:
        raise ValueError(f"unknown factor: {name}")
    return mined_factor_definition(row)


def mined_factor_row_for_backtest(
    name: str,
    symbol: str,
    duration: str,
) -> dict[str, Any] | None:
    rows = [
        row
        for row in mined_factor_rows_for_duration(symbol, duration)
        if str(row.get("factorName")) == name
    ]
    if len(rows) > 1:
        raise ValueError(f"duplicate mined factor for {symbol.upper()} {duration}: {name}")
    return deepcopy(rows[0]) if rows else None


def is_mined_factor_definition(factor: FactorDefinition) -> bool:
    return factor.source_file == MINED_FACTOR_SOURCE_FILE


def _latest_mined_rows_by_name() -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in mined_factor_rows():
        name = str(row.get("factorName") or "")
        if not name:
            continue
        current = latest.get(name)
        if current is None or _row_order_key(row) > _row_order_key(current):
            latest[name] = row
    return sorted(latest.values(), key=_payload_sort_key)


def _latest_mined_row_by_name(name: str) -> dict[str, Any] | None:
    for row in _latest_mined_rows_by_name():
        if str(row.get("factorName")) == name:
            return deepcopy(row)
    return None


def _includes_combo_category(category: str | None) -> bool:
    if category is None:
        return True
    return FactorCategory(category) == FactorCategory.PERFORMANCE


def _is_combo_factor_row(row: dict[str, Any]) -> bool:
    members = row.get("members")
    return isinstance(members, list) and len(members) >= 2


def _row_order_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("lastSeenAt") or row.get("firstSeenAt") or ""), int(row.get("promotionCount") or 0))


def _payload_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("category") or "performance"),
        str(row.get("duration") or ""),
        str(row.get("factorName") or ""),
    )
