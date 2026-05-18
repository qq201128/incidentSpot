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
    FactorDirection,
    factor_payload,
    get_factor,
    list_factor_categories,
    list_factors,
    list_factor_payloads,
)

AGENT_FACTOR_SOURCE_FILE = "agent_mined_factor_library.json"


def list_single_factor_payloads(category: str | None = None) -> list[dict[str, Any]]:
    return [*list_factor_payloads(category), *list_agent_factor_payloads(category)]


def list_single_factor_definitions(
    category: str | None = None,
    *,
    symbol: str | None = None,
    duration: str | None = None,
) -> list[FactorDefinition]:
    cat = FactorCategory(category) if category else None
    agent = list_agent_factor_definitions(category, symbol=symbol, duration=duration)
    return [*list_factors(cat), *agent]


def list_combo_factor_payloads() -> list[dict[str, Any]]:
    rows = [row for row in _latest_mined_rows_by_name() if _is_combo_factor_row(row)]
    return [mined_factor_payload(row) for row in rows]


def list_single_factor_categories() -> list[dict[str, Any]]:
    counts = Counter(_agent_factor_category(row).value for row in _latest_agent_rows_by_name())
    return [
        {**item, "count": int(item.get("count") or 0) + counts.get(item["key"], 0)}
        for item in list_factor_categories()
    ]


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
    agent_row = _latest_agent_row_by_name(name)
    if agent_row is not None:
        return agent_factor_payload(agent_row)
    row = _latest_mined_row_by_name(name)
    return mined_factor_payload(row) if row is not None else None


def factor_definition_for_backtest(name: str, symbol: str, duration: str) -> FactorDefinition:
    factor = get_factor(name)
    if factor is not None:
        return factor
    agent_row = agent_factor_row_for_backtest(name, symbol, duration)
    if agent_row is not None:
        return agent_factor_definition(agent_row)
    row = mined_factor_row_for_backtest(name, symbol, duration)
    if row is None:
        raise ValueError(f"unknown factor: {name}")
    return mined_factor_definition(row)


def list_agent_factor_payloads(category: str | None = None) -> list[dict[str, Any]]:
    if not _includes_agent_category(category):
        return []
    return [agent_factor_payload(row) for row in _latest_agent_rows_by_name()]


def list_agent_factor_definitions(
    category: str | None = None,
    *,
    symbol: str | None = None,
    duration: str | None = None,
) -> list[FactorDefinition]:
    if not _includes_agent_category(category):
        return []
    if symbol is not None and duration is not None:
        rows = _agent_factor_rows_for_duration(symbol, duration)
        return [agent_factor_definition(row) for row in rows]
    return [agent_factor_definition(row) for row in _latest_agent_rows_by_name()]


def agent_factor_payload(row: dict[str, Any]) -> dict[str, Any]:
    factor = agent_factor_definition(row)
    payload = factor_payload(factor)
    metrics = row.get("metrics") or {}
    return {
        **payload,
        "symbol": _row_symbol(row),
        "duration": str(row.get("duration") or ""),
        "promotionCount": int(row.get("promotionCount") or 0),
        "candidateStatus": str(row.get("candidateStatus") or "unknown"),
        "qualityPassed": bool(row.get("qualityPassed")),
        "metrics": dict(metrics) if isinstance(metrics, dict) else {},
    }


def agent_factor_definition(row: dict[str, Any]) -> FactorDefinition:
    duration = str(row.get("duration") or "")
    return FactorDefinition(
        name=str(row["factorName"]),
        category=_agent_factor_category(row),
        description=str(row.get("factorDisplayName") or row.get("displayName") or row["factorName"]),
        formula=str(row["formula"]),
        source_file=AGENT_FACTOR_SOURCE_FILE,
        timeframes=(duration,) if duration else (),
        direction=FactorDirection.NEUTRAL,
        parameters={"symbol": _row_symbol(row), "duration": duration},
    )


def agent_factor_row_for_backtest(
    name: str,
    symbol: str,
    duration: str,
) -> dict[str, Any] | None:
    rows = [
        row
        for row in _agent_factor_rows_for_duration(symbol, duration)
        if str(row.get("factorName")) == name
    ]
    if len(rows) > 1:
        raise ValueError(f"duplicate agent factor for {symbol.upper()} {duration}: {name}")
    return deepcopy(rows[0]) if rows else None


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


def is_agent_factor_definition(factor: FactorDefinition) -> bool:
    return factor.source_file == AGENT_FACTOR_SOURCE_FILE


def _latest_agent_rows_by_name() -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in _agent_factor_rows():
        name = str(row.get("factorName") or "")
        if not name:
            continue
        current = latest.get(name)
        if current is None or _row_order_key(row) > _row_order_key(current):
            latest[name] = row
    return sorted(latest.values(), key=_payload_sort_key)


def _latest_agent_row_by_name(name: str) -> dict[str, Any] | None:
    for row in _latest_agent_rows_by_name():
        if str(row.get("factorName")) == name:
            return deepcopy(row)
    return None


def _agent_factor_rows() -> list[dict[str, Any]]:
    from app.services.agent_mined_factor_library import load_agent_factor_library

    return deepcopy(load_agent_factor_library().get("factors") or [])


def _agent_factor_rows_for_duration(symbol: str, duration: str) -> list[dict[str, Any]]:
    from app.services.agent_mined_factor_library import agent_factor_rows_for_duration

    return agent_factor_rows_for_duration(symbol, duration)


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


def _includes_agent_category(category: str | None) -> bool:
    if category is None:
        return True
    return FactorCategory(category) == FactorCategory.STATISTIC


def _agent_factor_category(_row: dict[str, Any]) -> FactorCategory:
    return FactorCategory.STATISTIC


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


def _row_symbol(row: dict[str, Any]) -> str:
    return str(row.get("symbol") or "").strip().upper()
