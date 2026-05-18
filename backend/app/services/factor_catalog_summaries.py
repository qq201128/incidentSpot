from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.services.agent_mined_factor_library import load_agent_factor_library
from app.services.factor_mined_library import MINED_FACTOR_SOURCE_FILE, mined_factor_rows
from app.services.factor_registry import FactorCategory, FactorDirection, list_factor_payloads

AGENT_FACTOR_SOURCE_FILE = "agent_mined_factor_library.json"


def list_single_factor_summaries(category: str | None = None) -> list[dict[str, Any]]:
    native = [_native_summary(row) for row in list_factor_payloads(category)]
    return [*native, *list_agent_factor_summaries(category)]


def list_combo_factor_summaries() -> list[dict[str, Any]]:
    rows = [
        row
        for row in _latest_mined_rows_by_name()
        if _is_combo_factor_row(row) and not _has_combo_member(row)
    ]
    return [_mined_summary(row) for row in rows]


def list_agent_factor_summaries(category: str | None = None) -> list[dict[str, Any]]:
    if not _includes_agent_category(category):
        return []
    return [_agent_summary(row) for row in _latest_agent_rows_by_name()]


def _native_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(row.get("name") or ""),
        "category": str(row.get("category") or ""),
        "categoryName": str(row.get("categoryName") or row.get("category") or ""),
        "displayName": str(row.get("displayName") or row.get("description") or row.get("name") or ""),
        "description": str(row.get("description") or row.get("displayName") or row.get("name") or ""),
        "sourceFile": str(row.get("sourceFile") or ""),
        "direction": str(row.get("direction") or FactorDirection.NEUTRAL.value),
    }


def _agent_summary(row: dict[str, Any]) -> dict[str, Any]:
    name = str(row.get("factorName") or "")
    display = str(row.get("factorDisplayName") or row.get("displayName") or name)
    return {
        "name": name,
        "category": FactorCategory.STATISTIC.value,
        "categoryName": "统计因子",
        "displayName": display,
        "description": display,
        "sourceFile": AGENT_FACTOR_SOURCE_FILE,
        "direction": FactorDirection.NEUTRAL.value,
        "symbol": _row_symbol(row),
        "duration": str(row.get("duration") or ""),
        "promotionCount": int(row.get("promotionCount") or 0),
        "candidateStatus": str(row.get("candidateStatus") or "unknown"),
        "qualityPassed": bool(row.get("qualityPassed")),
    }


def _mined_summary(row: dict[str, Any]) -> dict[str, Any]:
    name = str(row.get("factorName") or "")
    display = str(row.get("factorDisplayName") or row.get("description") or name)
    return {
        "name": name,
        "category": FactorCategory.PERFORMANCE.value,
        "categoryName": "绩效因子",
        "displayName": display,
        "description": display,
        "sourceFile": MINED_FACTOR_SOURCE_FILE,
        "direction": FactorDirection.HIGHER_BETTER.value,
        "symbol": _row_symbol(row),
        "duration": str(row.get("duration") or ""),
        "promotionCount": int(row.get("promotionCount") or 0),
    }


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


def _agent_factor_rows() -> list[dict[str, Any]]:
    return deepcopy(load_agent_factor_library().get("factors") or [])


def _includes_agent_category(category: str | None) -> bool:
    if category is None:
        return True
    return FactorCategory(category) == FactorCategory.STATISTIC


def _is_combo_factor_row(row: dict[str, Any]) -> bool:
    members = row.get("members")
    return isinstance(members, list) and len(members) >= 2


def _has_combo_member(row: dict[str, Any]) -> bool:
    members = row.get("members")
    if not isinstance(members, list):
        return False
    return any(_is_combo_name(member.get("name")) for member in members if isinstance(member, dict))


def _is_combo_name(name: object) -> bool:
    raw = str(name or "")
    return raw.startswith("combo__") or raw.startswith("goal_combo__")


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
